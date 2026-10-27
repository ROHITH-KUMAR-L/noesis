"""The single shared prompt serializer (plan.md §6.2).

Every method must serialize through this class so token counts are comparable.
The system instructions and the current question are owned here, never by the
store, so they can never become action candidates (plan.md §4.2.2).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from memory_rl.memory.models import MemoryItem, MemoryStatus
from memory_rl.memory.tokenizer import Tokenizer

DEFAULT_ITEM_TEMPLATE = "[{memory_type}] {text}"
MEMORY_HEADER = "MEMORY:"
QUESTION_HEADER = "TASK:"


@dataclass(frozen=True, slots=True)
class SerializedPrompt:
    """Exact serialized input plus its token accounting.

    ``fixed_text`` is the same skeleton with the memory block left empty, so
    ``fixed_prompt_tokens`` measures instructions, headers, and the question.
    ``total_prompt_tokens`` is always the count of ``text`` itself.
    """

    text: str
    memory_text: str
    fixed_text: str
    memory_tokens: int
    fixed_prompt_tokens: int
    total_prompt_tokens: int
    included_ids: tuple[str, ...]


class PromptSerializer:
    """Deterministic prompt builder shared by every policy and baseline."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        system_instructions: str,
        *,
        item_template: str = DEFAULT_ITEM_TEMPLATE,
        memory_header: str = MEMORY_HEADER,
        question_header: str = QUESTION_HEADER,
    ) -> None:
        if not system_instructions.strip():
            raise ValueError("system_instructions must be non-empty")
        for placeholder in ("{memory_type}", "{text}"):
            if placeholder not in item_template:
                raise ValueError(f"item_template must contain {placeholder}")
        self._tokenizer = tokenizer
        self._system_instructions = system_instructions
        self._item_template = item_template
        self._memory_header = memory_header
        self._question_header = question_header

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    @property
    def system_instructions(self) -> str:
        return self._system_instructions

    def fixed_text(self, question: str) -> str:
        """Instructions, headers, and question, with no memory block."""
        return (
            f"{self._system_instructions}\n\n{self._memory_header}\n\n"
            f"{self._question_header}\n{question}"
        )

    def serialize(self, items: Sequence[MemoryItem], question: str) -> SerializedPrompt:
        """Serialize active memory and the question into the answerer input."""
        if not question.strip():
            raise ValueError("the current question must be non-empty")

        ids = tuple(item.memory_id for item in items)
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate memory items passed to the serializer")

        pruned = [item.memory_id for item in items if item.status is MemoryStatus.PRUNED]
        if pruned:
            raise ValueError(f"pruned items must never be serialized: {pruned}")

        memory_text = "\n".join(
            self._item_template.format(memory_type=item.memory_type.value, text=item.text)
            for item in items
        )
        fixed = self.fixed_text(question)
        if memory_text:
            text = (
                f"{self._system_instructions}\n\n{self._memory_header}\n{memory_text}\n\n"
                f"{self._question_header}\n{question}"
            )
        else:
            text = fixed

        tokenizer = self._tokenizer
        return SerializedPrompt(
            text=text,
            memory_text=memory_text,
            fixed_text=fixed,
            memory_tokens=tokenizer.count(memory_text),
            fixed_prompt_tokens=tokenizer.count(fixed),
            total_prompt_tokens=tokenizer.count(text),
            included_ids=ids,
        )
