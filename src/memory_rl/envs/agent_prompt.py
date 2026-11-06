"""Agent prompt serializer (plan.md §6.2).

Order: [system] → [task instruction + few-shot] → history items in turn order.
Each message is rendered with a fixed chat template and encoded *separately*, so
the token IDs of message *i* never depend on later messages: an edit to item *i*
changes only tokens from *i* onward, which is what makes prefix caching work.

The cost model counts these exact IDs. When the frozen LLM is a remote API with
its own tokenizer, provider-reported usage is recorded separately (agent loop)
and the simulator's counts are an approximation of the provider's.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from memory_rl.memory import EncodingTokenizer, MemoryItem, MemoryStatus, MemoryType

CHAT_TEMPLATE = "<|im_start|>{role}\n{content}<|im_end|>\n"
OBSERVATION_PREFIX = "Observation: "


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class AgentPrompt:
    messages: tuple[ChatMessage, ...]
    token_ids: tuple[int, ...]
    prefix_tokens: int
    memory_tokens: int
    total_prompt_tokens: int
    included_ids: tuple[str, ...]
    item_spans: dict[str, tuple[int, int]]  # memory_id -> [start, end) token positions

    def suffix_tokens_after(self, memory_id: str) -> int:
        """Tokens from the start of ``memory_id`` to the end (re-billed if it is edited)."""
        start, _ = self.item_spans[memory_id]
        return self.total_prompt_tokens - start


def item_message(item: MemoryItem) -> ChatMessage:
    if item.memory_type is MemoryType.ASSISTANT:
        return ChatMessage("assistant", item.text)
    if item.memory_type is MemoryType.TOOL_RESULT:
        return ChatMessage("user", OBSERVATION_PREFIX + item.text)
    raise ValueError(f"agent history supports ASSISTANT/TOOL_RESULT items, got {item.memory_type}")


class AgentPromptSerializer:
    """Shared by every method so prompt layout and token accounting match."""

    def __init__(self, tokenizer: EncodingTokenizer, system_prompt: str) -> None:
        if not system_prompt.strip():
            raise ValueError("system_prompt must be non-empty")
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt

    def _encode(self, message: ChatMessage) -> list[int]:
        return self.tokenizer.encode(
            CHAT_TEMPLATE.format(role=message.role, content=message.content)
        )

    def protected_messages(self, instruction: str, few_shot: str) -> tuple[ChatMessage, ...]:
        task = f"{few_shot}\n\nNow solve this task.\n{instruction}" if few_shot else instruction
        return (ChatMessage("system", self.system_prompt), ChatMessage("user", task))

    def serialize(
        self, instruction: str, few_shot: str, items: Sequence[MemoryItem]
    ) -> AgentPrompt:
        ids = [item.memory_id for item in items]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate memory items passed to the serializer")
        pruned = [item.memory_id for item in items if item.status is MemoryStatus.PRUNED]
        if pruned:
            raise ValueError(f"pruned items must never be serialized: {pruned}")

        messages = list(self.protected_messages(instruction, few_shot))
        token_ids: list[int] = []
        for message in messages:
            token_ids.extend(self._encode(message))
        prefix = len(token_ids)
        spans: dict[str, tuple[int, int]] = {}
        for item in items:
            message = item_message(item)
            start = len(token_ids)
            token_ids.extend(self._encode(message))
            spans[item.memory_id] = (start, len(token_ids))
            messages.append(message)
        return AgentPrompt(
            messages=tuple(messages),
            token_ids=tuple(token_ids),
            prefix_tokens=prefix,
            memory_tokens=len(token_ids) - prefix,
            total_prompt_tokens=len(token_ids),
            included_ids=tuple(ids),
            item_spans=spans,
        )
