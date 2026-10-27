"""Typed memory model shared by the store, operations, and prompt builder.

Simulator-only labels (``is_critical``, ``importance``, ``required_fact_ids``,
``reference_answer``) belong to task objects, never to memory items. This module
enforces that at construction time (plan.md §4.2.3, §6.1).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memory_rl.memory.tokenizer import Tokenizer


SIMULATOR_ONLY_LABELS: frozenset[str] = frozenset(
    {"is_critical", "importance", "required_fact_ids", "reference_answer"}
)


class MemoryType(StrEnum):
    """Where a memory unit came from. ``SYSTEM`` is never an action candidate."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL_RESULT = "tool_result"
    FACT = "fact"
    DECISION = "decision"
    SYSTEM = "system"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    COMPRESSED = "compressed"
    PRUNED = "pruned"


class OperationKind(StrEnum):
    """Canonical action order: KEEP, COMPRESS, PRUNE."""

    KEEP = "keep"
    COMPRESS = "compress"
    PRUNE = "prune"


@dataclass(frozen=True, slots=True)
class MemoryItem:
    """One unit of memory.

    ``original_text`` is immutable and kept for diagnostics only; the answerer
    only ever sees ``text`` while the item is active.
    """

    memory_id: str
    text: str
    original_text: str
    token_count: int
    turn_index: int
    memory_type: MemoryType
    status: MemoryStatus = MemoryStatus.ACTIVE
    compression_count: int = 0
    dependencies: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.memory_id:
            raise ValueError("memory_id must be a non-empty string")
        if not isinstance(self.memory_type, MemoryType):
            raise TypeError(f"memory_type must be a MemoryType, got {self.memory_type!r}")
        if not isinstance(self.status, MemoryStatus):
            raise TypeError(f"status must be a MemoryStatus, got {self.status!r}")
        if self.token_count < 0:
            raise ValueError("token_count must be >= 0")
        if self.compression_count < 0:
            raise ValueError("compression_count must be >= 0")
        if self.compression_count > 0 and self.status is MemoryStatus.ACTIVE:
            raise ValueError("an item with compression_count > 0 cannot still be ACTIVE")
        leaked = SIMULATOR_ONLY_LABELS.intersection(self.metadata)
        if leaked:
            raise ValueError(
                "memory metadata must be inference-available only; "
                f"remove simulator-only labels: {sorted(leaked)}"
            )

    @property
    def is_active(self) -> bool:
        """True while the item contributes to the answerer prompt."""
        return self.status is not MemoryStatus.PRUNED

    @property
    def is_system(self) -> bool:
        return self.memory_type is MemoryType.SYSTEM

    @classmethod
    def create(
        cls,
        *,
        memory_id: str,
        text: str,
        tokenizer: Tokenizer,
        turn_index: int = 0,
        memory_type: MemoryType = MemoryType.USER,
        dependencies: tuple[str, ...] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> MemoryItem:
        """Build an item and count its tokens with the configured tokenizer."""
        return cls(
            memory_id=memory_id,
            text=text,
            original_text=text,
            token_count=tokenizer.count(text),
            turn_index=turn_index,
            memory_type=memory_type,
            dependencies=dependencies,
            metadata=dict(metadata or {}),
        )


@dataclass(frozen=True, slots=True)
class OperationResult:
    """Outcome of one KEEP/COMPRESS/PRUNE attempt.

    Token fields describe *active* usage: a pruned item contributes 0 tokens, so
    ``token_delta`` is negative for a prune and 0 for a failed operation.
    """

    memory_id: str
    operation: OperationKind
    success: bool
    changed: bool
    reason: str
    previous_status: MemoryStatus
    new_status: MemoryStatus
    previous_token_count: int
    new_token_count: int
    token_delta: int
    item: MemoryItem

    @property
    def active_token_savings(self) -> int:
        return max(0, -self.token_delta)
