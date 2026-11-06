"""What a memory policy sees and returns (plan.md §6.5–6.6).

``ControllerState`` contains only inference-available information: item text,
type, sizes, positions, masks and pricing multipliers. No gold labels.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from memory_rl.memory import MemoryStatus, MemoryType, OperationKind


@dataclass(frozen=True, slots=True)
class ItemView:
    memory_id: str
    memory_type: MemoryType
    status: MemoryStatus
    text: str
    token_count: int  # tokens of the item text
    turn_index: int
    compression_count: int
    span_start: int  # first prompt token of the rendered message
    span_end: int
    valid_actions: tuple[OperationKind, ...]  # () = not a candidate this turn
    compression_gain: int  # tokens saved by COMPRESS (0 when masked)

    @property
    def is_candidate(self) -> bool:
        return bool(self.valid_actions)


@dataclass(frozen=True, slots=True)
class ControllerState:
    turn: int
    max_turns: int
    instruction: str
    items: tuple[ItemView, ...]
    prompt_tokens: int  # current prompt (before this turn's edit, before the LLM call)
    prefix_tokens: int
    context_limit: int
    last_cached_fraction: float
    cache_read_multiplier: float
    cache_write_multiplier: float
    cache_mode: str
    token_cap: int | None = None  # Phase 7 per-call cap, if any

    @property
    def candidates(self) -> tuple[ItemView, ...]:
        return tuple(item for item in self.items if item.is_candidate)

    def suffix_tokens_after(self, item: ItemView) -> int:
        return self.prompt_tokens - item.span_start


@dataclass(frozen=True, slots=True)
class Decision:
    """``edit=False`` means every item is KEEP. Missing ids under EDIT default to KEEP."""

    edit: bool
    actions: Mapping[str, OperationKind] = field(default_factory=dict)

    @classmethod
    def no_edit(cls) -> Decision:
        return cls(edit=False)


class MemoryPolicy(Protocol):
    @property
    def name(self) -> str: ...

    def reset(self, seed: int) -> None: ...

    def decide(self, state: ControllerState) -> Decision: ...


class AlwaysKeepPolicy:
    """Full history: never edit."""

    name = "full_history"

    def reset(self, seed: int) -> None:
        return None

    def decide(self, state: ControllerState) -> Decision:
        return Decision.no_edit()
