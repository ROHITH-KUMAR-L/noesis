"""Agent environment contract (plan.md §6.5)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Protocol

SplitName = Literal["train", "valid", "test"]


class TerminalReason(StrEnum):
    SUCCESS = "success"
    ENV_DONE = "env_done"
    STEP_LIMIT = "step_limit"
    CONTEXT_OVERFLOW = "context_overflow"
    BUDGET_VIOLATION = "budget_violation"
    FATAL_ERROR = "fatal_error"


@dataclass(frozen=True, slots=True)
class AgentObservation:
    """What the agent may see. ``instruction`` is set on reset only."""

    text: str
    instruction: str | None = None
    info: dict[str, Any] = field(default_factory=dict)


class AgentEnv(Protocol):
    name: str
    max_steps: int

    def reset(self, task_id: str, seed: int) -> AgentObservation: ...

    def step(self, action_text: str) -> tuple[AgentObservation, float, bool, dict[str, Any]]: ...

    def task_ids(self, split: SplitName) -> Sequence[str]: ...

    def few_shot(self) -> str: ...

    def system_prompt(self) -> str: ...
