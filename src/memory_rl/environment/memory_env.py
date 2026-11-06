"""Finite, leakage-safe memory-management environment (plan.md §7 Phase 2).

One episode is one pass over the task history: each item receives exactly one
KEEP/COMPRESS/PRUNE decision, in history order, and the episode then terminates.
The environment never enforces the budget by deleting or truncating anything —
over-budget and over-context examples are recorded as violations.

Gymnasium-like API without a Gymnasium dependency::

    observation = env.reset(task, seed=0)
    observation, reward, terminated, info = env.step(action)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from memory_rl.answering.interface import Answerer, exact_match
from memory_rl.environment.features import (
    FEATURE_DIM,
    empty_observation,
    extract_features,
)
from memory_rl.environment.reward import Phase2SuccessReward, RewardBreakdown
from memory_rl.environment.tasks import SyntheticTask
from memory_rl.memory import (
    Compressor,
    MemoryItem,
    MemoryStore,
    OperationKind,
    PromptSerializer,
    RuleBasedCompressor,
    Tokenizer,
)

DEFAULT_SYSTEM_INSTRUCTIONS = (
    "You are given a memory of earlier messages and a task. "
    "Answer the task using only that memory."
)

ACTION_ORDER: tuple[OperationKind, ...] = (
    OperationKind.KEEP,
    OperationKind.COMPRESS,
    OperationKind.PRUNE,
)
_ACTION_INDEX = {operation: index for index, operation in enumerate(ACTION_ORDER)}


@dataclass(frozen=True, slots=True)
class EnvConfig:
    """Episode configuration. Budget is a fraction of original memory tokens."""

    budget_fraction: float = 0.5
    context_limit_tokens: int = 8192
    compressor: Compressor = field(default_factory=RuleBasedCompressor)

    def __post_init__(self) -> None:
        if not 0.0 < self.budget_fraction <= 1.0:
            raise ValueError("budget_fraction must be in (0, 1]")
        if self.context_limit_tokens <= 0:
            raise ValueError("context_limit_tokens must be > 0")


class MemoryEnv:
    """Deterministic one-pass memory-management episode."""

    def __init__(
        self,
        *,
        tokenizer: Tokenizer,
        answerer: Answerer,
        config: EnvConfig | None = None,
        system_instructions: str = DEFAULT_SYSTEM_INSTRUCTIONS,
        reward: Phase2SuccessReward | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._answerer = answerer
        self._config = config or EnvConfig()
        self._serializer = PromptSerializer(tokenizer, system_instructions)
        self._reward = reward or Phase2SuccessReward()

        self._task: SyntheticTask | None = None
        self._store: MemoryStore | None = None
        self._order: list[str] = []
        self._cursor = 0
        self._decided = 0
        self._terminated = True
        self._termination_reason: str | None = None
        self._budget_tokens = 0
        self._original_memory_tokens = 0
        self._max_item_tokens = 1
        self._seed: int | None = None
        self._reward_breakdown: RewardBreakdown | None = None
        self._terminal_diagnostics: dict[str, Any] = {}

    # --- episode lifecycle ----------------------------------------------------

    def reset(self, task: SyntheticTask, *, seed: int | None = None) -> NDArray[np.float32]:
        """Start a fresh episode. All episode state is rebuilt (no leakage)."""
        items = [
            MemoryItem.create(
                memory_id=turn.memory_id,
                text=turn.text,
                tokenizer=self._tokenizer,
                turn_index=turn.turn_index,
                memory_type=turn.memory_type,
            )
            for turn in task.history
        ]
        self._task = task
        self._store = MemoryStore(items, tokenizer=self._tokenizer)
        self._order = [item.memory_id for item in items]
        self._cursor = 0
        self._decided = 0
        self._seed = seed
        self._reward_breakdown = None
        self._terminal_diagnostics = {}
        self._termination_reason = None
        self._original_memory_tokens = self._store.active_token_count
        self._budget_tokens = max(
            1, math.ceil(self._config.budget_fraction * self._original_memory_tokens)
        )
        self._max_item_tokens = max((item.token_count for item in items), default=1) or 1

        if not self._order:
            self._terminated = True
            self._termination_reason = "empty_history"
            return empty_observation()
        self._terminated = False
        return self._observation()

    def step(
        self, action: OperationKind | int
    ) -> tuple[NDArray[np.float32], float, bool, dict[str, Any]]:
        """Apply one action to the current item and advance."""
        if self._terminated:
            raise RuntimeError("episode already terminated; call reset() before step()")

        operation = self._coerce_action(action)
        item = self._current_item()
        if not self.action_mask()[_ACTION_INDEX[operation]]:
            valid = ", ".join(op.value for op in self.valid_actions())
            raise ValueError(
                f"invalid action {operation.value!r} for item {item.memory_id!r}; valid: {valid}"
            )

        store = self._require_store()
        result = store.apply(
            operation,
            item.memory_id,
            compressor=self._config.compressor if operation is OperationKind.COMPRESS else None,
        )
        if not result.success:  # the mask should make this unreachable; never hide it
            raise RuntimeError(
                f"masked action failed for {item.memory_id!r}: {result.reason}"
            )

        self._decided += 1
        self._cursor += 1
        terminated = self._advance_or_finish()

        info: dict[str, Any] = {
            "step": self._decided,
            "item_id": item.memory_id,
            "operation": operation.value,
            "operation_reason": result.reason,
            "token_delta": result.token_delta,
            "memory_tokens": self.memory_tokens,
            "budget_tokens": self._budget_tokens,
            "decided_candidates": self._decided,
            "total_candidates": len(self._order),
            "episode_progress": self._decided / max(len(self._order), 1),
            "terminated": terminated,
            "next_action_mask": self.action_mask(),
        }
        if terminated:
            info.update(self._terminal_diagnostics)

        reward = self._reward_breakdown.total if terminated and self._reward_breakdown else 0.0
        observation = empty_observation() if terminated else self._observation()
        return observation, reward, terminated, info

    # --- inspection -----------------------------------------------------------

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def termination_reason(self) -> str | None:
        return self._termination_reason

    @property
    def seed(self) -> int | None:
        return self._seed

    @property
    def memory_tokens(self) -> int:
        return self._require_store().active_token_count

    @property
    def original_memory_tokens(self) -> int:
        return self._original_memory_tokens

    @property
    def budget_tokens(self) -> int:
        return self._budget_tokens

    @property
    def action_space_size(self) -> int:
        return len(ACTION_ORDER)

    @property
    def reward_breakdown(self) -> RewardBreakdown | None:
        return self._reward_breakdown

    def valid_actions(self) -> tuple[OperationKind, ...]:
        """Valid actions for the current item, in canonical order."""
        if self._terminated or not self._order:
            return ()
        return self._require_store().valid_actions(
            self._current_item().memory_id, self._config.compressor
        )

    def action_mask(self) -> NDArray[np.bool_]:
        """Boolean mask aligned with ``ACTION_ORDER`` (KEEP, COMPRESS, PRUNE)."""
        mask = np.zeros(len(ACTION_ORDER), dtype=np.bool_)
        for operation in self.valid_actions():
            mask[_ACTION_INDEX[operation]] = True
        return mask

    def serialize_active_memory(self) -> Any:
        """The exact prompt a fixed answerer would receive right now."""
        return self._serializer.serialize(
            self._require_store().active_items(), self._require_task().question
        )

    # --- internals ------------------------------------------------------------

    def _advance_or_finish(self) -> bool:
        if self._cursor >= len(self._order):
            self._finish("completed_all_candidates")
            return True
        if not self.action_mask().any():
            self._finish("no_valid_action")
            return True
        return False

    def _finish(self, reason: str) -> None:
        task = self._require_task()
        store = self._require_store()
        prompt = self._serializer.serialize(store.active_items(), task.question)
        answer = self._answerer.answer(prompt, task.question)
        correct = exact_match(answer, task.labels.reference_answer)

        retained = tuple(
            fact.memory_id
            for fact in task.labels.required_facts
            if fact.text in prompt.memory_text
        )
        lost = tuple(
            fact.memory_id
            for fact in task.labels.required_facts
            if fact.text not in prompt.memory_text
        )
        budget_compliant = store.active_token_count <= self._budget_tokens
        context_ok = prompt.total_prompt_tokens <= self._config.context_limit_tokens

        self._reward_breakdown = self._reward.compute(
            correct=correct,
            memory_tokens=store.active_token_count,
            budget_tokens=self._budget_tokens,
            critical_facts_lost=len(lost),
            budget_compliant=budget_compliant,
        )
        self._terminal_diagnostics = {
            "termination_reason": reason,
            "answer": answer,
            "reference_answer": task.labels.reference_answer,
            "correct": correct,
            "memory_tokens": store.active_token_count,
            "total_prompt_tokens": prompt.total_prompt_tokens,
            "context_limit_tokens": self._config.context_limit_tokens,
            "context_ok": context_ok,
            "budget_compliant": budget_compliant,
            "critical_facts_retained": retained,
            "critical_facts_lost": lost,
            "reward_breakdown": self._reward_breakdown,
        }
        self._terminated = True
        self._termination_reason = reason

    def _observation(self) -> NDArray[np.float32]:
        task = self._require_task()
        item = self._current_item()
        return extract_features(
            item,
            question=task.question,
            turn_index=item.turn_index,
            episode_length=len(self._order),
            max_item_tokens=self._max_item_tokens,
            active_tokens=self.memory_tokens,
            budget_tokens=self._budget_tokens,
            decided_candidates=self._decided,
            total_candidates=len(self._order),
        )

    @property
    def current_item_id(self) -> str | None:
        """ID of the item awaiting a decision, or None once the episode ended."""
        if self._termination_reason is not None or self._cursor >= len(self._order):
            return None
        return self._order[self._cursor]

    def _current_item(self) -> MemoryItem:
        return self._require_store().get(self._order[self._cursor])

    def _require_store(self) -> MemoryStore:
        if self._store is None:
            raise RuntimeError("call reset() before using the environment")
        return self._store

    def _require_task(self) -> SyntheticTask:
        if self._task is None:
            raise RuntimeError("call reset() before using the environment")
        return self._task

    @staticmethod
    def _coerce_action(action: OperationKind | int) -> OperationKind:
        if isinstance(action, OperationKind):
            return action
        if isinstance(action, bool) or not isinstance(action, int):
            raise ValueError(f"invalid action {action!r}")
        if not 0 <= action < len(ACTION_ORDER):
            raise ValueError(
                f"invalid action index {action!r}; expected 0..{len(ACTION_ORDER) - 1}"
            )
        return ACTION_ORDER[action]


__all__ = [
    "ACTION_ORDER",
    "DEFAULT_SYSTEM_INSTRUCTIONS",
    "EnvConfig",
    "FEATURE_DIM",
    "MemoryEnv",
]
