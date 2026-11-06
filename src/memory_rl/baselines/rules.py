"""Hand-written memory policies (plan.md §7 Phase 5).

Every baseline implements ``MemoryPolicy`` and only picks actions inside the
masks it is given. None of them sees anything a real agent could not.

| name                 | rule                                                        | knob |
|----------------------|-------------------------------------------------------------|------|
| full_history         | never edit                                                  | —    |
| sliding_window       | prune items older than the last K turns                     | K    |
| obs_masking          | replace tool outputs older than M turns with a placeholder  | M    |
| threshold_compaction | once the prompt exceeds T: compress, then prune the oldest  | T    |
|                      | tool outputs down to T/2                                    |      |
| cadoc_rule           | CADOC online commit rule (Eq. 14), see ``CadocRule``        | M    |
| random               | random gate and actions under the masks                     | p    |
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from memory_rl.envs.policy_api import AlwaysKeepPolicy, ControllerState, Decision, ItemView
from memory_rl.memory import MemoryType, OperationKind

PLACEHOLDER = "[old observation omitted]"


@dataclass(frozen=True, slots=True)
class PlaceholderCompressor:
    """Observation masking: replace the whole text with a fixed placeholder."""

    placeholder: str = PLACEHOLDER
    name: str = "placeholder-mask"

    def compress(self, text: str) -> str | None:
        return self.placeholder


def _age(state: ControllerState, item: ItemView) -> int:
    return state.turn - item.turn_index


@dataclass
class SlidingWindow:
    k: int = 2
    name: str = "sliding_window"

    def reset(self, seed: int) -> None:
        return None

    def decide(self, state: ControllerState) -> Decision:
        actions = {
            item.memory_id: OperationKind.PRUNE
            for item in state.candidates
            if _age(state, item) > self.k
        }
        return Decision(edit=bool(actions), actions=actions)


@dataclass
class ObservationMasking:
    """Must-beat baseline (arXiv 2508.21433): keep thoughts/actions, mask old outputs."""

    m: int = 2
    name: str = "obs_masking"
    compressor: PlaceholderCompressor = field(default_factory=PlaceholderCompressor)

    def reset(self, seed: int) -> None:
        return None

    def decide(self, state: ControllerState) -> Decision:
        actions = {
            item.memory_id: OperationKind.COMPRESS
            for item in state.candidates
            if item.memory_type is MemoryType.TOOL_RESULT
            and _age(state, item) > self.m
            and OperationKind.COMPRESS in item.valid_actions
        }
        return Decision(edit=bool(actions), actions=actions)


@dataclass
class ThresholdCompaction:
    t: int = 3000
    name: str = "threshold_compaction"

    def reset(self, seed: int) -> None:
        return None

    def decide(self, state: ControllerState) -> Decision:
        if state.prompt_tokens <= self.t:
            return Decision.no_edit()
        actions: dict[str, OperationKind] = {}
        estimate = state.prompt_tokens
        for item in state.candidates:
            if OperationKind.COMPRESS in item.valid_actions and item.compression_gain > 0:
                actions[item.memory_id] = OperationKind.COMPRESS
                estimate -= item.compression_gain
        target = self.t // 2
        for item in state.candidates:  # oldest first
            if estimate <= target:
                break
            if item.memory_type is MemoryType.TOOL_RESULT:
                gain = item.compression_gain if item.memory_id in actions else 0
                actions[item.memory_id] = OperationKind.PRUNE
                estimate -= (item.span_end - item.span_start) - gain
        return Decision(edit=bool(actions), actions=actions)


@dataclass
class CadocRule:
    """CADOC-style analytical commit rule, re-implemented from Yao, Zhou & Xu (2026),
    "CADOC: Cache-Aware Dynamic Object Context for Long-Horizon Agents",
    arXiv:2609.37012, Sec. 3 (Eqs. 11–14).

    Costs are in uncached-token units with cached tokens weighted by ``w_r`` (Eq. 1);
    ``w_r`` is the pricing profile's cache-read multiplier.

    * pending objects: tool outputs older than ``m`` turns that the fixed compressor
      can still shorten; ``G_t = Σ (L_keep − L_commit)`` (Eq. 11);
    * waiting ledger: after every request that still carries them raw,
      ``W ← W + w_r · G_t`` (Eq. 12); reset to 0 after a commit;
    * shared reconstruction cost: ``Q_t = F_0 + (1 − w_r) · H_t``, with ``H_t`` the
      cached tokens after the newest pending object and before the cache frontier
      (Eq. 13); the frontier is the end of the previous prompt;
    * **commit all pending replacements if ``W_t + w_r · G_t ≥ Q_t``** (Eq. 14).

    Deviations (documented): CADOC replaces objects with retrievable "Cards"; this
    environment has no retrieval, so the replacement is the shared fixed compressor
    (originals stay in the archive only). ``F_0 = 0`` as in the paper's replay. An
    optional hard guard forces a commit when the prompt exceeds ``guard_fill`` of the
    context limit ("hard capacity limits can force an earlier commit").
    """

    m: int = 1
    f0: float = 0.0
    guard_fill: float = 0.9
    name: str = "cadoc_rule"
    ledger: float = 0.0

    def reset(self, seed: int) -> None:
        self.ledger = 0.0

    def pending(self, state: ControllerState) -> list[ItemView]:
        return [
            item
            for item in state.candidates
            if item.memory_type is MemoryType.TOOL_RESULT
            and _age(state, item) > self.m
            and OperationKind.COMPRESS in item.valid_actions
            and item.compression_gain > 0
        ]

    def decide(self, state: ControllerState) -> Decision:
        pending = self.pending(state)
        if not pending:
            return Decision.no_edit()
        w_r = state.cache_read_multiplier
        gain = float(sum(item.compression_gain for item in pending))
        newest_pending_end = max(item.span_end for item in pending)
        new_items = [item for item in state.items if item.turn_index >= state.turn - 1]
        frontier = min((i.span_start for i in new_items), default=state.prompt_tokens)
        shared = max(frontier - newest_pending_end, 0)
        q = self.f0 + (1.0 - w_r) * shared
        forced = state.prompt_tokens > self.guard_fill * state.context_limit
        if self.ledger + w_r * gain >= q or forced:
            self.ledger = 0.0
            return Decision(
                edit=True, actions={item.memory_id: OperationKind.COMPRESS for item in pending}
            )
        self.ledger += w_r * gain
        return Decision.no_edit()


@dataclass
class RandomPolicy:
    p_edit: float = 0.3
    name: str = "random"
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def decide(self, state: ControllerState) -> Decision:
        if not state.candidates or self.rng.random() >= self.p_edit:
            return Decision.no_edit()
        actions = {
            item.memory_id: item.valid_actions[int(self.rng.integers(len(item.valid_actions)))]
            for item in state.candidates
        }
        return Decision(edit=True, actions=actions)


#: method -> (constructor taking the knob, default knob grid)
BASELINES: dict[str, tuple[type, tuple[float, ...]]] = {
    "full_history": (AlwaysKeepPolicy, ()),
    "sliding_window": (SlidingWindow, (1, 2, 3, 4)),
    "obs_masking": (ObservationMasking, (0, 1, 2, 3)),
    "threshold_compaction": (ThresholdCompaction, (2500, 3000, 3500, 4000)),
    "cadoc_rule": (CadocRule, (1, 2, 3)),
    "random": (RandomPolicy, (0.1, 0.3)),
}


def build_baseline(method: str, knob: float | None = None) -> object:
    try:
        cls, grid = BASELINES[method]
    except KeyError as exc:
        raise ValueError(f"unknown baseline {method!r}; choose from {sorted(BASELINES)}") from exc
    if not grid:
        return cls()
    value = grid[0] if knob is None else knob
    if method == "random":
        return cls(p_edit=float(value))
    if method in ("sliding_window",):
        return cls(k=int(value))
    if method == "threshold_compaction":
        return cls(t=int(value))
    return cls(m=int(value))
