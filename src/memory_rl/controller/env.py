"""The agent loop as an RL environment for the memory controller (plan.md §7 Phase 4).

One MDP step = one agent turn: controller decision → one LLM call → one env step.

    cenv = ControllerEnv(factory, reward_config)
    obs = cenv.reset(task_id, seed)
    obs, reward, done, info = cenv.step(gate, item_actions)

``item_actions[slot]`` indexes ``features.ACTIONS`` (KEEP, COMPRESS, PRUNE) and
is ignored for non-candidate slots and when ``gate == 0`` (NO_EDIT). A masked
choice raises instead of being repaired.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from memory_rl.controller.features import ACTIONS, Observation, build_observation
from memory_rl.controller.reward import RewardConfig, RewardTerms, turn_reward
from memory_rl.envs.agent_loop import AgentEpisode, InvalidDecisionError
from memory_rl.envs.policy_api import ControllerState, Decision
from memory_rl.memory import OperationKind

EpisodeFactory = Callable[[str, int], AgentEpisode]


def decision_from_arrays(
    obs: Observation, gate: int, item_actions: Sequence[int] | np.ndarray
) -> Decision:
    if gate not in (0, 1):
        raise InvalidDecisionError(f"gate must be 0 or 1, got {gate}")
    if gate == 0:
        return Decision.no_edit()
    actions: dict[str, OperationKind] = {}
    for slot, memory_id in enumerate(obs.item_ids):
        if not obs.action_mask[slot].any():
            continue
        a = int(item_actions[slot])
        if not 0 <= a < len(ACTIONS) or not obs.action_mask[slot, a]:
            raise InvalidDecisionError(f"action {a} is masked for slot {slot} ({memory_id})")
        actions[memory_id] = ACTIONS[a]
    return Decision(edit=True, actions=actions)


@dataclass(slots=True)
class EpisodeLog:
    terms: list[RewardTerms] = field(default_factory=list)

    @property
    def episode_return(self) -> float:
        return float(sum(t.total for t in self.terms))

    def totals(self) -> dict[str, float]:
        return {
            "cost": float(sum(t.cost for t in self.terms)),
            "task": float(sum(t.task for t in self.terms)),
            "overflow": float(sum(t.overflow for t in self.terms)),
        }


class ControllerEnv:
    def __init__(self, factory: EpisodeFactory, reward: RewardConfig) -> None:
        self.factory = factory
        self.reward = reward
        self.episode: AgentEpisode | None = None
        self.log = EpisodeLog()
        self._obs: Observation | None = None
        self._state: ControllerState | None = None

    def reset(self, task_id: str, seed: int) -> Observation:
        self.episode = self.factory(task_id, seed)
        self.log = EpisodeLog()
        self._state = self.episode.begin()
        self._obs = build_observation(self._state)
        return self._obs

    @property
    def state(self) -> ControllerState:
        if self._state is None:
            raise RuntimeError("no active state; call reset()")
        return self._state

    def step(
        self, gate: int, item_actions: Sequence[int] | np.ndarray
    ) -> tuple[Observation | None, float, bool, dict[str, Any]]:
        if self.episode is None or self._obs is None or self._state is None:
            raise RuntimeError("call reset() first")
        decision = decision_from_arrays(self._obs, int(gate), list(item_actions))
        next_state, trace = self.episode.advance(decision)
        terms = turn_reward(trace, self.reward, success=self.episode.success)
        self.log.terms.append(terms)
        done = next_state is None
        info: dict[str, Any] = {"terms": terms, "trace": trace}
        if done:
            self._state, self._obs = None, None
            info["terminal_reason"] = trace.terminal_reason
            info["success"] = self.episode.success
            info["episode_return"] = self.log.episode_return
            info["reward_totals"] = self.log.totals()
            return None, terms.total, True, info
        assert next_state is not None
        self._state = next_state
        self._obs = build_observation(next_state)
        return self._obs, terms.total, False, info


def random_valid_action(
    obs: Observation, rng: np.random.Generator, p_edit: float = 0.3
) -> tuple[int, np.ndarray]:
    """Random gate and per-item actions under the masks (also the 'random' baseline)."""
    gate = int(rng.random() < p_edit)
    actions = np.zeros(len(obs.item_present), dtype=np.int64)
    for slot in np.flatnonzero(obs.candidate):
        actions[slot] = int(rng.choice(np.flatnonzero(obs.action_mask[slot])))
    return gate, actions
