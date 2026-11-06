"""Phase 4 controller-MDP tests (plan.md §7 Phase 4)."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from memory_rl.controller.env import ControllerEnv, decision_from_arrays, random_valid_action
from memory_rl.controller.features import (
    GLOBAL_FEATURE_NAMES,
    GLOBAL_FEATURES,
    ITEM_FEATURE_NAMES,
    ITEM_FEATURES,
    MAX_ITEMS,
    Observation,
    build_observation,
)
from memory_rl.controller.reward import RewardConfig, calibrate_z
from memory_rl.cost import CacheAwareCostMeter, get_profile
from memory_rl.envs.agent_loop import AgentEpisode, InvalidDecisionError, LoopConfig
from memory_rl.envs.base import TerminalReason
from memory_rl.envs.policy_api import AlwaysKeepPolicy, ControllerState, Decision
from memory_rl.envs.toy import KeyHuntEnv, generate_task
from memory_rl.llm import ScriptedKeyHuntLLM
from memory_rl.memory import MemoryType, OperationKind, TestTokenizer

PROFILE = get_profile("anthropic-sonnet-5-5")
TOKENIZER = TestTokenizer()
CONFIG = LoopConfig(context_limit_tokens=100_000)


def factory(task_id: str, seed: int) -> AgentEpisode:
    return AgentEpisode(
        KeyHuntEnv(), task_id, seed=seed, llm=ScriptedKeyHuntLLM(),
        meter=CacheAwareCostMeter(PROFILE), tokenizer=TOKENIZER, profile=PROFILE, config=CONFIG,
    )  # fmt: skip


def run_policy(policy: object, task_id: str) -> AgentEpisode:
    return factory(task_id, 0).run(policy)  # type: ignore[arg-type]


class PruneOldestAssistantEachTurn:
    """'Trim a little every turn': prunes the oldest thought/action already sent.

    (Pruning the thought generated on the previous turn is free for the cache: it
    was output, never part of a sent prompt. Trimming history that *was* sent is
    what re-bills the suffix.)
    """

    name = "trim_every_turn"

    def reset(self, seed: int) -> None:
        return None

    def decide(self, state: ControllerState) -> Decision:
        for item in state.candidates:
            if item.memory_type is MemoryType.ASSISTANT and item.turn_index < state.turn - 1:
                return Decision(edit=True, actions={item.memory_id: OperationKind.PRUNE})
        return Decision.no_edit()


# --- observations ---------------------------------------------------------------


def test_observation_shapes_are_fixed_and_finite() -> None:
    cenv = ControllerEnv(factory, RewardConfig("cache", z=0.01))
    obs: Observation | None = cenv.reset("keyhunt-train-0003", 0)
    rng = np.random.default_rng(0)
    while obs is not None:
        assert obs.items.shape == (MAX_ITEMS, ITEM_FEATURES)
        assert obs.global_features.shape == (GLOBAL_FEATURES,)
        assert np.isfinite(obs.items).all() and np.isfinite(obs.global_features).all()
        assert obs.items.max() <= 1.5 and obs.items.min() >= 0.0
        obs, _, _, _ = cenv.step(*random_valid_action(obs, rng))
    assert len(ITEM_FEATURE_NAMES) == ITEM_FEATURES and len(GLOBAL_FEATURE_NAMES) == GLOBAL_FEATURES


def test_feature_names_carry_no_simulator_labels() -> None:
    for name in (*ITEM_FEATURE_NAMES, *GLOBAL_FEATURE_NAMES):
        for word in ("critical", "gold", "key", "answer", "success", "required"):
            assert word not in name


def test_observation_independent_of_gold_labels() -> None:
    """Changing the gold keys (not visible yet) leaves the first observation unchanged."""
    env = KeyHuntEnv()
    ep = AgentEpisode(
        env, "keyhunt-valid-0002", seed=0, llm=ScriptedKeyHuntLLM(),
        meter=CacheAwareCostMeter(PROFILE), tokenizer=TOKENIZER, profile=PROFILE, config=CONFIG,
    )  # fmt: skip
    state = ep.begin()
    base = build_observation(state)
    task = env.task
    env._task = replace(task, keys=tuple("K-0000" for _ in task.keys))  # tamper with gold
    again = build_observation(ep.state())
    assert np.array_equal(base.items, again.items)
    assert np.array_equal(base.global_features, again.global_features)
    assert generate_task("keyhunt-valid-0002").keys == task.keys


# --- masks and decisions ----------------------------------------------------------


def test_masks_exclude_pruned_newest_and_protected() -> None:
    cenv = ControllerEnv(factory, RewardConfig("cache", z=0.01))
    obs: Observation | None = cenv.reset("keyhunt-train-0004", 0)
    obs, *_ = cenv.step(0, np.zeros(MAX_ITEMS, dtype=int))
    obs, *_ = cenv.step(0, np.zeros(MAX_ITEMS, dtype=int))
    assert obs is not None
    n = int(obs.item_present.sum())
    assert not obs.candidate[n - 1]  # newest item protected
    assert not obs.candidate[n:].any()  # padding
    target = int(np.flatnonzero(obs.candidate)[0])
    actions = np.zeros(MAX_ITEMS, dtype=int)
    actions[target] = 2  # PRUNE
    pruned_id = obs.item_ids[target]
    obs, *_ = cenv.step(1, actions)
    assert obs is not None and pruned_id not in obs.item_ids


def test_masked_action_raises() -> None:
    cenv = ControllerEnv(factory, RewardConfig("cache", z=0.01))
    obs: Observation | None = cenv.reset("keyhunt-train-0005", 0)
    for _ in range(2):
        obs, *_ = cenv.step(0, np.zeros(MAX_ITEMS, dtype=int))
    assert obs is not None
    slot = int(np.flatnonzero(obs.action_mask[:, 1])[0])
    memory_id = obs.item_ids[slot]
    compress = np.zeros(MAX_ITEMS, dtype=int)
    compress[slot] = 1
    obs, *_ = cenv.step(1, compress)
    assert obs is not None
    slot = obs.item_ids.index(memory_id)
    assert not obs.action_mask[slot, 1]  # compress at most once
    bad = np.zeros(MAX_ITEMS, dtype=int)
    bad[slot] = 1  # COMPRESS where masked
    with pytest.raises(InvalidDecisionError):
        decision_from_arrays(obs, 1, bad)
    with pytest.raises(InvalidDecisionError):
        decision_from_arrays(obs, 2, bad)


# --- reward ---------------------------------------------------------------------


def test_reward_terms_sum_to_return_and_terminal_has_task_reward() -> None:
    cenv = ControllerEnv(factory, RewardConfig("cache", z=0.01, lam=2.0))
    obs: Observation | None = cenv.reset("keyhunt-train-0006", 0)
    total, info = 0.0, cast(dict[str, Any], {})
    while obs is not None:
        obs, reward, done, info = cenv.step(0, np.zeros(MAX_ITEMS, dtype=int))
        total += reward
    assert math.isclose(total, info["episode_return"])
    totals = info["reward_totals"]
    assert math.isclose(total, totals["cost"] + totals["task"] + totals["overflow"])
    assert totals["task"] == 1.0 and info["success"] is True
    assert cenv.episode is not None
    cost = sum(float(t.cost["total_cost"]) for t in cenv.episode.turns if t.cost)
    assert math.isclose(totals["cost"], -2.0 * cost / 0.01)


def test_overflow_penalty_is_logged_separately() -> None:
    def tight(task_id: str, seed: int) -> AgentEpisode:
        return AgentEpisode(
            KeyHuntEnv(), task_id, seed=seed, llm=ScriptedKeyHuntLLM(),
            meter=CacheAwareCostMeter(PROFILE), tokenizer=TOKENIZER, profile=PROFILE,
            config=LoopConfig(context_limit_tokens=2500),
        )  # fmt: skip

    cenv = ControllerEnv(tight, RewardConfig("cache", z=0.01, overflow_penalty=0.5))
    obs: Observation | None = cenv.reset("keyhunt-train-0001", 0)
    info: dict[str, object] = {}
    while obs is not None:
        obs, _, _, info = cenv.step(0, np.zeros(MAX_ITEMS, dtype=int))
    assert info["terminal_reason"] == TerminalReason.CONTEXT_OVERFLOW
    assert info["reward_totals"] == pytest.approx(  # type: ignore[comparison-overlap]
        {"cost": info["reward_totals"]["cost"], "task": 0.0, "overflow": -0.5}  # type: ignore[index]
    )


def test_reward_config_validation() -> None:
    with pytest.raises(ValueError):
        RewardConfig("dollars", z=1.0)
    with pytest.raises(ValueError):
        RewardConfig("cache", z=0.0)


def test_always_keep_equals_full_history_cost_and_calibration() -> None:
    ids = list(KeyHuntEnv().task_ids("train"))[:4]
    z = calibrate_z(
        KeyHuntEnv(), llm=ScriptedKeyHuntLLM(), tokenizer=TOKENIZER, profile=PROFILE,
        variant="cache", task_ids=ids, loop_config=CONFIG,
    )  # fmt: skip
    manual = [run_policy(AlwaysKeepPolicy(), t).record(method="fh")["total_cost"] for t in ids]
    assert z == pytest.approx(sum(manual) / len(manual))


def test_h2_mechanism_trimming_costs_more_dollars_but_fewer_tokens() -> None:
    """Prune-one-item-every-turn: fewer prompt tokens, higher cache-adjusted bill."""
    worse = 0
    for task_id in list(KeyHuntEnv().task_ids("valid"))[:20]:
        keep = run_policy(AlwaysKeepPolicy(), task_id).record(method="fh")
        trim = run_policy(PruneOldestAssistantEachTurn(), task_id).record(method="trim")
        assert trim["success"] and keep["success"]
        assert trim["token_cost"] < keep["token_cost"]
        worse += trim["total_cost"] > keep["total_cost"]
    assert worse == 20


# --- exit gate --------------------------------------------------------------------


def test_exit_gate_1000_random_policy_episodes() -> None:
    rng = np.random.default_rng(1234)
    cenv = ControllerEnv(factory, RewardConfig("cache", z=0.01))
    train = list(KeyHuntEnv().task_ids("train"))
    reasons: dict[str, int] = {}
    for i in range(1000):
        task_id = train[i % len(train)]
        obs: Observation | None = cenv.reset(task_id, seed=i)
        assert cenv.episode is not None and cenv.episode.store.size == 0  # clean reset
        steps, done, info = 0, False, cast(dict[str, Any], {})
        while not done:
            assert obs is not None
            obs, reward, done, info = cenv.step(*random_valid_action(obs, rng, p_edit=0.4))
            assert math.isfinite(reward)
            steps += 1
            assert steps <= cenv.episode.env.max_steps
        episode = cenv.episode
        prompt = episode.current_prompt()
        active = sum(item.token_count for item in episode.store.active_items())
        assert active <= prompt.memory_tokens  # counts are consistent
        reasons[info["terminal_reason"]] = reasons.get(info["terminal_reason"], 0) + 1
    assert sum(reasons.values()) == 1000
    assert set(reasons) <= {r.value for r in TerminalReason}
