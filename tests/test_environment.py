"""Phase 2 environment tests: finite episodes, budgets, masks, and loss (plan.md §7).

Exit criteria covered here: 1,000 random valid episodes without hangs, invalid
transitions, non-finite observations/rewards, inconsistent token counts, or state
leakage between resets; and a constructed task whose critical fact is lost and
measured as lost.
"""

from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest

from memory_rl.answering import SyntheticExactMatchAnswerer
from memory_rl.environment import (
    FEATURE_DIM,
    EnvConfig,
    MemoryEnv,
    Split,
    SyntheticTask,
    TaskFamily,
    TaskLabels,
    generate_dataset,
    templates_for,
)
from memory_rl.memory import IdentityCompressor, OperationKind, TestTokenizer

TOKENIZER = TestTokenizer()


def build_env(**overrides: object) -> MemoryEnv:
    config = EnvConfig(**overrides)  # type: ignore[arg-type]
    return MemoryEnv(
        tokenizer=TOKENIZER,
        answerer=SyntheticExactMatchAnswerer(),
        config=config,
    )


def one_task(family: TaskFamily = TaskFamily.FACT_RECALL, split: Split = Split.TRAIN) -> SyntheticTask:
    return generate_dataset(family=family, split=split, count=1, seed=3)[0]


def run_all_keep(env: MemoryEnv, task: SyntheticTask) -> dict[str, object]:
    env.reset(task)
    info: dict[str, object] = {}
    terminated = False
    while not terminated:
        _, _, terminated, info = env.step(OperationKind.KEEP)
    return info


# --- task generation ----------------------------------------------------------


def test_generation_is_deterministic_per_seed() -> None:
    first = generate_dataset(split=Split.TRAIN, count=6, seed=42)
    second = generate_dataset(split=Split.TRAIN, count=6, seed=42)

    assert first == second
    assert all(task.seed == 42 for task in first)


def test_different_seeds_vary_values_and_order() -> None:
    first = generate_dataset(family=TaskFamily.FACT_RECALL, split=Split.TRAIN, count=8, seed=1)
    second = generate_dataset(family=TaskFamily.FACT_RECALL, split=Split.TRAIN, count=8, seed=2)

    first_answers = {task.labels.reference_answer for task in first}
    second_answers = {task.labels.reference_answer for task in second}
    assert first_answers != second_answers
    assert [t.question for t in first] == [t.question for t in second]  # same template


def test_splits_use_disjoint_templates_and_cover_every_family() -> None:
    splits = {
        split: {template.template_id for template in templates_for(split=split)}
        for split in Split
    }
    assert splits[Split.TRAIN].isdisjoint(splits[Split.VALIDATION])
    assert splits[Split.TRAIN].isdisjoint(splits[Split.TEST])
    assert splits[Split.VALIDATION].isdisjoint(splits[Split.TEST])

    for split in Split:
        families = {template.family for template in templates_for(split=split)}
        assert families == set(TaskFamily)


def test_history_length_and_critical_position_vary() -> None:
    tasks = generate_dataset(family=TaskFamily.DISTRACTOR_HEAVY, split=Split.TRAIN, count=10, seed=5)
    lengths = {len(task.history) for task in tasks}
    positions = {
        task.labels.required_facts[0].memory_id for task in tasks
    }
    assert len(lengths) > 1
    assert len(positions) > 1


def test_required_facts_point_at_real_history_items() -> None:
    for template in templates_for():
        task = generate_dataset(
            family=template.family, split=template.split, count=1, seed=9
        )[0]
        ids = {turn.memory_id for turn in task.history}
        for fact in task.labels.required_facts:
            assert fact.memory_id in ids
            assert fact.text


# --- episode mechanics --------------------------------------------------------


def test_each_item_is_decided_exactly_once_in_history_order() -> None:
    task = one_task()
    env = build_env()
    env.reset(task)

    decided: list[str] = []
    terminated = False
    while not terminated:
        _, _, terminated, info = env.step(OperationKind.KEEP)
        decided.append(str(info["item_id"]))

    assert decided == [turn.memory_id for turn in task.history]
    assert env.termination_reason == "completed_all_candidates"


def test_action_mask_matches_valid_actions_and_rejects_invalid() -> None:
    task = one_task()
    env = build_env()
    env.reset(task)

    mask = env.action_mask()
    assert mask.dtype == np.bool_
    assert mask.shape == (3,)
    valid = env.valid_actions()
    assert list(mask) == [operation in valid for operation in OperationKind]

    invalid = next(operation for operation in OperationKind if operation not in valid)
    with pytest.raises(ValueError, match="invalid action"):
        env.step(invalid)
    with pytest.raises(ValueError, match="invalid action index"):
        env.step(9)


def test_compression_is_masked_when_the_compressor_cannot_shorten() -> None:
    task = one_task()
    env = build_env(compressor=IdentityCompressor())
    env.reset(task)

    while not env.terminated:
        assert OperationKind.COMPRESS not in env.valid_actions()
        assert env.action_mask()[1] is np.False_
        with pytest.raises(ValueError, match="invalid action"):
            env.step(OperationKind.COMPRESS)
        env.step(OperationKind.KEEP)


def test_step_after_termination_raises() -> None:
    task = one_task()
    env = build_env()
    run_all_keep(env, task)

    with pytest.raises(RuntimeError, match="already terminated"):
        env.step(OperationKind.KEEP)


def test_empty_history_terminates_with_a_reason() -> None:
    task = SyntheticTask(
        task_id="empty",
        family=TaskFamily.FACT_RECALL,
        template_id="empty",
        seed=0,
        history=(),
        question="What is the user's name?",
        labels=TaskLabels(reference_answer="Alice", required_facts=()),
    )
    env = build_env()

    observation = env.reset(task)

    assert env.terminated
    assert env.termination_reason == "empty_history"
    assert observation.shape == (FEATURE_DIM,)
    with pytest.raises(RuntimeError):
        env.step(OperationKind.KEEP)


def test_reset_leaves_no_state_from_the_previous_episode() -> None:
    task = one_task(family=TaskFamily.DISTRACTOR_HEAVY)
    env = build_env()
    baseline = env.reset(task).copy()

    while not env.terminated:
        env.step(OperationKind.PRUNE)
    assert env.memory_tokens == 0

    again = env.reset(task)

    assert env.memory_tokens == env.original_memory_tokens
    assert np.array_equal(again, baseline)
    assert env.action_mask().any()


# --- budgets and context limits -----------------------------------------------


def test_budget_is_a_fraction_of_original_memory_tokens() -> None:
    task = one_task()
    env = build_env(budget_fraction=0.5)
    env.reset(task)

    assert env.budget_tokens == max(1, int(np.ceil(0.5 * env.original_memory_tokens)))
    assert env.original_memory_tokens == sum(len(turn.text) for turn in task.history)


def test_over_budget_is_reported_not_truncated() -> None:
    task = one_task()
    env = build_env(budget_fraction=0.01)
    info = run_all_keep(env, task)

    assert info["budget_compliant"] is False
    assert env.memory_tokens > env.budget_tokens
    prompt = env.serialize_active_memory()
    for turn in task.history:
        assert turn.text in prompt.memory_text


def test_context_limit_is_a_reported_failure_not_a_truncation() -> None:
    task = one_task()
    env = build_env(context_limit_tokens=5)
    info = run_all_keep(env, task)

    assert info["context_ok"] is False
    assert info["total_prompt_tokens"] > 5
    prompt = env.serialize_active_memory()
    assert len(prompt.memory_text) > 0


# --- answering and critical-information loss ----------------------------------


def test_keeping_everything_answers_every_template_correctly() -> None:
    """Guards against drift between template wording and answerer patterns."""
    for template in templates_for():
        task = generate_dataset(
            family=template.family, split=template.split, count=1, seed=11
        )[0]
        env = build_env(budget_fraction=1.0)
        info = run_all_keep(env, task)

        assert info["correct"] is True, (template.template_id, info["answer"])
        assert info["critical_facts_lost"] == ()


def test_critical_fact_can_be_lost_and_is_measured() -> None:
    task = one_task(family=TaskFamily.FACT_RECALL, split=Split.TRAIN)
    critical_id = task.labels.required_facts[0].memory_id
    env = build_env(budget_fraction=1.0)
    env.reset(task)

    terminated = False
    info: dict[str, object] = {}
    while not terminated:
        operation = OperationKind.PRUNE if env.valid_actions() and _current_id(env) == critical_id else OperationKind.KEEP
        _, _, terminated, info = env.step(operation)

    assert info["correct"] is False
    assert info["answer"] == "unknown"
    assert info["critical_facts_lost"] == (critical_id,)
    assert info["critical_facts_retained"] == ()


def _current_id(env: MemoryEnv) -> str:
    return env.valid_actions() and env._current_item().memory_id  # type: ignore[attr-defined]


def test_pruning_everything_empties_the_prompt() -> None:
    task = one_task()
    env = build_env()
    env.reset(task)

    while not env.terminated:
        env.step(OperationKind.PRUNE)

    prompt = env.serialize_active_memory()
    assert prompt.memory_text == ""
    assert prompt.included_ids == ()
    assert env.memory_tokens == 0
    assert env.reward_breakdown is not None
    assert env.reward_breakdown.task_success == 0.0


# --- observation contract -----------------------------------------------------


def test_observation_is_finite_fixed_dimension_and_bounded() -> None:
    task = one_task(family=TaskFamily.DISTRACTOR_HEAVY)
    env = build_env()
    observation = env.reset(task)

    assert observation.shape == (FEATURE_DIM,)
    assert observation.dtype == np.float32
    assert np.isfinite(observation).all()
    assert (observation >= 0.0).all()
    # Only budget_used_ratio may exceed 1 (documented cap of 2).
    assert (observation[:8] <= 1.0).all()


def test_token_counts_stay_consistent_across_steps() -> None:
    task = one_task()
    env = build_env()
    env.reset(task)

    while not env.terminated:
        before = env.memory_tokens
        _, _, terminated, info = env.step(OperationKind.KEEP)
        assert env.memory_tokens == before
        assert info["memory_tokens"] == env.memory_tokens
        assert terminated or info["memory_tokens"] == sum(
            len(turn.text) for turn in task.history
        )


# --- exit criterion -----------------------------------------------------------


def test_random_valid_policies_run_1000_episodes_without_failures() -> None:
    tasks = tuple(
        task
        for family in TaskFamily
        for task in generate_dataset(family=family, split=Split.TRAIN, count=2, seed=7)
    )
    env = build_env()
    rng = random.Random(20261027)
    episodes = 1000

    for episode in range(episodes):
        task = tasks[episode % len(tasks)]
        observation = env.reset(task, seed=episode)
        assert observation.shape == (FEATURE_DIM,)
        assert np.isfinite(observation).all()

        steps = 0
        terminated = False
        info: dict[str, object] = {}
        while not terminated:
            valid = env.valid_actions()
            assert valid, f"no valid action in episode {episode}"
            action = rng.choice(valid)
            assert env.action_mask()[list(OperationKind).index(action)]

            observation, reward, terminated, info = env.step(action)
            steps += 1

            assert np.isfinite(observation).all()
            assert np.isfinite(reward)
            assert info["memory_tokens"] == env.memory_tokens
            assert steps <= len(task.history)

        assert steps == len(task.history)
        assert info["termination_reason"] == "completed_all_candidates"
        assert info["budget_tokens"] == env.budget_tokens
        assert isinstance(info["correct"], bool)
        assert info["critical_facts_lost"] == tuple(
            fact_id
            for fact_id in info["critical_facts_lost"]  # type: ignore[union-attr]
            if fact_id not in info["critical_facts_retained"]  # type: ignore[operator]
        )
