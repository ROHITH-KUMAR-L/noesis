"""Leakage tests: simulator-only labels must never influence the policy (plan.md §4.2).

These tests mutate labels and assert the observation is unchanged, and assert that
pruned content never reaches the serialized answer prompt.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from memory_rl.answering import UNKNOWN_ANSWER, SyntheticExactMatchAnswerer
from memory_rl.environment import (
    FEATURE_DIM,
    FEATURE_NAMES,
    EnvConfig,
    MemoryEnv,
    Split,
    TaskFamily,
    TaskLabels,
    generate_dataset,
)
from memory_rl.memory import OperationKind, TestTokenizer

TOKENIZER = TestTokenizer()


def build_env() -> MemoryEnv:
    return MemoryEnv(
        tokenizer=TOKENIZER, answerer=SyntheticExactMatchAnswerer(), config=EnvConfig()
    )


def _task(family: TaskFamily = TaskFamily.FACT_RECALL):
    return generate_dataset(family=family, split=Split.TRAIN, count=1, seed=21)[0]


def test_labels_do_not_change_observations() -> None:
    task = _task()
    env = build_env()

    baseline = env.reset(task, seed=1)
    tampered = replace(
        task,
        labels=TaskLabels(reference_answer="completely different", required_facts=()),
    )
    mutated = env.reset(tampered, seed=1)

    assert np.array_equal(baseline, mutated)


def test_required_fact_metadata_does_not_change_observations() -> None:
    task = _task()
    env = build_env()
    baseline = env.reset(task, seed=2)

    relabelled = replace(
        task,
        labels=TaskLabels(
            reference_answer=task.labels.reference_answer,
            required_facts=tuple(
                replace(fact, text="not in memory") for fact in task.labels.required_facts
            ),
        ),
    )
    mutated = env.reset(relabelled, seed=2)

    assert np.array_equal(baseline, mutated)


def test_feature_schema_names_contain_no_labels() -> None:
    forbidden = ("critical", "required", "answer", "importance", "reference")
    for name in FEATURE_NAMES:
        assert not any(word in name for word in forbidden), name
    assert len(FEATURE_NAMES) == FEATURE_DIM


def test_non_terminal_info_exposes_no_labels() -> None:
    task = _task()
    env = build_env()
    env.reset(task)

    _, _, terminated, info = env.step(OperationKind.KEEP)

    assert not terminated
    for key in ("reference_answer", "critical_facts_lost", "critical_facts_retained"):
        assert key not in info


def test_pruned_content_never_reaches_the_answer_prompt() -> None:
    task = _task(family=TaskFamily.DISTRACTOR_HEAVY)
    env = build_env()
    env.reset(task)

    while not env.terminated:
        env.step(OperationKind.PRUNE)

    prompt = env.serialize_active_memory()
    for turn in task.history:
        assert turn.text not in prompt.memory_text
        assert turn.text not in prompt.text

    assert prompt.included_ids == ()
    assert prompt.memory_tokens == 0
    assert env.termination_reason == "completed_all_candidates"


def test_answerer_only_sees_active_memory() -> None:
    task = _task(family=TaskFamily.MULTI_STEP_DEPENDENCY)
    env = build_env()
    env.reset(task)

    # Prune the tool result that carries the dependency key, keep everything else.
    tool_id = next(
        turn.memory_id
        for turn in task.history
        if turn.memory_type.value == "tool_result" and turn.text.startswith("Tool:")
    )
    terminated = False
    info: dict[str, object] = {}
    while not terminated:
        current = env.current_item_id
        action = OperationKind.PRUNE if current == tool_id else OperationKind.KEEP
        _, _, terminated, info = env.step(action)

    assert info["answer"] == UNKNOWN_ANSWER
    assert info["correct"] is False
    prompt = env.serialize_active_memory()
    pruned_text = next(turn.text for turn in task.history if turn.memory_id == tool_id)
    assert pruned_text not in prompt.memory_text


def test_memory_items_carry_no_simulator_metadata() -> None:
    task = _task()
    env = build_env()
    env.reset(task)

    for item in env.serialize_active_memory().included_ids:
        assert item  # ids only; labels never enter the prompt

    prompt = env.serialize_active_memory()
    # The answer may legitimately appear in memory content; it must never be
    # injected by the serializer (instructions, headers, question).
    assert task.labels.reference_answer not in prompt.fixed_text
    for fact in task.labels.required_facts:
        # The fact text is legitimate memory content; the label object is not.
        assert fact.memory_id not in prompt.text
