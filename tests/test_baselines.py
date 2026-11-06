"""Phase 5 baseline tests (plan.md §7 Phase 5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_rl.baselines import (
    BASELINES,
    CadocRule,
    ObservationMasking,
    SlidingWindow,
    ThresholdCompaction,
    build_baseline,
)
from memory_rl.baselines.llmlingua import llmlingua_available
from memory_rl.baselines.rules import PLACEHOLDER
from memory_rl.cost import CacheAwareCostMeter, get_profile
from memory_rl.envs.agent_loop import AgentEpisode, LoopConfig
from memory_rl.envs.policy_api import ControllerState, Decision, MemoryPolicy
from memory_rl.envs.toy import KeyHuntEnv
from memory_rl.evaluation.metrics import aggregate, bootstrap_ci, paired_difference_ci
from memory_rl.evaluation.records import read_jsonl, validate_record
from memory_rl.evaluation.run_eval import main as eval_main
from memory_rl.llm import ScriptedKeyHuntLLM
from memory_rl.memory import MemoryType, OperationKind, TestTokenizer

PROFILE = get_profile("anthropic-sonnet-5-5")
TOKENIZER = TestTokenizer()


class Recorder:
    """Wraps a policy and records (state, decision) pairs."""

    def __init__(self, inner: MemoryPolicy) -> None:
        self.inner = inner
        self.name = inner.name
        self.compressor = getattr(inner, "compressor", None)
        self.log: list[tuple[ControllerState, Decision]] = []

    def reset(self, seed: int) -> None:
        self.inner.reset(seed)

    def decide(self, state: ControllerState) -> Decision:
        decision = self.inner.decide(state)
        self.log.append((state, decision))
        return decision


def run(policy: MemoryPolicy, task_id: str = "keyhunt-valid-0001", limit: int = 100_000):
    episode = AgentEpisode(
        KeyHuntEnv(), task_id, seed=0, llm=ScriptedKeyHuntLLM(),
        meter=CacheAwareCostMeter(PROFILE), tokenizer=TOKENIZER, profile=PROFILE,
        compressor=getattr(policy, "compressor", None),
        config=LoopConfig(context_limit_tokens=limit),
    )  # fmt: skip
    return episode.run(policy)


def test_sliding_window_boundary() -> None:
    rec = Recorder(SlidingWindow(k=2))
    run(rec)
    for state, decision in rec.log:
        for item in state.candidates:
            pruned = decision.actions.get(item.memory_id) is OperationKind.PRUNE
            assert pruned == (state.turn - item.turn_index > 2)


def test_obs_masking_keeps_actions_and_protected_content() -> None:
    rec = Recorder(ObservationMasking(m=1))
    episode = run(rec)
    for _, decision in rec.log:
        assert set(decision.actions.values()) <= {OperationKind.COMPRESS}
    for item in episode.store.archived_items():
        if item.memory_type is MemoryType.ASSISTANT:
            assert item.compression_count == 0  # thoughts/actions untouched
        elif item.compression_count:
            assert item.text == PLACEHOLDER
    prompt = episode.current_prompt()
    assert prompt.messages[0].content == episode.env.system_prompt()
    assert episode.success


def test_threshold_compaction_triggers_exactly_above_t() -> None:
    rec = Recorder(ThresholdCompaction(t=8000))
    run(rec)
    for state, decision in rec.log:
        assert decision.edit == (state.prompt_tokens > 8000 and bool(state.candidates))


def test_cadoc_rule_follows_eq14_ledger() -> None:
    """Commit iff W + w_r*G >= Q; otherwise the ledger grows by w_r*G (Eqs. 12-14)."""
    rule = CadocRule(m=1, guard_fill=10.0)
    rec = Recorder(rule)
    run(rec)
    ledger = 0.0
    for state, decision in rec.log:
        pending = rule.pending(state)
        if not pending:
            assert not decision.edit
            continue
        w_r = state.cache_read_multiplier
        gain = sum(item.compression_gain for item in pending)
        newest = max(item.span_end for item in pending)
        frontier = min(
            (i.span_start for i in state.items if i.turn_index >= state.turn - 1),
            default=state.prompt_tokens,
        )
        q = (1 - w_r) * max(frontier - newest, 0)
        commit = ledger + w_r * gain >= q
        assert decision.edit == commit
        ledger = 0.0 if commit else ledger + w_r * gain


def test_every_baseline_respects_masks_and_produces_valid_records() -> None:
    for method, (_, grid) in BASELINES.items():
        for knob in grid or [None]:
            policy = build_baseline(method, knob)
            episode = run(policy, limit=40_000)  # type: ignore[arg-type]
            record = episode.record(method=method, knob=str(knob))
            validate_record(record)  # masks are enforced by the loop (raises otherwise)


def test_unknown_baseline_is_an_error() -> None:
    with pytest.raises(ValueError, match="unknown baseline"):
        build_baseline("magic")


@pytest.mark.skipif(not llmlingua_available(), reason="optional extra [llmlingua] not installed")
def test_llmlingua_policy_runs() -> None:  # pragma: no cover - needs the optional extra
    from memory_rl.baselines.llmlingua import LLMLinguaPolicy

    run(LLMLinguaPolicy(m=2, rate=0.5))


# --- metrics and runner ----------------------------------------------------------


def test_bootstrap_ci_brackets_mean() -> None:
    mean, lo, hi = bootstrap_ci([[1.0], [2.0], [3.0], [4.0]])
    assert lo <= mean <= hi and mean == 2.5
    diff, dlo, dhi = paired_difference_ci({"a": 2.0, "b": 3.0}, {"a": 1.0, "b": 1.0})
    assert diff == 1.5 and dlo <= diff <= dhi


def test_run_eval_writes_records_and_summary(tmp_path: Path) -> None:
    out = tmp_path / "eval"
    args = ["--method", "obs_masking", "--knob", "1", "--tasks", "3", "--seeds", "0", "1",
            "--profiles", "openai-gpt-5-4", "--out", str(out)]  # fmt: skip
    assert eval_main(args) == 0
    records = read_jsonl(out / "records.jsonl")
    assert len(records) == 6 and {r["seed"] for r in records} == {0, 1}
    summary = json.loads((out / "summary.json").read_text())
    assert summary["rows"][0]["method"] == "obs_masking"
    assert aggregate(records)[0]["episodes"] == 6
    with pytest.raises(SystemExit, match="overwrite"):
        eval_main(args)
