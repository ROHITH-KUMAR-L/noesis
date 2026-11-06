"""Phase 3 agent-harness tests (plan.md §7 Phase 3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_rl.cost import CacheAwareCostMeter, TokenCostMeter, get_profile
from memory_rl.envs.agent_loop import (
    AgentEpisode,
    InvalidDecisionError,
    LoopConfig,
)
from memory_rl.envs.agent_loop import (
    main as loop_main,
)
from memory_rl.envs.agent_prompt import AgentPromptSerializer
from memory_rl.envs.base import TerminalReason
from memory_rl.envs.policy_api import AlwaysKeepPolicy, ControllerState, Decision
from memory_rl.envs.toy import KeyHuntEnv, generate_task, parse_action
from memory_rl.evaluation.records import read_jsonl, validate_record, write_jsonl
from memory_rl.llm import (
    LLMCallCapExceeded,
    LLMConfig,
    LLMError,
    OpenAICompatibleBackend,
    ScriptedKeyHuntLLM,
    load_dotenv,
)
from memory_rl.memory import MemoryItem, MemoryType, OperationKind, TestTokenizer

PROFILE = get_profile("anthropic-sonnet-5-5")
TOKENIZER = TestTokenizer()  # fast; token IDs = code points (so ~4x BPE counts)
CHAR_SCALE = LoopConfig(context_limit_tokens=100_000)


def episode(task_id: str = "keyhunt-valid-0000", **kwargs: object) -> AgentEpisode:
    kwargs.setdefault("config", CHAR_SCALE)
    return AgentEpisode(
        KeyHuntEnv(),
        task_id,
        seed=0,
        llm=ScriptedKeyHuntLLM(),
        meter=CacheAwareCostMeter(PROFILE),
        tokenizer=TOKENIZER,
        profile=PROFILE,
        **kwargs,  # type: ignore[arg-type]
    )


class PruneEverything:
    name = "prune_everything"

    def reset(self, seed: int) -> None:
        return None

    def decide(self, state: ControllerState) -> Decision:
        actions = {item.memory_id: OperationKind.PRUNE for item in state.candidates}
        return Decision(edit=bool(actions), actions=actions)


class PruneKeysOnce:
    """Prunes every opened credential record once the next turn starts."""

    name = "prune_keys"

    def reset(self, seed: int) -> None:
        self.done = False

    def decide(self, state: ControllerState) -> Decision:
        targets = {
            item.memory_id: OperationKind.PRUNE
            for item in state.candidates
            if "access credentials)" in item.text and not self.done
        }
        if targets:
            self.done = True
        return Decision(edit=bool(targets), actions=targets)


# --- env ----------------------------------------------------------------------


def test_tasks_are_deterministic_and_splits_disjoint() -> None:
    env = KeyHuntEnv()
    assert generate_task("keyhunt-train-0001") == generate_task("keyhunt-train-0001")
    splits = [set(env.task_ids(s)) for s in ("train", "valid", "test")]
    assert not (splits[0] & splits[1]) and not (splits[0] & splits[2])
    assert not (splits[1] & splits[2])


def test_parse_action_takes_last_action() -> None:
    assert parse_action("Thought: x\nAction: search[alpha]") == ("search", "alpha")
    assert parse_action("Action: open[REC-1]\nAction: submit[K-1, K-2]") == (
        "submit",
        "K-1, K-2",
    )
    assert parse_action("no action here") is None


def test_invalid_action_is_reported_not_crashing() -> None:
    env = KeyHuntEnv()
    env.reset("keyhunt-valid-0000", 0)
    obs, reward, done, info = env.step("I refuse")
    assert "Invalid action" in obs.text and reward == 0.0 and not done
    assert info["valid_action"] is False


# --- episodes -----------------------------------------------------------------


def test_mock_llm_completes_toy_episode_deterministically() -> None:
    first = episode().run(AlwaysKeepPolicy()).record(method="full_history")
    second = episode().run(AlwaysKeepPolicy()).record(method="full_history")
    assert first == second
    assert first["success"] is True
    assert first["terminal_reason"] == TerminalReason.SUCCESS
    assert first["edits"] == 0
    validate_record(first)


def test_full_history_succeeds_on_validation_tasks() -> None:
    env_ids = list(KeyHuntEnv().task_ids("valid"))[:20]
    results = [episode(t).run(AlwaysKeepPolicy()).record(method="fh") for t in env_ids]
    assert all(r["success"] for r in results)


def test_cost_is_accumulated_over_every_call() -> None:
    ep = episode().run(AlwaysKeepPolicy())
    record = ep.record(method="full_history")
    assert record["llm_calls"] == record["turns"] == len(ep.turns)
    traced = sum(t.cost["total_cost"] for t in ep.turns if t.cost)  # type: ignore[index]
    assert record["total_cost"] == pytest.approx(traced)
    assert record["cached_tokens_total"] > 0  # append-only history hits the cache
    assert record["token_cost"] == record["prompt_tokens_total"]


def test_token_meter_variant_runs_and_reports_both_costs() -> None:
    ep = AgentEpisode(
        KeyHuntEnv(), "keyhunt-valid-0001", seed=0, llm=ScriptedKeyHuntLLM(),
        meter=TokenCostMeter(), tokenizer=TOKENIZER, profile=PROFILE, config=CHAR_SCALE,
    )  # fmt: skip
    record = ep.run(AlwaysKeepPolicy()).record(method="fh", variant="token")
    assert record["cost_unit"] == "tokens"
    assert record["total_cost"] == record["prompt_tokens_total"]


def test_context_overflow_is_a_recorded_failure_without_truncation() -> None:
    ep = episode(config=LoopConfig(context_limit_tokens=1500)).run(AlwaysKeepPolicy())
    record = ep.record(method="full_history")
    assert record["terminal_reason"] == TerminalReason.CONTEXT_OVERFLOW
    assert record["success"] is False
    assert ep.turns[-1].prompt_tokens > 1500  # the full prompt was measured, not cut
    assert ep.turns[-1].cost is None  # and never sent


def test_aggressive_pruning_loses_needed_facts_and_fails() -> None:
    record = episode().run(PruneEverything()).record(method="prune_all")
    assert record["success"] is False
    assert record["terminal_reason"] == TerminalReason.STEP_LIMIT
    assert record["actions_prune"] > 0


def test_pruning_a_key_forces_extra_calls() -> None:
    base = episode().run(AlwaysKeepPolicy()).record(method="fh")
    pruned = episode().run(PruneKeysOnce()).record(method="prune_keys")
    assert pruned["llm_calls"] > base["llm_calls"]


# --- serializer and invariants ------------------------------------------------


def _items(n: int) -> list[MemoryItem]:
    kinds = (MemoryType.ASSISTANT, MemoryType.TOOL_RESULT)
    return [
        MemoryItem.create(
            memory_id=f"m{i}", text=f"text {i} " * (i + 1), tokenizer=TOKENIZER,
            turn_index=i // 2, memory_type=kinds[i % 2],
        )  # fmt: skip
        for i in range(n)
    ]


def test_serializer_puts_protected_content_first_and_is_prefix_stable() -> None:
    ser = AgentPromptSerializer(TOKENIZER, "SYSTEM")
    items = _items(6)
    full = ser.serialize("Task: t.", "FEWSHOT", items)
    assert [m.role for m in full.messages[:2]] == ["system", "user"]
    assert full.messages[0].content == "SYSTEM" and "FEWSHOT" in full.messages[1].content
    shorter = ser.serialize("Task: t.", "FEWSHOT", items[:3])
    assert full.token_ids[: shorter.total_prompt_tokens] == shorter.token_ids
    start, end = full.item_spans["m3"]
    assert full.suffix_tokens_after("m3") == full.total_prompt_tokens - start
    assert end > start >= full.prefix_tokens


def test_pruned_item_is_absent_from_the_next_prompt() -> None:
    ep = episode()
    state: ControllerState | None = ep.begin()
    for _ in range(3):
        state, _ = ep.advance(Decision.no_edit())
    assert state is not None
    target = state.candidates[0]
    state, _ = ep.advance(Decision(edit=True, actions={target.memory_id: OperationKind.PRUNE}))
    prompt = ep.current_prompt()
    assert target.memory_id not in prompt.included_ids
    assert all(target.text not in m.content for m in prompt.messages)
    assert ep.store.get(target.memory_id).original_text == target.text  # archived


def test_masks_newest_item_and_compress_once() -> None:
    ep = episode()
    state: ControllerState | None = ep.begin()
    for _ in range(2):
        state, _ = ep.advance(Decision.no_edit())
    assert state is not None
    assert not state.items[-1].is_candidate
    obs = next(i for i in state.candidates if OperationKind.COMPRESS in i.valid_actions)
    assert obs.compression_gain > 0
    state, _ = ep.advance(Decision(edit=True, actions={obs.memory_id: OperationKind.COMPRESS}))
    assert state is not None
    again = next(i for i in state.items if i.memory_id == obs.memory_id)
    assert OperationKind.COMPRESS not in again.valid_actions


def test_invalid_decisions_are_rejected() -> None:
    ep = episode()
    ep.begin()
    with pytest.raises(InvalidDecisionError):
        ep.advance(Decision(edit=True, actions={"nope": OperationKind.PRUNE}))
    ep2 = episode()
    ep2.begin()
    with pytest.raises(InvalidDecisionError):
        ep2.advance(Decision(edit=False, actions={"m001": OperationKind.PRUNE}))


def test_gold_labels_never_reach_prompts_or_metadata() -> None:
    ep = episode("keyhunt-valid-0005")
    state: ControllerState | None = ep.begin()
    task = ep.env.task  # type: ignore[attr-defined]
    first = ep.current_prompt()
    blob = "\n".join(m.content for m in first.messages)
    for key in task.keys:
        assert key not in blob
    for record_id in task.credential_ids:
        assert record_id not in blob
    while state is not None:
        state, _ = ep.advance(Decision.no_edit())
    for item in ep.store.archived_items():
        assert dict(item.metadata) == {}
    record = ep.record(method="fh")
    assert not any(key in json.dumps(record) for key in task.keys)


# --- records, LLM backend, CLI -------------------------------------------------


def test_records_roundtrip_and_refuse_overwrite(tmp_path: Path) -> None:
    record = episode().run(AlwaysKeepPolicy()).record(method="fh")
    path = tmp_path / "r.jsonl"
    assert write_jsonl(path, [record], overwrite=False) == 1
    assert read_jsonl(path) == [record]
    with pytest.raises(FileExistsError):
        write_jsonl(path, [record], overwrite=False)


def test_dotenv_and_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("NOESIS_LLM_BASE_URL", "NOESIS_LLM_API_KEY", "NOESIS_LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(LLMError, match="missing"):
        LLMConfig.from_env(tmp_path / "absent.env")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nNOESIS_LLM_BASE_URL=http://x/v1/\nNOESIS_LLM_API_KEY='k'\n"
        "export NOESIS_LLM_MODEL=m\nNOESIS_LLM_MAX_CALLS=1\n"
    )
    assert load_dotenv(env_file)["NOESIS_LLM_API_KEY"] == "k"
    config = LLMConfig.from_env(env_file)
    assert (config.base_url, config.model, config.max_calls) == ("http://x/v1", "m", 1)
    assert "api_key" not in config.redacted()


def test_backend_call_cap_and_response_parsing() -> None:
    backend = OpenAICompatibleBackend(LLMConfig("http://127.0.0.1:9", "k", "m", max_calls=0))
    with pytest.raises(LLMCallCapExceeded):
        backend.generate([], max_tokens=5)
    parsed = OpenAICompatibleBackend._parse(
        {"choices": [{"message": {"content": "hi"}}], "usage": {"completion_tokens": 3}}, 0.5
    )
    assert (parsed.text, parsed.output_tokens) == ("hi", 3)
    with pytest.raises(LLMError):
        OpenAICompatibleBackend._parse({"oops": 1}, 0.1)


def test_paid_llm_requires_opt_in() -> None:
    with pytest.raises(SystemExit, match="allow-paid"):
        loop_main(["--llm", "openai", "--smoke"])


def test_cli_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    assert loop_main(["--smoke"]) == 0
    assert '"success": true' in capsys.readouterr().out
