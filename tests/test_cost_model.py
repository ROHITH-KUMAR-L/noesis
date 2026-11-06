"""Phase 2 cost-model tests (plan.md §7 Phase 2). All expected values are hand-computed."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from memory_rl.cost import (
    CacheAwareCostMeter,
    EpisodeCostAccumulator,
    PrefixCacheSimulator,
    PricingError,
    PricingProfile,
    TokenCostMeter,
    available_profiles,
    get_profile,
    load_profile,
)
from memory_rl.cost.cache_sim import main as cache_sim_main
from memory_rl.cost.cache_sim import run_scripted

ILLUSTRATIVE = get_profile("illustrative-test")  # $1/MTok in, read 0.1, write 1.25, min 10
U = 1e-6  # $ per uncached input token under the illustrative profile


def automatic(profile: PricingProfile = ILLUSTRATIVE, **changes: object) -> PricingProfile:
    return replace(profile, mode="automatic", **changes)  # type: ignore[arg-type]


def ids(n: int, start: int = 1) -> list[int]:
    return list(range(start, start + n))


def history_prompt(prefix: int, items: int, item_len: int, edited: int | None = None) -> list[int]:
    out = ids(prefix)
    for k in range(items):
        block = [(k + 1) * 10_000 + j for j in range(item_len)]
        if k == edited:
            block[0] = -1  # first token of the edited item changes
        out.extend(block)
    return out


# --- profiles ------------------------------------------------------------------


def test_repository_profiles_load_and_record_source_and_date() -> None:
    profiles = available_profiles()
    assert {"anthropic-sonnet-5-5", "openai-gpt-5-4", "openai-gpt-6-1-sol"} <= set(profiles)
    modes = {profile.mode for profile in profiles.values() if not profile.illustrative}
    assert modes == {"automatic", "breakpoint"}
    for profile in profiles.values():
        assert profile.source and profile.retrieved
        if not profile.illustrative:
            assert profile.source.startswith("https://")


@pytest.mark.parametrize("missing", ["source", "retrieved"])
def test_loader_rejects_profile_without_provenance(tmp_path: Path, missing: str) -> None:
    fields = {
        "name": "x",
        "input_per_mtok": 1,
        "output_per_mtok": 1,
        "cache_read_multiplier": 0.1,
        "cache_write_multiplier": 1,
        "min_cacheable_tokens": 0,
        "block_size": 1,
        "ttl_seconds": 60,
        "mode": "automatic",
        "source": "https://example.com",
        "retrieved": "2026-11-04",
    }
    del fields[missing]
    path = tmp_path / "p.yaml"
    path.write_text("\n".join(f"{k}: {v}" for k, v in fields.items()))
    with pytest.raises(PricingError, match=missing):
        load_profile(path)


@pytest.mark.parametrize(
    "change",
    [
        {"input_per_mtok": -1.0},
        {"cache_read_multiplier": -0.1},
        {"block_size": 0},
        {"ttl_seconds": 0.0},
        {"mode": "magic"},
        {"source": " "},
    ],
)
def test_invalid_profiles_are_rejected(change: dict[str, object]) -> None:
    with pytest.raises(PricingError):
        replace(ILLUSTRATIVE, **change)  # type: ignore[arg-type]


def test_unknown_profile_name_lists_available() -> None:
    with pytest.raises(PricingError, match="available"):
        get_profile("does-not-exist")


# --- simulator semantics -------------------------------------------------------


@pytest.mark.parametrize("profile", [ILLUSTRATIVE, automatic()])
def test_cold_first_call(profile: PricingProfile) -> None:
    cost = PrefixCacheSimulator(profile).call(ids(150), output_tokens=20, now=0.0)
    assert cost.cached_tokens == 0
    assert cost.written_tokens == 150
    assert cost.uncached_tokens == 0
    assert cost.cache_write_cost == pytest.approx(150 * 1.25 * U)
    assert cost.output_cost == pytest.approx(20 * 4 * U)
    assert cost.total_cost == pytest.approx(150 * 1.25 * U + 80 * U)


@pytest.mark.parametrize("profile", [ILLUSTRATIVE, automatic()])
def test_append_only_second_call_reads_previous_prompt(profile: PricingProfile) -> None:
    sim = PrefixCacheSimulator(profile)
    sim.call(ids(150), 0, now=0.0)
    cost = sim.call(ids(200), 0, now=1.0)
    assert cost.cached_tokens == 150
    assert cost.written_tokens == 50
    assert cost.total_cost == pytest.approx(150 * 0.1 * U + 50 * 1.25 * U)


def test_block_rounding_floors_cached_length() -> None:
    sim = PrefixCacheSimulator(automatic(block_size=128, min_cacheable_tokens=0))
    sim.call(ids(300), 0, now=0.0)
    cost = sim.call(ids(400), 0, now=1.0)
    assert cost.cached_tokens == 256  # floor(300 / 128) * 128
    assert cost.written_tokens == 144


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_block_boundary_plus_minus_one(delta: int) -> None:
    sim = PrefixCacheSimulator(automatic(block_size=128, min_cacheable_tokens=0))
    sim.call(ids(256 + delta), 0, now=0.0)
    cost = sim.call(ids(600), 0, now=1.0)
    assert cost.cached_tokens == (128 if delta < 0 else 256)


def test_edit_position_sets_cached_length_and_early_edits_cost_more() -> None:
    profile = automatic()
    totals = []
    for edited in range(10):
        sim = PrefixCacheSimulator(profile)
        sim.call(history_prompt(100, 10, 50), 0, now=0.0)
        cost = sim.call(history_prompt(100, 10, 50, edited=edited), 0, now=1.0)
        assert cost.cached_tokens == 100 + 50 * edited  # ends at the first changed token
        totals.append(cost.total_cost)
    assert all(a > b for a, b in zip(totals, totals[1:], strict=False))


def test_prefix_below_minimum_is_never_cached() -> None:
    for profile in (ILLUSTRATIVE, automatic()):
        sim = PrefixCacheSimulator(profile)
        first = sim.call(ids(5), 0, now=0.0)
        second = sim.call(ids(5), 0, now=1.0)
        assert first.written_tokens == 0 and first.uncached_tokens == 5
        assert second.cached_tokens == 0


def test_ttl_expiry_gives_a_cold_call_and_reads_refresh() -> None:
    sim = PrefixCacheSimulator(automatic())
    sim.call(ids(100), 0, now=0.0)
    assert sim.call(ids(100), 0, now=250.0).cached_tokens == 100  # refreshes at t=250
    assert sim.call(ids(100), 0, now=500.0).cached_tokens == 100  # 250 s after refresh
    assert sim.call(ids(100), 0, now=801.0).cached_tokens == 0  # 301 s idle: expired


def test_breakpoint_and_automatic_accounting_differ() -> None:
    prompt = ids(150)
    bp = PrefixCacheSimulator(ILLUSTRATIVE).call(prompt, 0, now=0.0, breakpoints=[100])
    auto = PrefixCacheSimulator(automatic()).call(prompt, 0, now=0.0)
    assert (bp.written_tokens, bp.uncached_tokens) == (100, 50)
    assert (auto.written_tokens, auto.uncached_tokens) == (150, 0)

    # A partial overlap only counts in automatic mode: reads land on written entries.
    diverging = ids(120) + ids(30, start=9_000)
    bp_sim = PrefixCacheSimulator(ILLUSTRATIVE)
    bp_sim.call(prompt, 0, now=0.0)
    assert bp_sim.call(diverging, 0, now=1.0).cached_tokens == 0
    auto_sim = PrefixCacheSimulator(automatic())
    auto_sim.call(prompt, 0, now=0.0)
    assert auto_sim.call(diverging, 0, now=1.0).cached_tokens == 120


def test_breakpoint_after_edited_item_still_reads_the_protected_prefix() -> None:
    sim = PrefixCacheSimulator(ILLUSTRATIVE)
    sim.call(history_prompt(100, 3, 50), 0, now=0.0, breakpoints=[100, 250])
    cost = sim.call(history_prompt(100, 3, 50, edited=1), 0, now=1.0, breakpoints=[100, 250])
    assert cost.cached_tokens == 100
    assert cost.written_tokens == 150


def test_breakpoint_validation() -> None:
    sim = PrefixCacheSimulator(ILLUSTRATIVE)
    with pytest.raises(ValueError):
        sim.call(ids(10), 0, now=0.0, breakpoints=[11])
    with pytest.raises(ValueError):
        sim.call(ids(100), 0, now=0.0, breakpoints=[10, 20, 30, 40, 50])
    with pytest.raises(ValueError):
        PrefixCacheSimulator(automatic()).call(ids(10), 0, now=0.0, breakpoints=[5])


def test_identical_consecutive_prompts_fully_cached() -> None:
    sim = PrefixCacheSimulator(ILLUSTRATIVE)
    sim.call(ids(80), 0, now=0.0)
    cost = sim.call(ids(80), 0, now=1.0)
    assert (cost.cached_tokens, cost.written_tokens, cost.uncached_tokens) == (80, 0, 0)


def test_time_cannot_go_backwards_and_reset_is_cold() -> None:
    sim = PrefixCacheSimulator(ILLUSTRATIVE)
    sim.call(ids(50), 0, now=10.0)
    with pytest.raises(ValueError):
        sim.call(ids(50), 0, now=5.0)
    sim.reset()
    assert sim.call(ids(50), 0, now=0.0).cached_tokens == 0


def test_cost_terms_sum_finite_and_deterministic() -> None:
    def run() -> list[float]:
        sim = PrefixCacheSimulator(get_profile("anthropic-sonnet-5-5"))
        out = []
        for t in range(6):
            cost = sim.call(history_prompt(600, t + 1, 300, edited=0 if t == 3 else None), 40, t)
            terms = cost.input_cost + cost.cache_read_cost + cost.cache_write_cost
            assert math.isclose(cost.total_cost, terms + cost.output_cost)
            assert math.isfinite(cost.total_cost)
            out.append(cost.total_cost)
        return out

    assert run() == run()


# --- meters --------------------------------------------------------------------


def test_token_meter_ignores_caching() -> None:
    meter = TokenCostMeter(output_weight=0.5)
    first = meter.call(ids(100), 10, now=0.0)
    second = meter.call(ids(100), 10, now=1.0)
    assert first == second
    assert second.cached_tokens == 0
    assert second.total_cost == 105.0


def test_cache_meter_and_accumulator() -> None:
    meter = CacheAwareCostMeter(ILLUSTRATIVE)
    acc = EpisodeCostAccumulator()
    for t in range(3):
        acc.add(meter.call(ids(100 + 50 * t), 0, now=float(t)))
    summary = acc.summary()
    assert summary["llm_calls"] == 3
    assert summary["prompt_tokens_total"] == 100 + 150 + 200
    assert summary["cached_tokens_total"] == 0 + 100 + 150
    expected = (100 * 1.25 + (100 * 0.1 + 50 * 1.25) + (150 * 0.1 + 50 * 1.25)) * U
    assert summary["total_cost"] == pytest.approx(expected)
    meter.reset()
    assert meter.call(ids(100), 0, now=0.0).cached_tokens == 0


# --- exit gate: hand-computed scripted five-call episodes ----------------------


def test_worked_example_from_docstring() -> None:
    never = run_scripted(automatic(), "never_prune")
    window = run_scripted(automatic(), "window")
    assert [c.total_cost / U for c in never] == pytest.approx([187.5, 77.5, 82.5, 87.5, 92.5])
    assert [c.total_cost / U for c in window] == pytest.approx(
        [187.5, 77.5, 82.5, 197.5, 197.5]
    )
    assert sum(c.prompt_tokens for c in window) < sum(c.prompt_tokens for c in never)
    assert sum(c.total_cost for c in window) > sum(c.total_cost for c in never)


HAND_TABLES = {
    # profile: (never_prune per-call $, window per-call $)
    "anthropic-sonnet-5-5": (
        [0.00375, 0.0014, 0.00145, 0.0015, 0.00155],
        [0.00375, 0.0014, 0.00145, 0.00625, 0.00625],
    ),
    "openai-gpt-5-4": (
        [0.00375, 0.001832, 0.00193, 0.002028, 0.002126],
        [0.00375, 0.001832, 0.00193, 0.00625, 0.00625],
    ),
}


@pytest.mark.parametrize("name", sorted(HAND_TABLES))
def test_scripted_episode_matches_hand_computed_table(name: str) -> None:
    profile = get_profile(name)
    expected_never, expected_window = HAND_TABLES[name]
    never = run_scripted(profile, "never_prune", prefix=1000, item=500)
    window = run_scripted(profile, "window", prefix=1000, item=500)
    assert [c.total_cost for c in never] == pytest.approx(expected_never, abs=1e-12)
    assert [c.total_cost for c in window] == pytest.approx(expected_window, abs=1e-12)
    assert sum(c.total_cost for c in window) > sum(c.total_cost for c in never)


def test_demo_cli_runs(capsys: pytest.CaptureFixture[str]) -> None:
    assert cache_sim_main(["--demo", "--profile", "openai-gpt-5-4"]) == 0
    assert "never_prune" in capsys.readouterr().out
