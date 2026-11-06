"""Aggregate per-episode records (plan.md §7 Phase 8, prd.md §8 metrics).

Every number here is computed from records; nothing is estimated. Confidence
intervals are percentile bootstrap over tasks (episodes are resampled with
their seeds kept together per task).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

GROUP_KEYS = ("method", "variant", "knob", "env", "split", "pricing_profile")


def bootstrap_ci(
    values_by_task: Sequence[Sequence[float]],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean and (1-alpha) percentile CI of the per-task means, resampling tasks."""
    task_means = np.array([float(np.mean(v)) for v in values_by_task], dtype=float)
    if task_means.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, task_means.size, size=(n_boot, task_means.size))
    boots = task_means[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(task_means.mean()), float(lo), float(hi)


def paired_difference_ci(
    a: Mapping[str, float], b: Mapping[str, float], *, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of (a − b) over shared tasks with a bootstrap CI (paired by task)."""
    shared = sorted(set(a) & set(b))
    diffs = [[a[t] - b[t]] for t in shared]
    return bootstrap_ci(diffs, n_boot=n_boot, seed=seed)


def _by_task(records: Sequence[Mapping[str, Any]], key: str) -> list[list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[record["task_id"]].append(float(record[key]))
    return [grouped[t] for t in sorted(grouped)]


def aggregate(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record[k] for k in GROUP_KEYS)].append(record)
    rows = []
    for key, recs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        row: dict[str, Any] = dict(zip(GROUP_KEYS, key, strict=True))
        n = len(recs)
        successes = sum(bool(r["success"]) for r in recs)
        row["episodes"] = n
        row["tasks"] = len({r["task_id"] for r in recs})
        row["seeds"] = sorted({r["seed"] for r in recs})
        for name, field in (
            ("success", "success"),
            ("cost", "total_cost"),
            ("tokens", "prompt_tokens_total"),
        ):
            mean, lo, hi = bootstrap_ci(_by_task(recs, field))
            row[f"{name}_mean"], row[f"{name}_lo"], row[f"{name}_hi"] = mean, lo, hi
        prompt = sum(r["prompt_tokens_total"] for r in recs)
        row["cached_fraction"] = sum(r["cached_tokens_total"] for r in recs) / prompt
        row["llm_calls_mean"] = float(np.mean([r["llm_calls"] for r in recs]))
        edits = sum(r["edits"] for r in recs)
        row["edits_mean"] = edits / n
        row["tokens_removed_per_edit"] = (
            sum(r["tokens_removed"] for r in recs) / edits if edits else 0.0
        )
        total_cost = sum(r["total_cost"] for r in recs)
        row["cost_per_success"] = total_cost / successes if successes else math.inf
        reasons: dict[str, int] = defaultdict(int)
        for r in recs:
            reasons[r["terminal_reason"]] += 1
        row["terminal_reasons"] = dict(reasons)
        per_seed: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for r in recs:
            per_seed[r["seed"]].append(r)
        row["per_seed_success"] = {
            s: sum(bool(r["success"]) for r in rs) / len(rs) for s, rs in sorted(per_seed.items())
        }
        row["per_seed_cost"] = {
            s: float(np.mean([r["total_cost"] for r in rs])) for s, rs in sorted(per_seed.items())
        }
        rows.append(row)
    return rows


def format_table(rows: Sequence[Mapping[str, Any]]) -> str:
    header = (
        f"{'method':<22}{'knob':>7}{'profile':>22}{'success':>16}{'cost $ (95% CI)':>28}"
        f"{'tokens':>9}{'cache%':>8}{'edits':>7}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['method']:<22}{str(r['knob']):>7}{r['pricing_profile']:>22}"
            f"{r['success_mean']:>9.2f} ±{(r['success_hi'] - r['success_lo']) / 2:4.2f}"
            f"{r['cost_mean']:>14.5f} [{r['cost_lo']:.5f},{r['cost_hi']:.5f}]"
            f"{r['tokens_mean']:>9.0f}{100 * r['cached_fraction']:>7.1f}%{r['edits_mean']:>7.2f}"
        )
    return "\n".join(lines)
