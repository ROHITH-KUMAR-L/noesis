"""Common evaluation runner (plan.md §7 Phases 5 and 8).

    python -m memory_rl.evaluation.run_eval --config configs/baselines.yaml --overwrite
    python -m memory_rl.evaluation.run_eval --method obs_masking --knob 1 --split valid

Every method runs through the same harness: same tasks, LLM, prompt layout,
tokenizer, cost model and pricing profile. Writes ``records.jsonl`` and
``summary.json`` to the output directory and refuses to overwrite without
``--overwrite``. Tune on ``valid``; touch ``test`` once.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml

from memory_rl.baselines import BASELINES, build_baseline
from memory_rl.cost import CacheAwareCostMeter, PricingProfile, get_profile
from memory_rl.envs.agent_loop import (
    AgentEpisode,
    LoopConfig,
    build_env,
    build_llm,
    build_loop_config,
)
from memory_rl.envs.base import AgentEnv
from memory_rl.envs.policy_api import MemoryPolicy
from memory_rl.evaluation.metrics import aggregate, format_table
from memory_rl.evaluation.records import git_commit, write_jsonl
from memory_rl.llm.backend import LLMBackend
from memory_rl.memory import Compressor, EncodingTokenizer, TiktokenTokenizer

DEFAULT_TOKENIZER = "o200k_base"


def run_policy_on_tasks(
    make_policy: Callable[[], MemoryPolicy],
    *,
    method: str,
    knob: Any,
    env: AgentEnv,
    task_ids: Sequence[str],
    seeds: Sequence[int],
    split: str,
    profile: PricingProfile,
    llm: LLMBackend,
    tokenizer: EncodingTokenizer,
    loop_config: LoopConfig | None = None,
    variant: str = "-",
    keep_episodes: bool = False,
) -> tuple[list[dict[str, Any]], list[AgentEpisode]]:
    records, episodes = [], []
    for seed in seeds:
        for task_id in task_ids:
            policy = make_policy()
            compressor: Compressor | None = getattr(policy, "compressor", None)
            episode = AgentEpisode(
                env, task_id, seed=seed, llm=llm, meter=CacheAwareCostMeter(profile),
                tokenizer=tokenizer, profile=profile, compressor=compressor,
                config=loop_config, split=split,
            )  # fmt: skip
            episode.run(policy)
            records.append(
                episode.record(method=method, variant=variant, knob=str(knob))
            )
            if keep_episodes:
                episodes.append(episode)
    return records, episodes


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must be a YAML mapping")
    return data


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate memory policies on the agent harness")
    parser.add_argument("--config")
    parser.add_argument("--method", help="single method (overrides config methods)")
    parser.add_argument("--knob", type=float)
    parser.add_argument("--checkpoint", help="PPO checkpoint for --method ppo")
    parser.add_argument("--env")
    parser.add_argument("--llm", choices=["mock", "openai"])
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--split", choices=["train", "valid", "test"])
    parser.add_argument("--profiles", nargs="+")
    parser.add_argument("--tasks", type=int, help="first N tasks of the split")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--out")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="2 tasks, 1 seed, one profile")
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    split = args.split or cfg.get("split", "valid")
    if split == "test" and not (args.allow_paid or cfg.get("confirm_test", False)):
        if not cfg.get("confirm_test"):
            print("note: running on the TEST split; this should happen once (plan.md §8)")
    env_name = args.env or cfg.get("env", "toy")
    env = build_env(env_name, **cfg.get("env_overrides", {}))
    llm = build_llm(args.llm or cfg.get("llm", "mock"), allow_paid=args.allow_paid)
    profiles = args.profiles or cfg.get("profiles", ["anthropic-sonnet-5-5"])
    n_tasks = args.tasks or cfg.get("tasks", 60)
    seeds = args.seeds or cfg.get("seeds", [0, 1, 2])
    loop_config = build_loop_config(env_name, **cfg.get("loop", {}))
    if args.smoke:
        n_tasks, seeds, profiles = 2, [0], profiles[:1]
    task_ids = list(env.task_ids(split))[:n_tasks]

    if args.method:
        grid = {args.method: [args.knob] if args.knob is not None else None}
    else:
        grid = cfg.get("methods") or {m: None for m in BASELINES}
    tokenizer = TiktokenTokenizer(cfg.get("tokenizer", DEFAULT_TOKENIZER))

    out_dir = Path(args.out or cfg.get("out", f"results/eval_{split}"))
    records_path = out_dir / "records.jsonl"
    if records_path.exists() and not args.overwrite:
        raise SystemExit(f"{records_path} exists; pass --overwrite to replace it")

    started = time.time()
    all_records: list[dict[str, Any]] = []
    for profile_name in profiles:
        profile = get_profile(profile_name)
        for method, knobs in grid.items():
            checkpoint = args.checkpoint or cfg.get("checkpoint")
            make, knob_list = _policy_factory(method, knobs, checkpoint)
            for knob in knob_list:
                records, _ = run_policy_on_tasks(
                    functools.partial(make, knob),
                    method=method, knob=knob if knob is not None else "-",
                    env=env, task_ids=task_ids, seeds=seeds, split=split, profile=profile,
                    llm=llm, tokenizer=tokenizer, loop_config=loop_config,
                    variant=_variant(method),
                )  # fmt: skip
                all_records.extend(records)
                print(f"done {profile_name} {method} knob={knob} ({len(records)} episodes)")

    write_jsonl(records_path, all_records, overwrite=args.overwrite)
    rows = aggregate(all_records)
    summary = {
        "split": split,
        "profiles": profiles,
        "tasks": n_tasks,
        "seeds": seeds,
        "llm": llm.name,
        "tokenizer": tokenizer.name,
        "git_commit": git_commit(),
        "elapsed_s": round(time.time() - started, 1),
        "note": "toy env + mock LLM unless llm says otherwise; not an ALFWorld result",
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(format_table(rows))
    print(f"wrote {records_path} and {out_dir / 'summary.json'}")
    return 0


def _variant(method: str) -> str:
    if method.startswith("ppo"):
        return method.split("_", 1)[1] if "_" in method else "cache"
    return "-"


def _policy_factory(
    method: str, knobs: Sequence[Any] | None, checkpoint: str | None
) -> tuple[Callable[[Any], MemoryPolicy], list[Any]]:
    if method.startswith("ppo"):
        from memory_rl.controller.policy import load_policy

        if not checkpoint:
            raise SystemExit("--method ppo needs --checkpoint")
        return (lambda _knob: load_policy(checkpoint)), [Path(checkpoint).stem]
    if method not in BASELINES:
        raise SystemExit(f"unknown method {method!r}")
    default_grid: list[Any] = list(BASELINES[method][1]) or [None]
    knob_list = list(knobs) if knobs else default_grid
    return (lambda knob: build_baseline(method, knob)), knob_list  # type: ignore[return-value]


if __name__ == "__main__":
    sys.exit(main())
