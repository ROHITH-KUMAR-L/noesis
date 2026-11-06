"""Demo web API (FastAPI) over the real simulator, harness and policies.

    python -m memory_rl.web            # serves the API and the built UI on :8000

Every number the UI shows comes from these endpoints, which run the actual
code: the toy KeyHunt env, the scripted mock LLM, the prefix-cache simulator
and the trained checkpoints. Nothing is precomputed or invented client-side.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from memory_rl.baselines import BASELINES, build_baseline
from memory_rl.cost import (
    CacheAwareCostMeter,
    PrefixCacheSimulator,
    available_profiles,
    get_profile,
)
from memory_rl.cost.cache_sim import scripted_prompts
from memory_rl.envs.agent_loop import AgentEpisode, build_env, build_loop_config
from memory_rl.envs.policy_api import MemoryPolicy
from memory_rl.envs.toy import KeyHuntEnv
from memory_rl.llm.mock import ScriptedKeyHuntLLM
from memory_rl.memory import TiktokenTokenizer

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"
DIST = ROOT / "web" / "dist"

METHOD_LABELS = {
    "full_history": "Full history",
    "sliding_window": "Sliding window",
    "obs_masking": "Observation masking",
    "threshold_compaction": "Threshold compaction",
    "cadoc_rule": "CADOC rule",
    "random": "Random",
    "ppo_cache": "PPO (cache-aware reward)",
    "ppo_token": "PPO (token reward)",
}


@lru_cache(maxsize=1)
def _tokenizer() -> TiktokenTokenizer:
    return TiktokenTokenizer("o200k_base")


@lru_cache(maxsize=1)
def _env() -> KeyHuntEnv:
    env = build_env("toy")
    assert isinstance(env, KeyHuntEnv)
    return env


def _checkpoints() -> dict[str, Path]:
    """name -> best.pt for every finished training run under checkpoints/."""
    if not CHECKPOINTS.exists():
        return {}
    return {p.parent.name: p for p in sorted(CHECKPOINTS.glob("*/best.pt"))}


def _checkpoint_meta(path: Path) -> dict[str, Any]:
    history = path.parent / "history.json"
    if history.exists():
        meta = json.loads(history.read_text())["meta"]
        return {k: meta.get(k) for k in ("variant", "lam", "profile", "seed", "z")}
    return {}


def _policy(method: str, knob: float | None, checkpoint: str | None) -> MemoryPolicy:
    if method in ("ppo_cache", "ppo_token"):
        from memory_rl.controller.policy import load_policy

        runs = _checkpoints()
        if not checkpoint or checkpoint not in runs:
            raise HTTPException(404, f"unknown checkpoint {checkpoint!r}; train one first")
        return load_policy(runs[checkpoint])
    if method not in BASELINES:
        raise HTTPException(404, f"unknown method {method!r}")
    return build_baseline(method, knob)  # type: ignore[return-value]


app = FastAPI(title="Noesis demo API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    env = _env()
    loop = build_loop_config("toy")
    profiles = [
        {
            "name": p.name,
            "model": p.model,
            "mode": p.mode,
            "input_per_mtok": p.input_per_mtok,
            "output_per_mtok": p.output_per_mtok,
            "cache_read_multiplier": p.cache_read_multiplier,
            "cache_write_multiplier": p.cache_write_multiplier,
            "min_cacheable_tokens": p.min_cacheable_tokens,
            "block_size": p.block_size,
            "ttl_seconds": p.ttl_seconds,
            "source": p.source,
            "retrieved": p.retrieved,
            "illustrative": p.illustrative,
        }
        for p in available_profiles().values()
    ]
    methods = [
        {"id": name, "label": METHOD_LABELS.get(name, name), "knobs": list(grid)}
        for name, (_, grid) in BASELINES.items()
    ]
    checkpoints = [
        {"id": name, **_checkpoint_meta(path)} for name, path in _checkpoints().items()
    ]
    tasks = []
    for task_id in list(env.task_ids("valid"))[:40]:
        env.reset(task_id, 0)
        tasks.append({"id": task_id, "keys": len(env.task.projects)})
    return {
        "env": env.name,
        "llm": ScriptedKeyHuntLLM.name,
        "tokenizer": _tokenizer().name,
        "context_limit": loop.context_limit_tokens,
        "profiles": profiles,
        "methods": methods,
        "checkpoints": checkpoints,
        "tasks": tasks,
    }


class EpisodeRequest(BaseModel):
    task_id: str
    method: str
    knob: float | None = None
    checkpoint: str | None = None
    profile: str = "anthropic-sonnet-5-5"


@app.post("/api/episode")
def episode(req: EpisodeRequest) -> dict[str, Any]:
    env = _env()
    if req.task_id not in set(env.task_ids("valid")):
        raise HTTPException(404, "task must come from the validation split")
    try:
        profile = get_profile(req.profile)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    policy = _policy(req.method, req.knob, req.checkpoint)
    run = AgentEpisode(
        env, req.task_id, seed=0, llm=ScriptedKeyHuntLLM(), meter=CacheAwareCostMeter(profile),
        tokenizer=_tokenizer(), profile=profile, compressor=getattr(policy, "compressor", None),
        config=build_loop_config("toy"), split="valid",
    )  # fmt: skip
    run.run(policy)
    record = run.record(method=req.method, knob=str(req.knob))
    prompt = run.current_prompt()
    return {
        "task": {"id": req.task_id, "instruction": run.instruction, "keys": len(env.task.projects)},
        "prefix_tokens": prompt.prefix_tokens,
        "context_limit": run.config.context_limit_tokens,
        "turns": [
            {
                "turn": t.turn,
                "edit": t.edit,
                "actions": t.actions,
                "items": t.items,
                "prompt_tokens": t.prompt_tokens,
                "cost": t.cost,
                "llm_text": t.llm_text,
                "observation": t.observation,
                "terminal_reason": t.terminal_reason,
            }
            for t in run.turns
        ],
        "record": {
            k: record[k]
            for k in (
                "success", "terminal_reason", "llm_calls", "total_cost", "token_cost",
                "prompt_tokens_total", "cached_tokens_total", "written_tokens_total",
                "uncached_tokens_total", "output_tokens_total", "edits", "actions_compress",
                "actions_prune", "tokens_removed", "cache_read_cost", "cache_write_cost",
                "input_cost", "output_cost",
            )  # fmt: skip
        },
    }


class PlaygroundRequest(BaseModel):
    profile: str = "anthropic-sonnet-5-5"
    prefix: int = Field(1500, ge=0, le=20_000)
    item: int = Field(400, ge=1, le=10_000)
    calls: int = Field(10, ge=1, le=40)
    strategy: Literal["never_prune", "window", "compact_once", "trim_every_turn"] = "never_prune"
    window: int = Field(3, ge=1, le=40)
    compact_at: int = Field(6, ge=1, le=40)


def _prompts(req: PlaygroundRequest) -> list[list[int]]:
    if req.strategy in ("never_prune", "window"):
        return scripted_prompts(
            req.strategy, calls=req.calls, prefix=req.prefix, item=req.item, window=req.window
        )
    protected = list(range(1, req.prefix + 1))
    prompts = []
    for call in range(1, req.calls + 1):
        items = list(range(1, call + 1))
        if req.strategy == "compact_once" and call >= req.compact_at:
            items = [k for k in items if k > req.compact_at - req.window]  # one big cut
        elif req.strategy == "trim_every_turn" and call > 1:
            items = items[1:]  # drop the oldest item, every call
            items = [k for k in items if k >= call - req.window - 1] or items[-1:]
        prompts.append(protected + [k * 100_000 + j for k in items for j in range(req.item)])
    return prompts


@app.post("/api/playground")
def playground(req: PlaygroundRequest) -> dict[str, Any]:
    try:
        profile = get_profile(req.profile)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    results: dict[str, Any] = {}
    for strategy in ("never_prune", req.strategy):
        sim = PrefixCacheSimulator(profile)
        variant = req.model_copy(update={"strategy": strategy})
        calls = []
        for index, prompt in enumerate(_prompts(variant)):
            breakpoints = (
                sorted({req.prefix, len(prompt)} - {0}) if profile.mode == "breakpoint" else None
            )
            cost = sim.call(prompt, output_tokens=60, now=float(index) * 5, breakpoints=breakpoints)
            calls.append(cost.to_dict())
        results[strategy] = {
            "calls": calls,
            "total_cost": sum(c["total_cost"] for c in calls),
            "total_tokens": sum(c["prompt_tokens"] for c in calls),
        }
    return {"profile": profile.name, "baseline": "never_prune", "strategy": req.strategy, **results}


@app.get("/api/results")
def results() -> dict[str, Any]:
    runs = []
    for path in sorted(RESULTS.glob("*/summary.json")):
        data = json.loads(path.read_text())
        runs.append(
            {
                "name": path.parent.name,
                "split": data.get("split"),
                "tasks": data.get("tasks"),
                "seeds": data.get("seeds"),
                "llm": data.get("llm"),
                "note": data.get("note"),
                "git_commit": data.get("git_commit"),
                "rows": data.get("rows", []),
            }
        )
    return {"runs": runs, "labels": METHOD_LABELS}


if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="ui")
