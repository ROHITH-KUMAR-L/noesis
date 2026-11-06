"""Train the PPO memory controller (plan.md §7 Phase 6).

    python -m memory_rl.controller.train --config configs/ppo.yaml --variant cache --seed 0
    python -m memory_rl.controller.train --smoke

Training uses TRAIN tasks only; checkpoint selection uses deterministic (argmax)
evaluation on VALID tasks, scored by the variant's own training objective. The
test split is never touched here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from memory_rl.controller.env import ControllerEnv
from memory_rl.controller.features import SCHEMA_VERSION
from memory_rl.controller.policy import (
    ControllerNet,
    NetConfig,
    PPOMemoryPolicy,
    save_checkpoint,
)
from memory_rl.controller.ppo import PPOConfig, collect, update
from memory_rl.controller.reward import RewardConfig, calibrate_z, step_cost
from memory_rl.cost import CacheAwareCostMeter, get_profile
from memory_rl.envs.agent_loop import AgentEpisode, build_env, build_loop_config
from memory_rl.evaluation.records import config_hash, git_commit
from memory_rl.llm.mock import ScriptedKeyHuntLLM
from memory_rl.memory import TiktokenTokenizer


@dataclass(frozen=True, slots=True)
class TrainConfig:
    env: str = "toy"
    profile: str = "anthropic-sonnet-5-5"
    variant: str = "cache"
    lam: float = 1.0
    seed: int = 0
    updates: int = 150
    validate_every: int = 10
    valid_tasks: int = 30
    tokenizer: str = "o200k_base"
    out: str = "checkpoints"
    keep_bias: float = 3.0
    no_edit_bias: float = 1.0
    tag: str = ""


def _load(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must be a YAML mapping")
    return data


def evaluate_policy(
    policy: PPOMemoryPolicy,
    make_episode: Any,
    task_ids: Sequence[str],
    reward: RewardConfig,
) -> dict[str, float]:
    successes, returns, costs, edits = [], [], [], []
    for task_id in task_ids:
        episode: AgentEpisode = make_episode(task_id, 0)
        episode.run(policy)
        cost_variant = sum(step_cost(t, reward.variant) for t in episode.turns)
        real_cost = sum(float(t.cost["total_cost"]) for t in episode.turns if t.cost)
        successes.append(float(episode.success))
        task_term = float(episode.success) * reward.task_reward
        returns.append(task_term - reward.lam * cost_variant / reward.z)
        costs.append(real_cost)
        edits.append(episode.counters.edits)
    return {
        "success": float(np.mean(successes)),
        "objective": float(np.mean(returns)),
        "real_cost": float(np.mean(costs)),
        "edits": float(np.mean(edits)),
    }


def train(config: TrainConfig, ppo: PPOConfig, *, log: Any = print) -> Path:
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    generator = torch.Generator().manual_seed(config.seed)

    env = build_env(config.env)
    loop = build_loop_config(config.env)
    profile = get_profile(config.profile)
    tokenizer = TiktokenTokenizer(config.tokenizer)
    llm = ScriptedKeyHuntLLM()

    def make_episode(task_id: str, seed: int, split: str = "train") -> AgentEpisode:
        return AgentEpisode(
            env, task_id, seed=seed, llm=llm, meter=CacheAwareCostMeter(profile),
            tokenizer=tokenizer, profile=profile, config=loop, split=split,
        )  # fmt: skip

    z = calibrate_z(
        env, llm=llm, tokenizer=tokenizer, profile=profile, variant=config.variant,
        loop_config=loop,
    )  # fmt: skip
    reward = RewardConfig(config.variant, z=z, lam=config.lam)
    cenv = ControllerEnv(make_episode, reward)
    train_ids = list(env.task_ids("train"))
    valid_ids = list(env.task_ids("valid"))[: config.valid_tasks]

    net = ControllerNet(NetConfig(keep_bias=config.keep_bias, no_edit_bias=config.no_edit_bias))
    optimizer = torch.optim.Adam(net.parameters(), lr=ppo.lr)
    run_name = f"ppo_{config.variant}_lam{config.lam:g}_{config.profile}_s{config.seed}{config.tag}"
    out_dir = Path(config.out) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "variant": config.variant,
        "lam": config.lam,
        "z": z,
        "profile": config.profile,
        "seed": config.seed,
        "env": config.env,
        "llm": llm.name,
        "tokenizer": tokenizer.name,
        "schema_version": SCHEMA_VERSION,
        "loop": loop.to_dict(),
        "ppo": ppo.to_dict(),
        "git_commit": git_commit(),
    }
    meta["config_hash"] = config_hash({**meta, "train": asdict(config)})
    history: list[dict[str, Any]] = []
    best = -float("inf")
    best_path = out_dir / "best.pt"

    def valid_episode(task_id: str, seed: int) -> AgentEpisode:
        return make_episode(task_id, seed, split="valid")

    started = time.time()
    for step in range(1, config.updates + 1):
        rollout = collect(
            cenv, net, train_ids, rng=rng, generator=generator, episodes=ppo.episodes_per_update
        )
        stats = update(net, optimizer, rollout, ppo, rng=rng)
        row: dict[str, Any] = {
            "update": step,
            "train_return": float(np.mean(rollout.episode_returns)),
            "train_success": float(np.mean(rollout.episode_success)),
            "train_edits": float(np.mean(rollout.episode_edits)),
            **stats,
            "elapsed_s": round(time.time() - started, 1),
        }
        if step % config.validate_every == 0 or step == config.updates:
            policy = PPOMemoryPolicy(net, name=run_name)
            val = evaluate_policy(policy, valid_episode, valid_ids, reward)
            row.update({f"valid_{k}": v for k, v in val.items()})
            if val["objective"] > best:
                best = val["objective"]
                save_checkpoint(best_path, net, {**meta, "update": step, "valid": val})
            log(json.dumps(row))
        history.append(row)
    save_checkpoint(out_dir / "last.pt", net, {**meta, "update": config.updates})
    (out_dir / "history.json").write_text(json.dumps({"meta": meta, "history": history}, indent=2))
    return best_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the PPO memory controller")
    parser.add_argument("--config")
    parser.add_argument("--variant", choices=["cache", "token"])
    parser.add_argument("--lam", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--profile")
    parser.add_argument("--out")
    parser.add_argument("--keep-bias", dest="keep_bias", type=float)
    parser.add_argument("--tag")
    parser.add_argument("--smoke", action="store_true", help="2 tiny updates for CI")
    args = parser.parse_args(argv)
    cfg = _load(args.config)
    train_cfg = {k: v for k, v in cfg.items() if k in TrainConfig.__dataclass_fields__}
    for key in ("variant", "lam", "seed", "updates", "profile", "out", "keep_bias", "tag"):
        value = getattr(args, key)
        if value is not None:
            train_cfg[key] = value
    ppo_cfg = PPOConfig(**cfg.get("ppo", {}))
    if args.smoke:
        train_cfg.update(updates=2, validate_every=1, valid_tasks=2)
        ppo_cfg = PPOConfig(episodes_per_update=2, epochs=1, minibatch_size=16)
    torch.set_num_threads(1)
    path = train(TrainConfig(**train_cfg), ppo_cfg)
    print(f"best checkpoint: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
