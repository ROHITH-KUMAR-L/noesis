"""PPO-clip with GAE for the memory controller (plan.md §7 Phase 6).

Starting hyper-parameters (pipeline validation values, not tuned optima):
ε = 0.2, γ = 1.0 (episodic), λ_GAE = 0.95, entropy 0.01, value 0.5, Adam 3e-4,
grad-norm 0.5, 4 epochs, minibatches over turns. Fails loudly on non-finite
losses, advantages or weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from memory_rl.controller.env import ControllerEnv
from memory_rl.controller.features import Observation
from memory_rl.controller.policy import ControllerNet, ObsBatch, act, evaluate_actions


class NonFiniteError(FloatingPointError):
    pass


@dataclass(frozen=True, slots=True)
class PPOConfig:
    episodes_per_update: int = 32
    epochs: int = 4
    minibatch_size: int = 128
    clip: float = 0.2
    gamma: float = 1.0
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Rollout:
    obs: list[Observation] = field(default_factory=list)
    gate: list[int] = field(default_factory=list)
    items: list[np.ndarray] = field(default_factory=list)
    log_prob: list[float] = field(default_factory=list)
    value: list[float] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    done: list[bool] = field(default_factory=list)
    episode_returns: list[float] = field(default_factory=list)
    episode_success: list[bool] = field(default_factory=list)
    episode_edits: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.reward)


def compute_gae(
    rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, gamma: float, lam: float
) -> tuple[np.ndarray, np.ndarray]:
    """Episodic GAE; a terminal step never bootstraps from the next state."""
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float64)
    last = 0.0
    for t in range(n - 1, -1, -1):
        nonterminal = 0.0 if dones[t] else 1.0
        next_value = values[t + 1] if t + 1 < n else 0.0
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    returns = adv + values
    if not (np.isfinite(adv).all() and np.isfinite(returns).all()):
        raise NonFiniteError("non-finite advantages or returns")
    return adv, returns


def collect(
    cenv: ControllerEnv,
    net: ControllerNet,
    task_ids: list[str],
    *,
    rng: np.random.Generator,
    generator: torch.Generator,
    episodes: int,
) -> Rollout:
    rollout = Rollout()
    net.eval()
    for _ in range(episodes):
        task_id = task_ids[int(rng.integers(len(task_ids)))]
        obs: Observation | None = cenv.reset(task_id, seed=int(rng.integers(2**31)))
        done = False
        info: dict[str, Any] = {}
        while not done:
            assert obs is not None
            gate, items, logp, value = act(net, obs, greedy=False, generator=generator)
            next_obs, reward, done, info = cenv.step(gate, items)
            rollout.obs.append(obs)
            rollout.gate.append(gate)
            rollout.items.append(items)
            rollout.log_prob.append(logp)
            rollout.value.append(value)
            rollout.reward.append(reward)
            rollout.done.append(done)
            obs = next_obs
        rollout.episode_returns.append(float(info["episode_return"]))
        rollout.episode_success.append(bool(info["success"]))
        assert cenv.episode is not None
        rollout.episode_edits.append(cenv.episode.counters.edits)
    return rollout


def update(
    net: ControllerNet,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    config: PPOConfig,
    *,
    rng: np.random.Generator,
) -> dict[str, float]:
    adv, returns = compute_gae(
        np.asarray(rollout.reward, dtype=np.float64),
        np.asarray(rollout.value, dtype=np.float64),
        np.asarray(rollout.done, dtype=bool),
        config.gamma,
        config.gae_lambda,
    )
    adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
    batch = ObsBatch.from_observations(rollout.obs)
    gate = torch.as_tensor(rollout.gate, dtype=torch.long)
    items = torch.as_tensor(np.stack(rollout.items), dtype=torch.long)
    old_logp = torch.as_tensor(rollout.log_prob, dtype=torch.float32)
    adv_t = torch.as_tensor(adv_n, dtype=torch.float32)
    ret_t = torch.as_tensor(returns, dtype=torch.float32)

    net.train()
    stats: dict[str, list[float]] = {"policy_loss": [], "value_loss": [], "entropy": [], "kl": []}
    n = len(rollout)
    for _ in range(config.epochs):
        order = rng.permutation(n)
        for start in range(0, n, config.minibatch_size):
            idx = torch.as_tensor(order[start : start + config.minibatch_size])
            mb = batch.index(idx)
            out = net(mb)
            joint = evaluate_actions(out, mb, gate[idx], items[idx])
            ratio = torch.exp(joint.log_prob - old_logp[idx])
            a = adv_t[idx]
            policy_loss = -torch.min(
                ratio * a, torch.clamp(ratio, 1 - config.clip, 1 + config.clip) * a
            ).mean()
            value_loss = 0.5 * (joint.value - ret_t[idx]).pow(2).mean()
            entropy = joint.entropy.mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
            if not torch.isfinite(loss):
                raise NonFiniteError("non-finite PPO loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.max_grad_norm)
            optimizer.step()
            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy"].append(entropy.item())
            stats["kl"].append((old_logp[idx] - joint.log_prob).mean().item())
    for param in net.parameters():
        if not torch.isfinite(param).all():
            raise NonFiniteError("non-finite network weights after update")
    return {k: float(np.mean(v)) for k, v in stats.items()}
