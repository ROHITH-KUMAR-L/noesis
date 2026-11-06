"""Controller reward (plan.md §6.7).

    r_t  = −λ · cost_t / Z            per LLM call
    r_T += R_task                     terminal (KeyHunt: success ∈ {0, 1})
    r_T −= overflow_penalty           if the episode ended in context overflow

``cost_t`` is the cache-adjusted $ cost (``variant="cache"``) or the prompt-token
count (``variant="token"``). ``Z`` is the mean full-history episode cost on a
fixed calibration subset of TRAIN tasks, computed per env × profile × variant.
There is no reward for edits in themselves; every term is logged separately.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from memory_rl.cost import CacheAwareCostMeter, PricingProfile
from memory_rl.envs.agent_loop import AgentEpisode, LoopConfig, TurnTrace
from memory_rl.envs.base import AgentEnv, TerminalReason
from memory_rl.envs.policy_api import AlwaysKeepPolicy
from memory_rl.llm.backend import LLMBackend
from memory_rl.memory import EncodingTokenizer

CALIBRATION_TASKS = 32


@dataclass(frozen=True, slots=True)
class RewardConfig:
    variant: str  # "cache" | "token"
    z: float
    lam: float = 1.0
    task_reward: float = 1.0
    overflow_penalty: float = 0.0

    def __post_init__(self) -> None:
        if self.variant not in ("cache", "token"):
            raise ValueError("variant must be 'cache' or 'token'")
        if not (math.isfinite(self.z) and self.z > 0):
            raise ValueError("z must be a finite positive normaliser")
        if self.lam < 0 or self.overflow_penalty < 0:
            raise ValueError("lam and overflow_penalty must be >= 0")

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RewardTerms:
    cost: float
    task: float
    overflow: float

    @property
    def total(self) -> float:
        return self.cost + self.task + self.overflow


def step_cost(trace: TurnTrace, variant: str) -> float:
    if trace.cost is None:  # overflow turn: nothing was sent
        return 0.0
    if variant == "cache":
        return float(trace.cost["total_cost"])
    assert trace.token_cost is not None
    return float(trace.token_cost)


def turn_reward(trace: TurnTrace, config: RewardConfig, *, success: bool) -> RewardTerms:
    cost = -config.lam * step_cost(trace, config.variant) / config.z
    task = overflow = 0.0
    if trace.terminal_reason is not None:
        task = config.task_reward if success else 0.0
        if trace.terminal_reason == TerminalReason.CONTEXT_OVERFLOW:
            overflow = -config.overflow_penalty
    return RewardTerms(cost=cost, task=task, overflow=overflow)


def calibrate_z(
    env: AgentEnv,
    *,
    llm: LLMBackend,
    tokenizer: EncodingTokenizer,
    profile: PricingProfile,
    variant: str,
    task_ids: Sequence[str] | None = None,
    loop_config: LoopConfig | None = None,
) -> float:
    """Mean full-history episode cost on the first ``CALIBRATION_TASKS`` train tasks."""
    ids = list(task_ids if task_ids is not None else env.task_ids("train")[:CALIBRATION_TASKS])
    totals = []
    for task_id in ids:
        episode = AgentEpisode(
            env, task_id, seed=0, llm=llm, meter=CacheAwareCostMeter(profile),
            tokenizer=tokenizer, profile=profile, config=loop_config, split="train",
        )  # fmt: skip
        episode.run(AlwaysKeepPolicy())
        totals.append(sum(step_cost(t, variant) for t in episode.turns))
    z = sum(totals) / len(totals)
    if not z > 0:
        raise ValueError("calibration produced a non-positive Z")
    return z
