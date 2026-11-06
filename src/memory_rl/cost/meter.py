"""Cost meters used by the agent loop and the reward (plan.md §6.4, §6.7).

``CacheAwareCostMeter`` prices calls through the prefix-cache simulator in $.
``TokenCostMeter`` has the same interface but ignores caching: its
``total_cost`` is prompt tokens (plus ``output_weight × output tokens``). It is
the training signal of the token-reward comparison controller; frontiers are
always compared on the cache-aware $ cost.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from memory_rl.cost.cache_sim import CallCost, PrefixCacheSimulator
from memory_rl.cost.pricing import PricingProfile

CostVariant = Literal["cache", "token"]


class CostMeter(Protocol):
    @property
    def variant(self) -> CostVariant: ...

    @property
    def unit(self) -> str: ...

    def reset(self) -> None: ...

    def call(
        self,
        prompt_ids: Sequence[int],
        output_tokens: int,
        now: float,
        *,
        breakpoints: Sequence[int] | None = None,
    ) -> CallCost: ...


class CacheAwareCostMeter:
    """Real-cost meter: $ after prefix-caching discounts."""

    variant: CostVariant = "cache"
    unit = "usd"

    def __init__(self, profile: PricingProfile) -> None:
        self.profile = profile
        self._sim = PrefixCacheSimulator(profile)

    def reset(self) -> None:
        self._sim.reset()

    def call(
        self,
        prompt_ids: Sequence[int],
        output_tokens: int,
        now: float,
        *,
        breakpoints: Sequence[int] | None = None,
    ) -> CallCost:
        if self.profile.mode == "automatic":
            breakpoints = None
        return self._sim.call(prompt_ids, output_tokens, now, breakpoints=breakpoints)


class TokenCostMeter:
    """Token-count meter: caching is ignored on purpose."""

    variant: CostVariant = "token"
    unit = "tokens"

    def __init__(self, output_weight: float = 0.0) -> None:
        if output_weight < 0:
            raise ValueError("output_weight must be >= 0")
        self.output_weight = output_weight

    def reset(self) -> None:
        return None

    def call(
        self,
        prompt_ids: Sequence[int],
        output_tokens: int,
        now: float,
        *,
        breakpoints: Sequence[int] | None = None,
    ) -> CallCost:
        prompt = len(prompt_ids)
        output_cost = self.output_weight * output_tokens
        return CallCost(
            prompt_tokens=prompt,
            cached_tokens=0,
            written_tokens=0,
            uncached_tokens=prompt,
            output_tokens=output_tokens,
            input_cost=float(prompt),
            cache_read_cost=0.0,
            cache_write_cost=0.0,
            output_cost=output_cost,
            total_cost=prompt + output_cost,
        )


@dataclass(slots=True)
class EpisodeCostAccumulator:
    """Sums every call of an episode (invariant §4.2.11)."""

    calls: list[CallCost] = field(default_factory=list)

    def add(self, cost: CallCost) -> None:
        self.calls.append(cost)

    @property
    def llm_calls(self) -> int:
        return len(self.calls)

    def total(self, attribute: str) -> float:
        return float(sum(getattr(cost, attribute) for cost in self.calls))

    def summary(self) -> dict[str, float | int]:
        keys = (
            "prompt_tokens",
            "cached_tokens",
            "written_tokens",
            "uncached_tokens",
            "output_tokens",
        )
        out: dict[str, float | int] = {f"{key}_total": int(self.total(key)) for key in keys}
        for key in ("input_cost", "cache_read_cost", "cache_write_cost", "output_cost"):
            out[key] = self.total(key)
        out["total_cost"] = self.total("total_cost")
        out["llm_calls"] = self.llm_calls
        prompt = out["prompt_tokens_total"]
        out["cached_fraction"] = out["cached_tokens_total"] / prompt if prompt else 0.0
        return out
