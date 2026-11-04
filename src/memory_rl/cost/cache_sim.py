"""Prefix-cache cost simulator (plan.md §6.4, Phase 2).

Every LLM call is priced from the exact token IDs sent. The simulator keeps the
live cache entries of one episode and splits each prompt into

* ``cached``   — tokens served from a live entry (billed at ``input × m_read``),
* ``written``  — tokens newly stored in the cache (billed at ``input × m_write``),
* ``uncached`` — the rest (billed at the plain input price),

plus output tokens at the output price.

Two cache semantics are simulated (``PricingProfile.mode``):

``automatic`` (OpenAI-style)
    ``cached`` is the longest common prefix with any live entry, rounded down to
    ``block_size`` and zeroed below ``min_cacheable_tokens``. After the call the
    whole prompt becomes an entry and ``written = prompt − cached``.

``breakpoint`` (Anthropic ``cache_control``-style)
    Entries are written only at the caller's breakpoints (default: the end of
    the prompt; at most ``max_breakpoints``). A read hits only when a *whole*
    entry is a prefix of the new prompt — reads land on positions a previous
    request wrote. ``written`` runs from ``cached`` to the last breakpoint that
    is at least ``min_cacheable_tokens`` long; anything after it is uncached.
    This is stricter than the plan's LCP wording for this mode and is the
    documented provider behaviour; the deviation is intentional.

Entries expire ``ttl_seconds`` after their last use; a read refreshes them. The
cache is per episode (``reset()``), so the first call is always cold.

Worked example — "prune a little every turn" vs "never prune"
-------------------------------------------------------------
Profile ``illustrative-test`` (input price ``u`` per token, read 0.1u, write
1.25u), automatic semantics, a 100-token protected prefix ``P`` and one new
50-token history item per call. Costs in units of ``u``:

=====  ======================  ==========  ===============  =======================  ==========
call   never prune (prompt)    cached      cost             window of 3 (prompt)      cost
=====  ======================  ==========  ===============  =======================  ==========
1      P+i1        = 150       0           150·1.25 = 187.5  P+i1          = 150       187.5
2      P+i1..i2    = 200       150         15 + 62.5 = 77.5  P+i1..i2      = 200        77.5
3      P+i1..i3    = 250       200         20 + 62.5 = 82.5  P+i1..i3      = 250        82.5
4      P+i1..i4    = 300       250         25 + 62.5 = 87.5  P+i2..i4      = 250       197.5
5      P+i1..i5    = 350       300         30 + 62.5 = 92.5  P+i3..i5      = 250       197.5
total  1250 tokens                         527.5             1100 tokens               742.5
=====  ======================  ==========  ===============  =======================  ==========

From call 4 the window drops ``i1``, so the prompt diverges right after ``P``:
only 100 tokens are cached (10u) and 150 are re-written (187.5u). The window
sends **12% fewer tokens but costs 41% more** than keeping everything. This is
the effect the cache-aware controller is meant to exploit
(``tests/test_cost_model.py::test_worked_example_from_docstring``).
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from memory_rl.cost.pricing import PricingProfile, get_profile

#: Anthropic allows at most four cache_control breakpoints per request.
DEFAULT_MAX_BREAKPOINTS = 4


@dataclass(frozen=True, slots=True)
class CallCost:
    """Token split and cost terms of one LLM call. ``total = Σ cost terms``."""

    prompt_tokens: int
    cached_tokens: int
    written_tokens: int
    uncached_tokens: int
    output_tokens: int
    input_cost: float
    cache_read_cost: float
    cache_write_cost: float
    output_cost: float
    total_cost: float

    def __post_init__(self) -> None:
        parts = self.cached_tokens + self.written_tokens + self.uncached_tokens
        if min(self.cached_tokens, self.written_tokens, self.uncached_tokens) < 0:
            raise ValueError("token counts must be non-negative")
        if parts != self.prompt_tokens:
            raise ValueError(
                f"cached+written+uncached ({parts}) must equal prompt_tokens "
                f"({self.prompt_tokens})"
            )
        terms = (self.input_cost, self.cache_read_cost, self.cache_write_cost, self.output_cost)
        if not all(math.isfinite(value) and value >= 0 for value in (*terms, self.total_cost)):
            raise ValueError("cost terms must be finite and non-negative")

    @property
    def cached_fraction(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(slots=True)
class _Entry:
    ids: np.ndarray
    last_used: float


def _as_array(prompt_ids: Sequence[int]) -> np.ndarray:
    array = np.asarray(prompt_ids, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError("prompt_ids must be a flat sequence of token IDs")
    return array


def common_prefix_length(a: np.ndarray, b: np.ndarray) -> int:
    """Length of the longest common prefix of two token-ID arrays."""
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return 0
    mismatch = np.flatnonzero(a[:n] != b[:n])
    return int(mismatch[0]) if mismatch.size else n


class PrefixCacheSimulator:
    """Deterministic per-episode prompt-cache model for one pricing profile."""

    def __init__(
        self, profile: PricingProfile, *, max_breakpoints: int = DEFAULT_MAX_BREAKPOINTS
    ) -> None:
        if max_breakpoints < 1:
            raise ValueError("max_breakpoints must be >= 1")
        self.profile = profile
        self.max_breakpoints = max_breakpoints
        self._entries: list[_Entry] = []
        self._last_now: float | None = None

    @property
    def live_entries(self) -> int:
        return len(self._entries)

    def reset(self) -> None:
        """New episode: cold cache."""
        self._entries = []
        self._last_now = None

    def call(
        self,
        prompt_ids: Sequence[int],
        output_tokens: int,
        now: float,
        *,
        breakpoints: Sequence[int] | None = None,
    ) -> CallCost:
        """Price one call made at simulated time ``now`` (seconds)."""
        if output_tokens < 0:
            raise ValueError("output_tokens must be >= 0")
        if not math.isfinite(now):
            raise ValueError("now must be finite")
        if self._last_now is not None and now < self._last_now:
            raise ValueError("simulated time must not go backwards within an episode")
        self._last_now = now

        prompt = _as_array(prompt_ids)
        self._expire(now)
        if self.profile.mode == "automatic":
            if breakpoints is not None:
                raise ValueError("breakpoints are only meaningful in breakpoint mode")
            cached, written = self._automatic(prompt, now)
        else:
            cached, written = self._breakpoint(prompt, now, breakpoints)
        return self._price(int(prompt.shape[0]), cached, written, output_tokens)

    # --- semantics ----------------------------------------------------------

    def _expire(self, now: float) -> None:
        ttl = self.profile.ttl_seconds
        self._entries = [entry for entry in self._entries if now - entry.last_used <= ttl]

    def _automatic(self, prompt: np.ndarray, now: float) -> tuple[int, int]:
        profile = self.profile
        best, best_entry = 0, None
        for entry in self._entries:
            lcp = common_prefix_length(prompt, entry.ids)
            if lcp > best:
                best, best_entry = lcp, entry
        cached = (best // profile.block_size) * profile.block_size
        if cached < profile.min_cacheable_tokens:
            cached = 0
        if cached and best_entry is not None:
            best_entry.last_used = now

        n = int(prompt.shape[0])
        if n < profile.min_cacheable_tokens:
            return cached, 0
        # The new prompt dominates any entry that is its prefix.
        self._entries = [
            entry
            for entry in self._entries
            if not (
                entry.ids.shape[0] <= n
                and common_prefix_length(prompt, entry.ids) == entry.ids.shape[0]
            )
        ]
        self._entries.append(_Entry(prompt.copy(), now))
        return cached, n - cached

    def _breakpoint(
        self, prompt: np.ndarray, now: float, breakpoints: Sequence[int] | None
    ) -> tuple[int, int]:
        profile = self.profile
        n = int(prompt.shape[0])
        points = self._validate_breakpoints(n, breakpoints)

        cached, hit = 0, None
        for entry in self._entries:
            length = int(entry.ids.shape[0])
            if (
                length > cached
                and length <= n
                and length >= profile.min_cacheable_tokens
                and common_prefix_length(prompt, entry.ids) == length
            ):
                cached, hit = length, entry
        if hit is not None:
            hit.last_used = now

        written_to = cached
        for point in points:
            if point < profile.min_cacheable_tokens or point <= cached:
                continue
            ids = prompt[:point]
            existing = next(
                (
                    entry
                    for entry in self._entries
                    if entry.ids.shape[0] == point and common_prefix_length(ids, entry.ids) == point
                ),
                None,
            )
            if existing is None:
                self._entries.append(_Entry(ids.copy(), now))
            else:
                existing.last_used = now
            written_to = max(written_to, point)
        return cached, written_to - cached

    def _validate_breakpoints(self, n: int, breakpoints: Sequence[int] | None) -> list[int]:
        if breakpoints is None:
            return [n] if n else []
        points = sorted({int(point) for point in breakpoints})
        if any(point < 1 or point > n for point in points):
            raise ValueError(f"breakpoints must lie in [1, {n}], got {list(breakpoints)}")
        if len(points) > self.max_breakpoints:
            raise ValueError(
                f"at most {self.max_breakpoints} breakpoints per call, got {len(points)}"
            )
        return points

    # --- pricing ------------------------------------------------------------

    def _price(self, prompt_tokens: int, cached: int, written: int, output: int) -> CallCost:
        profile = self.profile
        uncached = prompt_tokens - cached - written
        input_cost = uncached * profile.input_per_token
        read_cost = cached * profile.cache_read_per_token
        write_cost = written * profile.cache_write_per_token
        output_cost = output * profile.output_per_token
        return CallCost(
            prompt_tokens=prompt_tokens,
            cached_tokens=cached,
            written_tokens=written,
            uncached_tokens=uncached,
            output_tokens=output,
            input_cost=input_cost,
            cache_read_cost=read_cost,
            cache_write_cost=write_cost,
            output_cost=output_cost,
            total_cost=input_cost + read_cost + write_cost + output_cost,
        )


# --- demo -----------------------------------------------------------------------


def scripted_prompts(
    strategy: str, *, calls: int = 5, prefix: int = 100, item: int = 50, window: int = 3
) -> list[list[int]]:
    """Token-ID prompts for the docstring scenario. Item ``k`` uses IDs ``k*1000+j``."""
    protected = list(range(1, prefix + 1))
    prompts: list[list[int]] = []
    for call in range(1, calls + 1):
        items = list(range(1, call + 1))
        if strategy == "window":
            items = items[-window:]
        elif strategy != "never_prune":
            raise ValueError(f"unknown strategy {strategy!r}")
        body = [k * 1000 + j for k in items for j in range(item)]
        prompts.append(protected + body)
    return prompts


def run_scripted(profile: PricingProfile, strategy: str, **kwargs: int) -> list[CallCost]:
    sim = PrefixCacheSimulator(profile)
    return [
        sim.call(prompt, output_tokens=0, now=float(index))
        for index, prompt in enumerate(scripted_prompts(strategy, **kwargs))
    ]


def _demo(profile_name: str) -> int:
    profile = get_profile(profile_name)
    print(f"profile: {profile.name}  mode={profile.mode}  source={profile.source}")
    if profile.illustrative:
        print("NOTE: illustrative profile - numbers are not provider prices.")
    for strategy in ("never_prune", "window"):
        costs = run_scripted(profile, strategy, prefix=1000, item=500)
        print(f"\n{strategy}")
        print(f"{'call':>4} {'prompt':>7} {'cached':>7} {'written':>8} {'uncached':>9} {'$':>12}")
        for index, cost in enumerate(costs, start=1):
            print(
                f"{index:>4} {cost.prompt_tokens:>7} {cost.cached_tokens:>7} "
                f"{cost.written_tokens:>8} {cost.uncached_tokens:>9} {cost.total_cost:>12.6f}"
            )
        total_tokens = sum(cost.prompt_tokens for cost in costs)
        total = sum(cost.total_cost for cost in costs)
        print(f"total tokens={total_tokens}  total ${total:.6f}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prefix-cache cost simulator")
    parser.add_argument("--demo", action="store_true", help="print the scripted comparison")
    parser.add_argument("--profile", default="anthropic-sonnet-5-5")
    args = parser.parse_args(argv)
    if not args.demo:
        parser.print_help()
        return 2
    return _demo(args.profile)


if __name__ == "__main__":
    sys.exit(main())
