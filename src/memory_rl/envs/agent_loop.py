"""The agent loop (plan.md §4.1, §6.5).

``AgentEpisode`` is stepwise so both a ``MemoryPolicy`` (baselines, evaluation)
and the PPO trainer can drive it::

    episode = AgentEpisode(env, task_id, seed=0, llm=..., meter=..., tokenizer=...)
    state = episode.begin()
    while state is not None:
        state, turn = episode.advance(policy.decide(state))
    record = episode.record(method="full_history")

Per turn: controller decision → memory edits → serialize → context check →
frozen LLM → cost meter → env step. Cost is summed over every call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from memory_rl.cost import (
    CacheAwareCostMeter,
    CallCost,
    CostMeter,
    EpisodeCostAccumulator,
    PricingProfile,
    TokenCostMeter,
    get_profile,
)
from memory_rl.envs.agent_prompt import AgentPrompt, AgentPromptSerializer
from memory_rl.envs.base import AgentEnv, TerminalReason
from memory_rl.envs.policy_api import (
    AlwaysKeepPolicy,
    ControllerState,
    Decision,
    ItemView,
    MemoryPolicy,
)
from memory_rl.evaluation.records import config_hash, git_commit, validate_record
from memory_rl.llm.backend import LLMBackend, LLMResponse
from memory_rl.memory import (
    Compressor,
    EncodingTokenizer,
    MemoryItem,
    MemoryStatus,
    MemoryStore,
    MemoryType,
    OperationKind,
    TiktokenTokenizer,
    valid_actions,
)
from memory_rl.memory.compressors import LineTruncationCompressor


class InvalidDecisionError(ValueError):
    """A policy chose an action outside the mask (never silently repaired)."""


@dataclass(frozen=True, slots=True)
class LoopConfig:
    context_limit_tokens: int = 8192
    max_output_tokens: int = 256
    env_step_seconds: float = 0.2
    #: breakpoint-mode profiles: also cache the protected prefix (robust agent-loop layout)
    prefix_breakpoint: bool = True
    #: newest N items are never candidates (the agent has not acted on them yet)
    protect_latest: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TurnTrace:
    """Full per-turn detail for inspection and the demo UI (toy data only)."""

    turn: int
    edit: bool
    actions: dict[str, str]
    items: list[dict[str, Any]]
    prompt_tokens: int
    cost: dict[str, float | int] | None
    token_cost: float | None
    llm_text: str
    observation: str
    reward: float
    terminal_reason: str | None = None
    time_s: float = 0.0


@dataclass(slots=True)
class _Counters:
    edits: int = 0
    forced_edits: int = 0
    keep: int = 0
    compress: int = 0
    prune: int = 0
    tokens_removed: int = 0
    edit_turns: list[int] = field(default_factory=list)


class AgentEpisode:
    def __init__(
        self,
        env: AgentEnv,
        task_id: str,
        *,
        seed: int,
        llm: LLMBackend,
        meter: CostMeter,
        tokenizer: EncodingTokenizer,
        profile: PricingProfile | None = None,
        compressor: Compressor | None = None,
        config: LoopConfig | None = None,
        split: str = "valid",
    ) -> None:
        self.env = env
        self.task_id = task_id
        self.seed = seed
        self.llm = llm
        self.meter = meter
        self.tokenizer = tokenizer
        self.profile = profile
        self.compressor = compressor or LineTruncationCompressor()
        self.config = config or LoopConfig()
        self.split = split
        self.serializer = AgentPromptSerializer(tokenizer, env.system_prompt())
        # The token meter is always run alongside so both costs are recorded.
        self._token_meter = TokenCostMeter()
        self.costs = EpisodeCostAccumulator()
        self.token_costs: list[float] = []
        self.turns: list[TurnTrace] = []
        self.store = MemoryStore(tokenizer=tokenizer)
        self.counters = _Counters()
        self.terminal_reason: TerminalReason | None = None
        self.success = False
        self.score = 0.0
        self.now = 0.0
        self.latency_s = 0.0
        self.provider_usage: list[dict[str, Any]] = []
        self._instruction = ""
        self._few_shot = env.few_shot()
        self._turn = 0
        self._last_cached_fraction = 0.0
        self._next_id = 0
        self._started = False

    # --- public API ---------------------------------------------------------

    @property
    def done(self) -> bool:
        return self.terminal_reason is not None

    @property
    def instruction(self) -> str:
        return self._instruction

    def begin(self) -> ControllerState:
        if self._started:
            raise RuntimeError("episode already started")
        self._started = True
        obs = self.env.reset(self.task_id, self.seed)
        if not obs.instruction:
            raise RuntimeError("env.reset must return the task instruction")
        self._instruction = obs.instruction
        self.meter.reset()
        if obs.text:
            self._add_item(obs.text, MemoryType.TOOL_RESULT)
        return self.state()

    def current_prompt(self) -> AgentPrompt:
        return self.serializer.serialize(
            self._instruction, self._few_shot, self.store.active_items()
        )

    def state(self) -> ControllerState:
        prompt = self.current_prompt()
        active = self.store.active_items()
        protected = {item.memory_id for item in active[len(active) - self.config.protect_latest :]}
        views = []
        for item in active:
            start, end = prompt.item_spans[item.memory_id]
            mask: tuple[OperationKind, ...] = ()
            gain = 0
            if item.memory_id not in protected:
                mask = self.item_mask(item)
                if OperationKind.COMPRESS in mask:
                    compressed = self.compressor.compress(item.text) or ""
                    gain = item.token_count - self.tokenizer.count(compressed)
            views.append(
                ItemView(
                    memory_id=item.memory_id,
                    memory_type=item.memory_type,
                    status=item.status,
                    text=item.text,
                    token_count=item.token_count,
                    turn_index=item.turn_index,
                    compression_count=item.compression_count,
                    span_start=start,
                    span_end=end,
                    valid_actions=mask,
                    compression_gain=gain,
                )
            )
        profile = self.profile
        return ControllerState(
            turn=self._turn,
            max_turns=self.env.max_steps,
            instruction=self._instruction,
            items=tuple(views),
            prompt_tokens=prompt.total_prompt_tokens,
            prefix_tokens=prompt.prefix_tokens,
            context_limit=self.config.context_limit_tokens,
            last_cached_fraction=self._last_cached_fraction,
            cache_read_multiplier=profile.cache_read_multiplier if profile else 1.0,
            cache_write_multiplier=profile.cache_write_multiplier if profile else 1.0,
            cache_mode=profile.mode if profile else "none",
        )

    def item_mask(self, item: MemoryItem) -> tuple[OperationKind, ...]:
        """Engine mask plus the compress-at-most-once rule (plan.md §6.3)."""
        mask = valid_actions(item, self.compressor, self.tokenizer)
        if item.compression_count >= 1:
            mask = tuple(op for op in mask if op is not OperationKind.COMPRESS)
        return mask

    def advance(self, decision: Decision) -> tuple[ControllerState | None, TurnTrace]:
        if not self._started or self.done:
            raise RuntimeError("advance() needs a started, unfinished episode")
        state = self.state()
        self._apply(decision, state)

        prompt = self.current_prompt()
        items_snapshot = self._snapshot()
        if prompt.total_prompt_tokens > self.config.context_limit_tokens:
            return None, self._finish_turn(
                decision, items_snapshot, prompt, None, None, "", "", 0.0,
                TerminalReason.CONTEXT_OVERFLOW,
            )  # fmt: skip

        response = self.llm.generate(prompt.messages, max_tokens=self.config.max_output_tokens)
        output_tokens = self.tokenizer.count(response.text)
        cost = self.meter.call(
            prompt.token_ids, output_tokens, self.now, breakpoints=self._breakpoints(prompt)
        )
        token_cost = self._token_meter.call(prompt.token_ids, output_tokens, self.now).total_cost
        self._account(cost, token_cost, response)

        self._add_item(response.text, MemoryType.ASSISTANT)
        obs, reward, done, info = self.env.step(response.text)
        self.now += self.config.env_step_seconds
        self.score += reward
        terminal: TerminalReason | None = None
        if done:
            if info.get("success"):
                self.success = True
                terminal = TerminalReason.SUCCESS
            elif info.get("step_limit"):
                terminal = TerminalReason.STEP_LIMIT
            else:
                terminal = TerminalReason.ENV_DONE
        else:
            self._add_item(obs.text, MemoryType.TOOL_RESULT)
        trace = self._finish_turn(
            decision, items_snapshot, prompt, cost, token_cost, response.text, obs.text,
            reward, terminal,
        )  # fmt: skip
        return (None if self.done else self.state()), trace

    def run(self, policy: MemoryPolicy) -> AgentEpisode:
        policy.reset(self.seed)
        state: ControllerState | None = self.begin()
        while state is not None:
            state, _ = self.advance(policy.decide(state))
        return self

    def record(
        self,
        *,
        method: str,
        variant: str = "-",
        knob: str = "-",
        extra_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.done:
            raise RuntimeError("episode is not finished")
        summary = self.costs.summary()
        counters = self.counters
        config = {
            "loop": self.config.to_dict(),
            "profile": self.profile.to_dict() if self.profile else None,
            "compressor": self.compressor.name,
            "meter": self.meter.variant,
            **(extra_config or {}),
        }
        record: dict[str, Any] = {
            "method": method,
            "variant": variant,
            "knob": knob,
            "env": self.env.name,
            "split": self.split,
            "task_id": self.task_id,
            "seed": self.seed,
            "llm": self.llm.name,
            "tokenizer": self.tokenizer.name,
            "pricing_profile": self.profile.name if self.profile else "none",
            "success": self.success,
            "score": self.score,
            "turns": len(self.turns),
            "llm_calls": summary["llm_calls"],
            "terminal_reason": str(self.terminal_reason),
            "prompt_tokens_total": summary["prompt_tokens_total"],
            "cached_tokens_total": summary["cached_tokens_total"],
            "written_tokens_total": summary["written_tokens_total"],
            "uncached_tokens_total": summary["uncached_tokens_total"],
            "output_tokens_total": summary["output_tokens_total"],
            "input_cost": summary["input_cost"],
            "cache_read_cost": summary["cache_read_cost"],
            "cache_write_cost": summary["cache_write_cost"],
            "output_cost": summary["output_cost"],
            "total_cost": summary["total_cost"],
            "token_cost": float(sum(self.token_costs)),
            "edits": counters.edits,
            "forced_edits": counters.forced_edits,
            "actions_keep": counters.keep,
            "actions_compress": counters.compress,
            "actions_prune": counters.prune,
            "tokens_removed": counters.tokens_removed,
            "edit_turns": list(counters.edit_turns),
            "latency_s": self.latency_s,
            "budget": None,
            "budget_compliant": None,
            "git_commit": git_commit(),
            "config_hash": config_hash(config),
            "cost_unit": "usd" if self.meter.variant == "cache" else "tokens",
            "provider_usage_available": bool(self.provider_usage),
        }
        validate_record(record)
        return record

    # --- internals ----------------------------------------------------------

    def _breakpoints(self, prompt: AgentPrompt) -> list[int] | None:
        if self.profile is None or self.profile.mode != "breakpoint":
            return None
        points = {prompt.total_prompt_tokens}
        if self.config.prefix_breakpoint and 0 < prompt.prefix_tokens:
            points.add(prompt.prefix_tokens)
        return sorted(points)

    def _apply(self, decision: Decision, state: ControllerState) -> None:
        if not decision.edit:
            if any(op is not OperationKind.KEEP for op in decision.actions.values()):
                raise InvalidDecisionError("NO_EDIT decision must not carry non-KEEP actions")
            return
        candidates = {item.memory_id: item for item in state.candidates}
        for memory_id, op in decision.actions.items():
            view = candidates.get(memory_id)
            if view is None:
                raise InvalidDecisionError(f"{memory_id!r} is not a candidate this turn")
            if op not in view.valid_actions:
                raise InvalidDecisionError(f"{op} is masked for {memory_id!r}")
        counters = self.counters
        counters.edits += 1
        counters.edit_turns.append(self._turn)
        for memory_id in candidates:
            op = decision.actions.get(memory_id, OperationKind.KEEP)
            result = self.store.apply(op, memory_id, compressor=self.compressor)
            if not result.success:
                raise InvalidDecisionError(f"{op} failed for {memory_id!r}: {result.reason}")
            if op is OperationKind.KEEP:
                counters.keep += 1
            elif op is OperationKind.COMPRESS:
                counters.compress += 1
            else:
                counters.prune += 1
            counters.tokens_removed += result.active_token_savings

    def _account(self, cost: CallCost, token_cost: float, response: LLMResponse) -> None:
        self.costs.add(cost)
        self.token_costs.append(token_cost)
        self._last_cached_fraction = cost.cached_fraction
        self.now += response.latency_s
        self.latency_s += response.latency_s
        if response.usage:
            self.provider_usage.append(dict(response.usage))

    def _add_item(self, text: str, memory_type: MemoryType) -> None:
        self._next_id += 1
        self.store.add(
            MemoryItem.create(
                memory_id=f"m{self._next_id:03d}",
                text=text,
                tokenizer=self.tokenizer,
                turn_index=self._turn,
                memory_type=memory_type,
            )
        )

    def _snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                "id": item.memory_id,
                "type": item.memory_type.value,
                "status": item.status.value,
                "tokens": item.token_count,
                "turn": item.turn_index,
                "text": item.text,
            }
            for item in self.store.archived_items()
        ]

    def _finish_turn(
        self,
        decision: Decision,
        items: list[dict[str, Any]],
        prompt: AgentPrompt,
        cost: CallCost | None,
        token_cost: float | None,
        llm_text: str,
        observation: str,
        reward: float,
        terminal: TerminalReason | None,
    ) -> TurnTrace:
        trace = TurnTrace(
            turn=self._turn,
            edit=decision.edit,
            actions={k: v.value for k, v in decision.actions.items() if v != OperationKind.KEEP},
            items=items,
            prompt_tokens=prompt.total_prompt_tokens,
            cost=cost.to_dict() if cost else None,
            token_cost=token_cost,
            llm_text=llm_text,
            observation=observation,
            reward=reward,
            terminal_reason=str(terminal) if terminal else None,
            time_s=self.now,
        )
        self.turns.append(trace)
        self._turn += 1
        if terminal is not None:
            self.terminal_reason = terminal
        return trace


def pruned_text_absent(prompt: AgentPrompt, store: MemoryStore) -> bool:
    """Invariant check used by tests: no pruned item's text is in any message."""
    pruned = [item for item in store.archived_items() if item.status is MemoryStatus.PRUNED]
    return all(item.memory_id not in prompt.included_ids for item in pruned)


def prompt_digest(prompt: AgentPrompt) -> str:
    return hashlib.sha256(json.dumps(prompt.token_ids).encode()).hexdigest()[:16]


# --- CLI ------------------------------------------------------------------------


def build_meter(variant: str, profile: PricingProfile) -> CostMeter:
    if variant == "cache":
        return CacheAwareCostMeter(profile)
    if variant == "token":
        return TokenCostMeter()
    raise ValueError(f"unknown cost variant {variant!r}")


def build_llm(kind: str, *, allow_paid: bool) -> LLMBackend:
    if kind == "mock":
        from memory_rl.llm.mock import ScriptedKeyHuntLLM

        return ScriptedKeyHuntLLM()
    if kind == "openai":
        if not allow_paid:
            raise SystemExit("--llm openai may spend money; re-run with --allow-paid")
        from memory_rl.llm.backend import LLMConfig, OpenAICompatibleBackend

        return OpenAICompatibleBackend(LLMConfig.from_env())
    raise SystemExit(f"unknown --llm {kind!r}")


def build_env(name: str) -> AgentEnv:
    if name == "toy":
        from memory_rl.envs.toy import KeyHuntEnv

        return KeyHuntEnv()
    raise SystemExit(f"unknown --env {name!r}; only 'toy' is implemented (ALFWorld: see README)")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run agent episodes with a memory policy")
    parser.add_argument("--env", default="toy")
    parser.add_argument("--llm", default="mock", choices=["mock", "openai"])
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--profile", default="anthropic-sonnet-5-5")
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="one short episode")
    args = parser.parse_args(argv)

    env = build_env(args.env)
    llm = build_llm(args.llm, allow_paid=args.allow_paid)
    profile = get_profile(args.profile)
    tokenizer = TiktokenTokenizer("o200k_base")
    episodes = 1 if args.smoke else args.episodes
    for task_id in list(env.task_ids(args.split))[:episodes]:
        episode = AgentEpisode(
            env, task_id, seed=args.seed, llm=llm, meter=CacheAwareCostMeter(profile),
            tokenizer=tokenizer, profile=profile, split=args.split,
        )  # fmt: skip
        record = episode.run(AlwaysKeepPolicy()).record(method="full_history")
        print(
            json.dumps(
                {k: record[k] for k in ("task_id", "success", "turns", "total_cost",
                                        "prompt_tokens_total", "cached_tokens_total",
                                        "terminal_reason")}  # fmt: skip
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
