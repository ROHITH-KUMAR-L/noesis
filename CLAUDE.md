# CLAUDE.md

Guidance for Claude (and other AI coding agents) working in this repo. Read this first, then
[`prd.md`](prd.md) (what and why) and [`plan.md`](plan.md) (implementation contract: invariants,
interfaces, per-phase tests and exit gates). If this file and those disagree on **scope or
claims**, this file is newer; flag the conflict instead of silently picking one.

Status lives in the README phase table. As of 2026-10-27: Phases 0–1 done, Phase 2 (cache cost
model) next, **no RL training has run and no result exists**.

## Project

**Cache-Aware Reinforcement Learning for Memory Management in Multi-Turn LLM Agents**
(AIML EL, 5th semester; package `memory_rl`).

### Problem

Multi-turn LLM agents resend their whole, growing history of messages and tool outputs on every
call, so cost and latency grow and the extra noise hurts accuracy. Learned pruning methods fix
this by using fewer tokens. Real LLM APIs, however, discount cached prompt prefixes: when a
pruner removes or rewrites an old message, everything after it misses the cache and is billed
at full price again. A method that saves tokens can therefore **raise** the real bill.

Agents need a memory manager that decides what to keep, compress or remove, **and when to
edit**, optimising the actual cost after caching discounts rather than the token count, and
respecting a given cost or context budget.

### Objectives

1. **RL memory controller.** A PPO controller that, before each LLM call, decides whether to
   edit the history; if it edits, it picks KEEP / COMPRESS / PRUNE for each past step. The
   agent runs on a frozen LLM; only the controller learns.
2. **Cache-aware reward.** Task success minus the real API cost after prefix-caching
   discounts, computed by a prefix-cache cost simulator configured with provider pricing.
3. **Budget-constrained control (stretch).** PPO-Lagrangian so the controller meets a given
   per-episode cost budget or per-call token budget, with the policy conditioned on the budget
   so one policy covers the range of budgets.
4. **Benchmark and compare** on ALFWorld (core) and WebShop (stretch) against: full history,
   sliding window, masking old tool outputs, LLMLingua, a hand-written cache-aware compaction
   rule, a CADOC-style analytical rule, and the same controller trained on a token-count
   reward. Main result: **task success vs real cost**.

### Scope priority

- **Core (must finish):** objectives 1, 2 and 4 on **one** benchmark (ALFWorld unless the team
  decides otherwise).
- **Stretch:** objective 3 (budget control, Phase 7) and the second benchmark (WebShop).
- Don't start stretch work while a core phase's exit gate is still failing.

### Novelty (claim exactly this, no stronger)

- **To our knowledge, the first RL-learned memory controller for agents trained on a
  cache-adjusted cost objective.** Prior cache-aware methods (TokenPilot, AKV, CADOC, CAPC) use
  fixed rules or formulas; prior learned pruners (TRACER, LRE) reward token count.
- **Learning *when* to edit, not just what.** An explicit edit-now-or-not gate. Hypothesis: it
  learns rare, large compactions that keep the cached prefix stable, while token-based pruners
  trim every turn and break the cache each time. This is tested directly (prd.md H3). CADOC
  already decides when to batch edits analytically, so it is the closest competitor here.
- **Budget-constrained, budget-conditioned control (stretch).** A single policy meets a given
  budget as a constraint at runtime, rather than tuning a threshold or penalty per budget.
- **Evaluation in real cost.** Show empirically when "fewer tokens" and "lower bill" disagree,
  which token-only evaluations cannot reveal.

Never write "the first" without "to our knowledge". Re-check arXiv for new prior art before the
report (Phase 9).

## Key prior art

| Work | ID | Relevance |
|---|---|---|
| CADOC | arXiv 2609.37012 | Closest. Cache-rebuild cost in the objective, but an analytical (EOQ-style) rule, not learned. Baseline. |
| CAPC | arXiv 2607.15516 | Cache-aware, analytical. Cite. |
| TRACER | arXiv 2608.29363 | RL over a frozen agent, raw-token penalty. Token-reward PPO variant is our stand-in. |
| LRE | arXiv 2606.20954 | Learned keep/evict scorer, token objective; budget via threshold. |
| Observation masking | arXiv 2508.21433 | Simple and strong. **Must-beat baseline** (B3). |
| Beyond Token Savings | arXiv 2609.32961 | Token savings vs real cost; motivation. |
| TokenPilot | arXiv 2606.17016 | Cache-aware hand-written rules. |
| AKV | arXiv 2610.28577 | Cache-aware hand-written rules. |
| PCRL | IEEE Access, 2024 | IEEE-indexed citation for RL-based prompt compression. |

Full literature notes, critique and the final proposal:
<https://claude.ai/artifact/CM2osupFWhbX6tQ6Majt5s>. Verify any bibliographic detail
(authors, venue, DOI) against the source before it goes in a report; don't fill gaps from memory.

## Working conventions

- **Frozen LLM.** Never fine-tune the agent LLM. Only the controller has trainable weights.
- **PPO controller** with a gate head + masked per-item KEEP/COMPRESS/PRUNE head + value head
  (plan.md §3, prd.md FR-6). Invalid actions get probability exactly 0.
- **All cost goes through the prefix-cache simulator** (`memory_rl/cost/`, Phase 2) with prices
  from versioned pricing-profile configs (source + date). No hardcoded prices, no ad-hoc token
  counting outside it. Use the frozen LLM's own tokenizer for accounting.
- **Seeds:** report every result over **≥ 3 seeds** (5 if compute allows) with mean and spread
  or CIs. A single-seed number is a debug run, not a result.
- **Splits:** tune knobs and select checkpoints on validation only; touch the test split once.
- **Never invent results, numbers, plots or citations.** Illustrative values must be labelled
  as such. If something wasn't run or verified, say so. A negative result is a valid outcome.
- Follow the ground rules in README and the invariants in plan.md §4.2 (no silent truncation,
  no simulator-only labels in observations, no paid-API spend without explicit approval, no
  secrets or benchmark data in git).

## Commands

```bash
uv pip install -e ".[dev]"
python -m memory_rl.smoke && ruff check . && mypy && python -m pytest
```

Run all four before committing; CI runs the same. Never skip or delete a test to get green.
