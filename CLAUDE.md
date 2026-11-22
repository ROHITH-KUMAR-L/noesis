# CLAUDE.md

Guidance for Claude (and other AI coding agents) working in this repo. Read this first, then
[`prd.md`](prd.md) (what and why) and [`plan.md`](plan.md) (implementation contract: invariants,
interfaces, per-phase tests and exit gates). If this file and those disagree on **scope or
claims**, this file is newer; flag the conflict instead of silently picking one.

Status lives in the README phase table; details and next steps in [`state.md`](state.md). As of
2026-11-22: Phases 0–2 and 4 pass their exit gates. Phases 3, 5 and 6 pass only their toy-env
parts; the ALFWorld parts (competence gate, baseline runs, PPO subset run) are pending. Phases
7–9 not started. PPO has only been trained on the toy env with a mock LLM (validation split).
**No test-split, ALFWorld or real-LLM result exists.**

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

Narrowed after the 2026-11-22 literature check ([`docs/novelty_check.md`](docs/novelty_check.md),
[`docs/gaps_and_novelty.md`](docs/gaps_and_novelty.md)).

- **To our knowledge, the first memory controller for LLM agents trained with RL on a
  cache-adjusted cost objective** (task success minus billed input cost after prefix-cache
  discounts), as a lightweight policy over a **frozen** LLM. Prior cache-aware context managers
  (CADOC, CAPC, TokenPilot, Self-GC, AKV) use analytical rules, schedules or LLM planners; prior
  learned ones (TRACER, LRE, MemAct, AutoCompact, ContextBudget) optimise success with token,
  context-length or storage penalties, and most fine-tune the agent.
- **A learned, cache-priced edit gate.** Learned edit timing exists (AutoCompact, ContextBudget)
  but is trained without cache cost; cache-aware timing exists (CADOC, Self-GC) but is not
  learned. The combination is the claim. "Rare, large compactions that keep the cached prefix
  stable" is hypothesis H3 until the no-gate ablation and edit-timing comparison vs CADOC show it.
- **Budget control (stretch), no "first".** Budget-conditioned and Lagrangian control already
  exist (ContextBudget, 2610.28409, LACONIC). Ours differs only in that the budget is billed cost
  after caching, for a controller over a frozen LLM.
- **Evaluation in real cost is a replication, not a novelty.** 2611.22238 and CAPC already show
  "fewer tokens" and "lower bill" can disagree. We test whether this holds for learned
  controllers (token-reward vs cache-reward PPO), reporting both token and real cost, including
  for observation masking (B3).

Never write "the first" without "to our knowledge". Every claim carries the caveat "toy env,
mock agent" until there is a real-LLM ALFWorld result. Re-check arXiv before the report (Phase 9).

### Planned contributions (decided 2026-11-22; none run yet)

Headline: **a price-adaptive, cache-aware learned memory controller.** Fixed rules must be
re-tuned per provider (Self-GC's own threshold "is not a universal constant"), and 2611.22238
finds that pruning flips from saving to costing money as the cache-read price ratio moves. One
learned policy that adapts *when* and *how much* to edit to the provider's cache pricing is the
gap no prior work fills.

1. **N1 Price-adaptive policy (headline, core).** `cache_read_multiplier` is already a global
   feature. Train one PPO policy on randomised pricing (read multiplier ~0.05–1.0, write
   premium, both cache modes) and test on held-out real profiles. Supported if its edit rate
   and timing shift with price and it matches or beats per-profile tuned rules (CADOC,
   threshold, break-even) across the range. Refuted if behaviour doesn't change with price or
   per-profile policies win clearly. Code: `cost/pricing.py`, `controller/train.py`,
   `controller/features.py`, `configs/pricing/`.
2. **N2 Gate analysis vs CADOC (core, supports claim 2 / H3).** No-gate ablation, edit-timing
   histograms, per-turn agreement with CADOC's Eq. 14 decision. Code: `controller/env.py`,
   `baselines/rules.py`, `evaluation/records.py`.
3. **N3 Break-even commit baseline (core).** A Self-GC-style rule (edit only when the saving
   beats the cache-rebuild cost) next to threshold compaction and CADOC, so the learned gate is
   compared with the strongest cache-aware rules. Code: `baselines/rules.py`.
4. **N4 Cache-model robustness (cheap analysis).** Do method rankings hold across TTL, write
   premium and breakpoint vs automatic mode? Kendall tau over the same records. Code:
   `cost/cache_sim.py`, `evaluation/run_eval.py`.

Deferred: dollar-budget PPO-Lagrangian (Phase 7) and WebShop, after N1–N4 and ALFWorld.
Dropped: agent fine-tuning, KV-level editing, token-level compressors, "first
budget-conditioned" or "first to learn when to compact" claims.

## Key prior art

| Work | ID | Relevance |
|---|---|---|
| CADOC | arXiv 2609.37012 | Closest. Cache-rebuild cost in the objective, but an analytical (EOQ-style) rule, not learned. Baseline. |
| Context Has a Price | arXiv 2611.22238 | Measurement study: removing old observations raises cost 38–116% at 1/10 cache-read price. Primary motivation citation; pre-empts "real cost" as a finding. |
| Self-GC | arXiv 2607.00692 | LLM planner picks fold/mask/prune, commits only if savings beat the cache break. Cache-aware timing, not learned. Candidate baseline. |
| CAPC | arXiv 2607.15516 | Cache-aware, analytical two-tier cost model. Cite. |
| AutoCompact | arXiv 2610.02163 | Learns when to compact (SFT then RL) but trains the agent; no cache term. |
| ContextBudget | arXiv 2604.01664 | RL learns when/how much to compress under varying budgets; no cache term, no Lagrangian. |
| TRACER | arXiv 2608.29363 | REINFORCE over per-tool retention ratios; objective success + tokens + re-invocations. Token-reward PPO is our stand-in. |
| LRE | arXiv 2606.20954 | Learned keep/evict scorer. "Token objective, budget via threshold" not in abstract (unverified). |
| Observation masking | arXiv 2508.21433 | Simple and strong. **Must-beat baseline** (B3). |
| TokenPilot | arXiv 2606.17016 | Cache-aware; abstract doesn't say rule-based or learned. |
| AKV | arXiv 2610.28577 | "Cache Engineering for Agents": edits the KV cache directly, with LLM-driven self-improvement of cache policies. Not prompt-level, not hand-written rules. |
| Beyond Token Savings | arXiv 2609.32961 | Abstract does not cover caching; don't cite it for the cache-cost argument. |
| Budgeted agents | arXiv 2610.28409 | One policy across token/call/step budgets; fine-tunes the agent. Threatens a "first budget-conditioned" claim. |
| PCRL | IEEE Access 12:72578–87, 2024, DOI 10.1109/ACCESS.2024.3403426 | IEEE citation for RL-based prompt compression. |
| Cost-aware memory RL | IEEE RO-MAN 2026, DOI 10.1109/RO-MAN72286.2026.11714580 | Closest IEEE paper: REINFORCE memory-write policy with storage penalty, no cache cost. |

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
