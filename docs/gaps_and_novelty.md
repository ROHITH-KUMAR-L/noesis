# Gaps and novelty opportunities (checked 2026-11-22)

Builds on `novelty_report.md` (same folder). Sources opened for this section: full HTML of
Context Has a Price (arXiv 2611.22238, first 100k chars), Self-GC (2607.00692v1), Cache Engineering
for Agents / AKV (2610.28577), CADOC (2609.37012), AutoCompact (2610.02163), ContextBudget (2604.01664v1),
CompactionRL (2607.05378v1); abstract only for Memex(RL) (2603.04257).
"Stated" = the authors say it. "Inference" = my reading of what they did not do.

Repo grounding (read-only): `state.md` says Phases 0-6 are done on the toy KeyHunt env with a mock LLM;
no ALFWorld and no real-LLM run yet; 6 GB GPU cannot host the planned 7B model. On validation (toy, 3 seeds),
cache-PPO edits about 2-4 times per episode at about $0.0175, while token-PPO edits 8.5 times at about $0.019-0.023. These are validation numbers, not
results. The simulator (`cost/cache_sim.py`) already models automatic-LCP and breakpoint modes,
write multiplier and TTL. `controller/features.py` has `suffix_share` per item and empty
`budget_norm` / `budget_remaining` global slots.

## 1. What the closest papers leave open

| Paper | Relevant limitation / future work | Stated or inference |
|---|---|---|
| Context Has a Price (2611.22238) | Cache accounting "omits provider-specific expiration, write charges, breakpoints, and load-dependent eviction". Prices applied to token streams, not invoices. Interaction between stopping and context removal called "another important question". Its break-even rule is "an accounting baseline ... not a universally beneficial policy". Cost crossover near cache-price ratio c ≈ 0.4-0.5. No learned policy as a contribution. | Stated (quotes via fetch) |
| Self-GC (2607.00692) | Commit rule CommitBenefit ≈ N_future(C−C′) − L_cache_break − L_GC; threshold 0.3 is "an operating policy rather than a universal constant"; L_cache_break not defined; no billed-cost audit. Future work: "replacing some planner calls with lighter task-conditioned policies". Nothing trained. | Stated |
| CADOC (2609.37012) | Optimality only under stationary assumptions, equal object gains, constant Q_h and w_r. No write premium, no TTL. One provider route. No learned component. Small success drops (Terminal-Bench 38→36/89, SWE-bench 31→29/50). | Assumptions stated; missing TTL/write premium is inference (not mentioned) |
| AKV / Cache Engineering (2610.28577) | Policy self-improvement via an LLM (MiniMax-M2.7) on one model, one benchmark; mechanism not specified. Future work: "training agents under this scheme". Requires KV-level control in their engine (inference: not available on closed APIs). | Stated + inference |
| AutoCompact (2610.02163) | Trains the agent (SFT+RL), reward task success; cached tokens priced at 20% only in cost accounting; single scaffold; RL at 32K vs 256K eval. Compaction rate rises after RL (44.3% → 58.5% of tasks), no cache-break analysis. | Stated; "no cache term in the reward" is inference from the text |
| ContextBudget (2604.01664) | Trains the agent with multi-turn GRPO; budget unit = tokens; remaining budget in state; enforced by reward gating (zero reward if any turn overflows), no Lagrangian; no caching; "coarse segment level"; sparse delayed reward. | Stated |
| CompactionRL (2607.05378) | Trains the agent with PPO; no caching or cost analysis; code benchmarks only; credit assignment is approximate. | Stated |
| Memex(RL) (2603.04257) | Trains agent; abstract mentions context budget, not cache or money. | Abstract only |

## 2. Gap list

| # | Gap | Evidence | How this project fills it | Effort | Risk |
|---|---|---|---|---|---|
| G1 | No learned context controller has cache-adjusted billed cost in its training signal. Cache-aware methods are analytic (CADOC, break-even rule) or LLM-planner + threshold (Self-GC). | Inference across all papers in novelty_report.md | Core of the project: `controller/reward.py` cache variant through `cost/cache_sim.py`. | S (built) | Medium: toy-only evidence until a real LLM runs |
| G2 | Fixed thresholds/rules do not adapt to price ratio; the sign of "remove context" flips around c ≈ 0.4-0.5. | Stated crossover (2611.22238); Self-GC threshold "not a universal constant" (stated) | Price-conditioned controller: add pricing features (read mult, write mult, mode) to global features, train over randomised profiles, test on held-out profiles. | M | Low-medium: may just learn "never edit when c small" (still informative) |
| G3 | Cost models omit write premiums, TTL expiry and breakpoint semantics. | Stated (2611.22238); not in CADOC (inference) | Simulator already has them. Evaluate whether method rankings change across automatic vs breakpoint mode, write mult 1.0 vs 1.25, TTL with simulated tool latency. | S | Low; effect might be small, which is still reportable |
| G4 | Analytic rule assumptions (stationary, equal object gains) vs heterogeneous agent histories. | CADOC assumptions stated; the failure case is inference | Toy env variant with high-variance observation lengths; compare PPO vs CADOC rule vs break-even rule there. | S-M | Medium: CADOC may match PPO (valid negative result) |
| G5 | Shorter prompts can lengthen episodes; interaction of stopping and removal is open. | Stated (2611.22238) | RL return already sums cost over all calls, so episode-length effects are in the objective. Report steps/episode and cost decomposition per method. | S | Low |
| G6 | Lightweight policy instead of an LLM planner for commit decisions. | Self-GC future work (stated) | The PPO MLP controller is a lightweight policy over a frozen LLM, trainable on CPU (~15 min sweep per state.md). | S (built) | Low; but we do not implement fold/recovery, so claim only the commit/gate part |
| G7 | Learned context managers all fine-tune the agent LLM (AutoCompact, ContextBudget, CompactionRL, MemAct, MEMENTO, Memex). None works as an add-on to a closed/frozen API model. | Inference | External controller, frozen LLM. Practical angle, not a primary novelty. | S (built) | Low |
| G8 | Budget in billed $ after caching, with a constraint method (Lagrangian) rather than reward gating/penalties. | ContextBudget uses gating on tokens (stated); 2610.28409 uses penalty rewards on tokens/calls (abstract) | Phase 7: PPO-Lagrangian on per-episode cache-adjusted cost; `budget_norm`/`budget_remaining` slots exist. | M-L | Medium-high: stretch, schedule risk |
| G9 | Real-environment, real-LLM evidence. All our numbers are toy + mock. | Project state | ALFWorld needs a real model. Options: smallest instruct model that fits 6 GB with vLLM prefix caching (feasibility unverified), or an approved small paid-API budget. | L | High: biggest threat to every claim |

## 3. Candidate novelty angles (ranked by defensibility × feasibility)

### A1. Cache-priced learned gate vs analytic cache-aware rules (core; mostly built)
- Claim: "To our knowledge, this is the first context controller for LLM agents trained by reinforcement learning on cache-adjusted billed cost; we compare it head-to-head with analytic cache-aware rules (CADOC-style, break-even) and a token-reward RL controller."
- Experiment: Phase 8 test split, ≥3 seeds, paired bootstrap CIs at matched success; add a break-even baseline (remove only if suffix re-bill < expected savings, as in 2611.22238) next to `cadoc_rule`; add a heterogeneous-observation-length toy variant (G4).
- Refuted if: CADOC or break-even rule is within CI of cache-PPO on cost at equal success (then report "learning adds no value over analytic rules here").
- Builds on: `controller/reward.py`, `controller/policy.py`, `controller/train.py`, `baselines/rules.py` (`cadoc_rule`, `threshold_compaction`), `cost/cache_sim.py`, `evaluation/metrics.py` (`paired_difference_ci`), `envs/toy.py`.

### A2. Price-conditioned controller (one policy across providers / cache-price ratios)
- Claim: "To our knowledge, the first context controller conditioned on the provider's cache pricing, so one policy changes its edit behaviour with the cache-read price instead of being retuned for each profile."
- Experiment: add pricing features to global features (bump `SCHEMA_VERSION`); train with randomised profiles (read multiplier spanning ~0.05-1, write multiplier 1.0/1.25, both modes); evaluate on held-out profiles vs (a) per-profile trained PPO, (b) Phase-9 fixed transfer, (c) CADOC with matched w_r. Plot edit rate vs c and locate any crossover; compare with the 0.4-0.5 crossover reported by 2611.22238 (different setting, so compare qualitatively).
- Refuted if: edit rate does not vary with c, or the conditioned policy is worse than per-profile policies by more than CI.
- Builds on: `controller/features.py` (global slots), `cost/pricing.py` (profiles + loader), `configs/pricing/*.yaml`, `cost/cache_sim.py`, `controller/train.py`.

### A3. Learned gate reproduces batching without being told (H3 mechanism)
- Claim: "The cache-reward controller learns rare, batched edits whose timing agrees with the CADOC commit rule, while the token-reward controller edits every turn; removing the gate removes the cost advantage."
- Experiment: per-turn agreement (e.g. Cohen's kappa) between PPO gate and CADOC Eq. 14 decision on the same states; edits/episode and tokens-removed/edit histograms; no-gate ablation (flat per-item actions).
- Refuted if: no-gate controller matches gated cost, or the cache-PPO edit pattern is not batched.
- Builds on: `controller/env.py` (gate), `controller/policy.py` (gate head), `baselines/rules.py` (CADOC ledger), `evaluation/records.py` (JSONL per-turn records).
- Note: this is evidence for claim (2) as a combination, not a standalone "first".

### A4. Cache-model fidelity changes method rankings (cheap analysis)
- Claim: "Write premiums, TTL expiry and breakpoint vs automatic caching, which recent cost studies leave out, change [or do not change] the ranking of context-management methods."
- Experiment: evaluate all methods under automatic vs breakpoint mode, write multiplier on/off, TTL with simulated per-step latency; report Kendall tau between rankings.
- Refuted if: rankings are identical (report as robustness of prior findings).
- Builds on: `cost/cache_sim.py` (modes, TTL), `cost/pricing.py`, `evaluation/run_eval.py`. Needs timestamps per call in `envs/agent_loop.py` if not present (unverified).
- Lower novelty (analysis, not method), but stated by 2611.22238 as missing.

### A5. Cost-budget constraint in billed dollars (stretch, Phase 7)
- Claim: "A PPO-Lagrangian controller over a frozen LLM, conditioned on a per-episode budget in cache-adjusted cost, meets the budget in ≥95% of episodes; budget-aware context RL so far uses token budgets with reward gating or penalties (ContextBudget, Lamb et al.)."
- Experiment: budget sweep; compliance rate; vs per-budget policies and vs a penalty-reward variant.
- Refuted if: compliance < target or no advantage over the penalty variant.
- Builds on: `controller/features.py` (`budget_norm`, `budget_remaining`), `controller/ppo.py`, planned `controller/lagrangian.py`.
- Lowest defensibility after the literature check; do it only once A1 passes.

Rank: A1 > A2 > A3 > A4 > A5. A1+A3 are core and nearly built. A2 adds novelty at moderate cost and uses
existing slots and profiles. A4 is a cheap add-on.

## 4. Not worth pursuing

- **Headline "fewer tokens can cost more"**: already shown at scale by 2611.22238 (and CAPC). Keep only as replication.
- **"First budget-conditioned context policy"**: ContextBudget and 2610.28409 cover budget-conditioned RL; only the $-after-cache Lagrangian variant remains.
- **"First to learn when to compact"**: AutoCompact, ContextBudget, MemAct already do it.
- **Fine-tuning the agent LLM end to end** (AutoCompact / CompactionRL / MemAct style): violates the frozen-LLM rule and is infeasible on a 6 GB GPU.
- **KV-cache-level editing (AKV style)**: needs a custom serving engine and weight access; not possible on API models; outside scope.
- **LLM-planner fold/recover with retrievable cards (Self-GC, CADOC cards)**: needs retrieval plus a planner LLM per call; adds cost and complexity without serving the core claim.
- **Learned token-level prompt compressors (PCRL, TACO-RL, LLMLingua-style)**: crowded; also token-level rewriting breaks cache by construction.
- **Real-invoice validation without approval**: forbidden by project rules; only with explicit approval and a capped budget.
- **WebShop** before ALFWorld (or a real-LLM toy run) passes: stretch, adds no novelty.

## 5. Recommended framing

To our knowledge, this project is the first to train a lightweight reinforcement-learning controller, over a
frozen LLM, on the cache-adjusted bill rather than on token count. We test whether learning the edit gate under
that objective produces cache-friendly batched compaction, and compare it against analytic cache-aware rules
(CADOC-style, break-even) and a token-reward controller. Optionally, a price-conditioned variant uses one policy
across provider cache pricing. Recent measurement work (Context Has a Price) shows that token savings can raise the bill; we study
whether a learned controller can exploit that, and when analytic rules are already enough.

Main caveat to state in the report: unless a real LLM and ALFWorld (G9) are run, every result is from a toy
environment with a scripted mock agent.
