# PRD: Cache-Aware RL Memory Controller for LLM Agents

| | |
|---|---|
| **Project** | Noesis (package `memory_rl`) |
| **Course** | AIML Experiential Learning (EL), 5th semester |
| **Document** | Product / research requirements. *What* we build and *why*, with phases. |
| **Companion** | [`plan.md`](plan.md) is the implementation contract (invariants, APIs, per-phase tests). [`CLAUDE.md`](CLAUDE.md) is authoritative on **scope and claims** and wins any conflict. |
| **Status** | Phases 0–1 complete (32 tests passing). Phase 2 is next. No RL result exists yet. |
| **Last updated** | 2026-10-27 |

---

## 1. Summary

Multi-turn LLM agents (ReAct-style agents on ALFWorld, WebShop, coding tasks, and similar) resend their whole growing history on every call. Cost and latency grow with the history, and long, noisy histories also hurt accuracy.

Learned context-pruning methods try to fix this by **using fewer tokens**. Real LLM APIs, however, **discount cached prompt prefixes**. Every time a pruner edits an old message, everything after that message stops matching the cache and is billed at full price again. So *fewer tokens does not mean a smaller bill*, and a token-minimising pruner can cost **more** than doing nothing.

We build an **RL (PPO) memory controller** that sits between a frozen LLM agent and its history. At every turn it decides **whether** to edit the history at all, and if so, chooses **KEEP / COMPRESS / PRUNE** for each past step. Its reward is **task success minus the real, cache-adjusted API cost**, not the raw token count.

As an extension, we make the controller **budget-constrained**: a cost or token budget goes in, and a policy that stays within that budget comes out, using constrained RL (PPO-Lagrangian) with a budget-conditioned policy.

## 2. Problem statement

1. **History growth.** An agent episode of *T* turns sends prompts of length ≈ *O(t)* at turn *t*, so total input tokens grow ≈ *O(T²)*.
2. **Pruning helps tokens but can hurt cost.** Providers bill cached prefix tokens at a fraction of the normal input price. Changing message *i* forfeits the cache discount on every token after *i*. A policy that prunes a little on every turn can therefore break the cache on every call.
3. **Existing learned methods optimise the wrong objective.** Learned pruning methods (TRACER, LRE) use token-count rewards. Cache-aware methods (TokenPilot, AKV, CADOC, CAPC) use fixed rules or analytical formulas, not learning. CADOC is the closest: it puts the cache-rebuild cost in its objective and decides analytically when to batch edits. "Beyond Token Savings" documents the gap between token savings and real cost.
   *These positioning claims come from the team's literature search (see §16 and `CLAUDE.md`) and must be re-verified in Phase 9 before they appear in any report.*
4. **The decision is sequential and has delayed credit.** An edit's cost depends on where the item sits and on every later call. That is exactly the setting where RL should beat fixed rules, and this project tests whether it does.

## 3. Goals and non-goals

### 3.1 Goals (objectives)

| ID | Goal |
|---|---|
| G1 | Build a PPO memory controller that chooses KEEP / COMPRESS / PRUNE per past step for an agent running on a **frozen** LLM. |
| G2 | Design a **cache-aware reward** that trades task success against the **real API cost after caching discounts**. |
| G3 | Compare on **ALFWorld** (core) and **WebShop** (stretch) against full history, sliding window, old-observation masking (**must-beat**), LLMLingua, a hand-written cache-aware compaction rule, a **CADOC-style analytical rule**, and **the same controller trained on a token-count reward**. Plot **success vs real cost**. |
| G4 (stretch) | **Budget-constrained control:** the controller respects a given per-episode cost budget or per-call token budget, and one budget-conditioned policy covers the whole success-vs-cost frontier. |
| G5 | Every number in the report comes from a real, reproducible run (fixed seeds and splits, plus per-episode logs). |

### 3.2 Scope priority

- **Core (must finish):** G1, G2, G3 and G5 on **one** benchmark: **ALFWorld**, unless the team decides otherwise.
- **Stretch:** G4 (budget control, Phase 7) and the second benchmark (WebShop).
- Don't start stretch work while a core phase's exit gate is still failing.

### 3.3 Non-goals

- Fine-tuning or training the LLM itself. The LLM is frozen; only the controller learns.
- Building a production memory service, multi-user persistence, or a UI.
- Retrieval (RAG) inside the controller's action space.
- Claiming general agent capability beyond the evaluated environments.
- Spending money on paid APIs during training. Paid calls are allowed only for an optional, pre-approved validation run (Phase 8).

## 4. Research questions and hypotheses

| ID | Hypothesis | How it is tested |
|---|---|---|
| **H1** | At matched task success, the cache-aware PPO controller has a **lower real (cache-adjusted) cost** than every baseline, including observation masking, the CADOC-style rule and the token-reward PPO controller. | Success vs real-cost Pareto plots; paired bootstrap at matched success (Phase 8) |
| **H2** | Token-count-optimised policies can use **fewer tokens yet cost more** than a cache-friendly policy, and sometimes more than full history. | Same runs plotted as success vs tokens *and* success vs $; report where the rankings disagree |
| **H3** | The two controllers behave differently. The token-reward controller edits **often and in small amounts**; the cache-aware controller makes **rare, large "compaction" edits** and keeps the prefix stable in between. CADOC already times its edits analytically, so the learned gate is also compared against CADOC's edit timing. | Edits per episode, tokens removed per edit, cache-hit rate, edit-timing histograms (incl. CADOC-style rule) |
| **H4** (stretch) | A single **budget-conditioned** PPO-Lagrangian policy meets the given budget in ≥ 95% of episodes. Its success is close to that of controllers trained separately for each budget. | Budget sweep; compliance rate; per-budget gap (Phase 7) |

Negative or mixed outcomes are valid results and get reported as such. H1–H4 are hypotheses, not promises.

## 5. Users and stakeholders

- **Research team (students):** build, run and report the study.
- **Faculty guide and evaluators:** need a clear question, fair baselines, honest results and a working demo.
- **Practitioners (motivating user):** anyone running agents on paid APIs who wants a lower bill without losing success.

## 6. System overview

```text
            ┌──────────── ALFWorld / WebShop environment ───────────┐
            │  task instruction, observations, reward / success     │
            └───────────────▲──────────────────────────┬────────────┘
                     action │                          │ observation
                            │                          ▼
┌──────────────┐   prompt   ┌──────────────────────────────────────┐
│ Frozen LLM   │◄───────────│ Agent harness (ReAct loop)           │
│ (vLLM, prefix│───────────►│  history = memory items              │
│  caching on) │  response  │  protected prefix: system + task     │
└──────────────┘            └──────────▲──────────────┬────────────┘
                                       │ edited       │ items + features
                                       │ history      ▼
                            ┌──────────────────────────────────────┐
                            │ Memory controller (PPO policy)       │
                            │  gate: edit now?                     │
                            │  per-item: KEEP / COMPRESS / PRUNE   │
                            └──────────▲───────────────────────────┘
                                       │ reward
                            ┌──────────┴───────────────────────────┐
                            │ Cache cost model (provider profiles) │
                            │  cached prefix → $ per call          │
                            └──────────────────────────────────────┘
```

Before every LLM call, the controller sees the current history and may edit it. The edited history is serialized (stable prefix first), the cost model works out how much of it hits the cache and prices the call, and the LLM responds. The environment advances. The episode ends on success, failure, the step limit, or a context or budget violation.

## 7. Functional requirements

### 7.1 Memory engine (built: Phase 1)

| ID | Requirement |
|---|---|
| FR-1.1 | Each past agent step (thought/action, tool observation) is a `MemoryItem` with a stable ID, current text, immutable original text, token count, type and status. |
| FR-1.2 | KEEP, COMPRESS and PRUNE are engine operations with structured results. COMPRESS is valid only if it produces fewer tokens. A pruned item never reaches the prompt. |
| FR-1.3 | System instructions and the task instruction are **protected** and can never be edited. |
| FR-1.4 | Simulator-only labels (gold actions, success flags, reference answers) can never enter memory metadata, features or prompts. |

### 7.2 Cache-aware cost model (Phase 2)

| ID | Requirement |
|---|---|
| FR-2.1 | Config-driven **provider pricing profiles**: input $/token, output $/token, cache-read multiplier, cache-write multiplier, minimum cacheable prefix, cache block size, TTL, mode (`automatic` prefix caching vs explicit `breakpoint`). Every profile records its source URL and date. Prices are never hardcoded in source. |
| FR-2.2 | A **prefix-cache simulator** computes, for each call, the cached prefix length (longest common token prefix with live cache entries, floored to the block size, zero if under the minimum), newly written cache tokens, and uncached tokens. |
| FR-2.3 | Per-call cost = `p_in·uncached + p_in·m_read·cached + p_in·m_write·written + p_out·output`, logged per term. |
| FR-2.4 | A token-count cost (same interface) for the token-reward baseline controller. |
| FR-2.5 | Deterministic and unit-tested against hand-computed examples, including edits at the first, middle and last item, TTL expiry, the below-minimum-prefix case and block rounding. |

### 7.3 Agent harness (Phase 3)

| ID | Requirement |
|---|---|
| FR-3.1 | Pluggable **LLM backend**: an OpenAI-compatible client for a local vLLM server (prefix caching on), and a scripted **mock LLM** for tests and CI. |
| FR-3.2 | ReAct agent loop for **ALFWorld** (core) and **WebShop** (stretch), plus a tiny **toy tool environment** for fast pipeline debugging. |
| FR-3.3 | Agent prompt layout puts the stable content first (system, then task, then history in order), so the prefix can be cached. The tokenizer must match the frozen LLM's tokenizer (HF tokenizer + chat template). |
| FR-3.4 | A hook before every LLM call where a memory policy (baseline or PPO) can edit history. |
| FR-3.5 | Per-episode trajectory logs: prompts (as hashes plus token counts), responses, actions, observations, edits, cache stats, cost terms, success/score, latency. |
| FR-3.6 | Context-limit overflow ends the episode as a recorded failure. Prompts are never silently truncated. |

### 7.4 Controller MDP and reward (Phase 4)

| ID | Requirement |
|---|---|
| FR-4.1 | **Action space (hierarchical):** at each turn a **gate** chooses `NO_EDIT` or `EDIT`; on `EDIT`, each eligible item gets KEEP / COMPRESS / PRUNE under its validity mask. |
| FR-4.2 | **Observations** (inference-available only), per item: age, position, token count, type, compressed flag, compression gain, relevance to the task instruction, and **suffix tokens after the item** (the re-billing cost of editing it). Global: turn / max turns, prompt tokens / context limit, last cached-prefix fraction, pricing multipliers; budget features added in Phase 7. |
| FR-4.3 | **Reward (cache variant):** `R = R_task − λ · Σ_calls cost_k / Z`, where `R_task` is ALFWorld success (0/1) or the WebShop score in [0,1], and `Z` is a fixed normaliser (mean full-history episode cost on a calibration set). |
| FR-4.4 | **Reward (token variant):** identical except `cost_k` is replaced by the prompt token count. Everything else (network, features, training budget, λ sweep) stays the same. |
| FR-4.5 | Every reward term is logged separately. Rewards and observations are always finite. Episodes have a finite horizon. |

### 7.5 Baselines (Phase 5)

| ID | Baseline | Swept knob |
|---|---|---|
| B1 | Full history (no edits) | — |
| B2 | Sliding window (last *K* turns) | *K* |
| B3 | Observation masking: old tool outputs → placeholder, thoughts/actions kept. **Must-beat baseline** (simple and strong in prior work). | age *M* |
| B4 | LLMLingua(-2) compression of observations older than *M*, compressed once | rate *r*, *M* |
| B5 | Threshold compaction: a hand-written cache-aware rule that lets context grow to *T* tokens, then compacts once (stand-in for TokenPilot/AKV-style rules) | *T* |
| B6 | Random valid policy; always-KEEP control | — |
| B7 | **Same PPO controller trained on token-count reward** (our stand-in for TRACER-style token-penalised RL) | λ |
| B8 | **CADOC-style analytical rule:** includes the cache-rebuild cost and decides analytically (EOQ-style) when to batch edits. Re-implemented from the paper's description; the formula must be checked against the source, and any deviation documented. | rule parameters as defined by CADOC |

All baselines use the same harness, LLM, prompt layout, tokenizer, cost model and scorer. Knobs are tuned on validation only.

### 7.6 PPO controller (Phase 6)

| ID | Requirement |
|---|---|
| FR-6.1 | Shared per-item encoder with pooled episode context. Gate head, per-item action head with masking, and value head. Joint log-prob = gate + Σ item log-probs. |
| FR-6.2 | PPO with clipped objective, GAE, entropy bonus, value loss, gradient clipping and masked logits (an invalid action has probability exactly 0). |
| FR-6.3 | Train both the cache-reward and token-reward variants across a λ sweep, with ≥ 3 seeds each (5 if compute allows). |
| FR-6.4 | Checkpoints store weights, optimiser state, config, feature-schema version, tokenizer/LLM identity, pricing profile, seed and step. Loading an incompatible checkpoint fails loudly. |

### 7.7 Budget-constrained extension (Phase 7, stretch)

| ID | Requirement |
|---|---|
| FR-7.1 | **Soft dollar budget (CMDP):** maximise E[R_task] subject to E[episode cost] ≤ B. A Lagrange multiplier is updated by dual ascent: `λ ← max(0, λ + η(Ĉ − B))`. |
| FR-7.2 | **Hard token budget:** every call must satisfy prompt tokens ≤ B_tok. When the next prompt would exceed it, the gate is **forced** to EDIT (logged as forced). If the cap is still exceeded after the edit, the episode ends as a recorded budget failure. |
| FR-7.3 | **Budget-conditioned policy:** B is sampled per episode from a range; normalised B and the remaining-budget fraction are observations. Multipliers are kept per budget bin. |
| FR-7.4 | Report compliance rate, success and cost for each budget, compared with per-budget policies and with baselines tuned to the same budget. |

### 7.8 Evaluation and reporting (Phase 8)

| ID | Requirement |
|---|---|
| FR-8.1 | Held-out test splits are touched only once, for final numbers. |
| FR-8.2 | Main plot: **task success vs real cost ($/episode)** per environment and pricing profile, as Pareto frontiers across each method's knob sweep. |
| FR-8.3 | Secondary plots: success vs input tokens (to show H2), cache-hit rate, edits per episode and edit timing (H3), budget compliance (H4). |
| FR-8.4 | Paired bootstrap confidence intervals over tasks, with per-seed variation shown. Failed and over-budget episodes are included. |
| FR-8.5 | Cost-model validation: compare simulated cached tokens against vLLM prefix-cache hit statistics. Optional and needing explicit approval: a small paid-API run comparing provider-reported usage to the simulator. |

## 8. Metrics

| Metric | Definition |
|---|---|
| **Success rate** | ALFWorld: fraction of games solved. WebShop: fraction with reward = 1, plus mean score. |
| **Real cost / episode** | Σ per-call cache-adjusted cost under a stated pricing profile (FR-2.3) |
| **Cost per success** | total cost / successes; undefined (reported as ∞) when there are 0 successes |
| **Input tokens / episode** | Σ prompt tokens (cached + uncached) |
| **Cache-hit rate** | Σ cached tokens / Σ prompt tokens |
| **Episode length** | turns and LLM calls. Pruning can make an agent take *more* steps, and that extra cost counts. |
| **Edit statistics** | edits per episode, tokens removed per edit, turn at which edits happen, action distribution |
| **Budget compliance** | fraction of episodes within B (cost) or with no call over B_tok |
| **Latency** | measured end-to-end time per episode (controller + compression + LLM), boundary stated |
| **Failure attribution** | context overflow, budget violation, step limit, wrong action after an edit |

## 9. Experimental setup (defaults; verify in Phase 3)

| Item | Default |
|---|---|
| Frozen LLM | Open instruct model of about 7–8B (e.g. Qwen2.5-7B-Instruct) served by vLLM with automatic prefix caching, temperature 0 |
| Fallback | Smaller (3–4B) or quantized model if the GPU is limited. Must pass the Phase 3 competence gate. |
| Environments | **Core:** ALFWorld (train games, then `valid_seen` for validation, `valid_unseen` for test). **Stretch:** WebShop (train, dev, test). Verify split sizes and licenses. |
| Pricing profiles | At least two contrasting profiles, e.g. an "automatic caching, moderate discount" profile and an "explicit breakpoints, large read discount, write premium" profile. Values come from provider pricing pages, with source and date. |
| Compute | One GPU (≥ 24 GB for a 7B bf16 model) for the LLM server; the controller trains on the same machine or a CPU. |
| Seeds | ≥ 3 training seeds per configuration (5 preferred) |

## 10. Phases and deliverables

Each phase has a hard **exit gate**; don't start the next dependent phase until it passes. Effort sizes (S ≈ < 1 week, M ≈ 1–2 weeks, L ≈ 2–3 weeks for a team of about 4) are planning estimates only.

| Phase | Name | Main deliverable | Exit gate | Effort | Depends on | Status |
|---|---|---|---|---|---|---|
| **0** | Setup | Package, Docker, CI, smoke | Smoke + CI green | S | — | ✅ Done |
| **1** | Memory engine | Items, ops, store, serializer, tokenizer | Phase 1 tests (32 passing) | M | 0 | ✅ Done |
| **2** | Cache cost model | Pricing profiles, prefix-cache simulator, cost meters | Hand-computed cost cases pass; prune-at-*i* re-bill test | M | 1 | ⏭ Next |
| **3** | Agent harness | LLM backends (vLLM + mock), toy env, ALFWorld adapter (WebShop adapter: stretch), logging | Mock runs in CI; frozen LLM reaches a non-trivial full-history success rate on ALFWorld validation | L | 1 | ☐ |
| **4** | Controller MDP + reward | Observation features, hierarchical actions, masks, both reward variants | 1,000 random-policy episodes (toy env) with no NaN, hang or leak; reward terms add up | M | 2, 3 | ☐ |
| **5** | Baselines | B1–B6 and B8 (CADOC-style) with knob sweeps on a common runner | All baselines produce schema-valid records on ALFWorld validation | M | 2, 3 | ☐ |
| **6** | PPO controller | PPO training, token-reward and cache-reward variants, checkpoints | PPO beats random/always-KEEP on the toy env; a small ALFWorld run trains end-to-end and reloads | L | 4 | ☐ |
| **7** | Budget-constrained ext. *(stretch)* | PPO-Lagrangian, hard token cap, budget-conditioned policy | Budget compliance ≥ 95% on validation at three budgets (or the gap is reported honestly) | M | 6 + all core gates | ☐ |
| **8** | Evaluation | Test-set runs, Pareto plots, statistics, cost-model validation | All methods × profiles × seeds (× budgets if Phase 7 done) evaluated on identical ALFWorld test tasks; WebShop if stretch done | M | 5, 6, (7) | ☐ |
| **9** | Analysis + report | Ablations, novelty re-check, failure analysis, final report and demo | Every claim in the report traces back to a logged run | M | 8 | ☐ |

### 10.1 Phase details

**Phase 2: Cache cost model.** Build `memory_rl/cost/`: `PricingProfile` (config-loaded, with source and date), `PrefixCacheSimulator` (token-level longest-common-prefix against live entries, block rounding, minimum prefix, TTL on a simulated clock, `automatic` and `breakpoint` modes), `CacheAwareCostMeter`, `TokenCostMeter`, and a per-call `CallCost` record. This needs no GPU and no LLM, so it can be built in parallel with Phase 3.
*Key test:* in a 10-item history, pruning item 2 re-bills about 8 items' worth of tokens, while pruning item 9 re-bills about one item's worth.

**Phase 3: Agent harness.** Add `memory_rl/llm/` (OpenAI-compatible vLLM client, mock LLM, HF tokenizer adapter) and `memory_rl/envs/` (toy tool env, ALFWorld adapter, ReAct loop, agent prompt serializer with the protected prefix first). The WebShop adapter is stretch work, built only after the core gates pass. Log trajectories in JSONL.
*Competence gate:* if the frozen LLM's full-history success on validation is too low to measure differences (for example < 20%), change the model or prompt **before** any RL. Without signal the study means nothing.

**Phase 4: Controller MDP and reward.** Add `memory_rl/controller/env.py`, `features.py`, `reward.py`. Define the hierarchical gate plus per-item action, masks, observation schema (versioned), both reward variants and the normaliser `Z`. Verification: always-KEEP, aggressive PRUNE and random policies on the toy env, plus leakage tests.

**Phase 5: Baselines.** Add `memory_rl/baselines/` covering B1–B6 and B8 (CADOC-style rule) and a common `run_policy(env, policy, tasks)` runner that writes standard per-episode records. LLMLingua is an optional dependency, installed only here.

**Phase 6: PPO controller.** Add `memory_rl/controller/{policy,ppo,train}.py`. Workflow: debug on the toy env, then a small ALFWorld subset, then full training runs. Train cache-reward and token-reward variants across the λ sweep and seeds. Select checkpoints on validation only.

**Phase 7: Budget-constrained extension (stretch).** Add `memory_rl/controller/lagrangian.py`, budget features and the hard-cap forcing rule. Compare the single budget-conditioned policy against per-budget policies. *Behaviour check for H3:* under a hard token cap, does the cache-aware policy compact rarely and in large chunks while the token policy trims every turn?

**Phase 8: Evaluation.** Run `memory_rl/evaluation/` once on the frozen test split for every method, knob, budget, pricing profile and seed. Produce tables, Pareto plots, bootstrap CIs and the cost-model validation. The paid-API validation is optional and needs the team's explicit approval and budget.

**Phase 9: Analysis and report.** Ablations: no gate (flat per-item actions); no suffix-tokens feature; no COMPRESS; pricing-profile transfer (train under one profile, test under another); a different LLM. Also the novelty re-check on arXiv against CADOC, CAPC, TRACER, LRE, TokenPilot, AKV, observation masking, "Beyond Token Savings" and any newer work; failure analysis; the final report, slides and a live demo episode.

### 10.2 Workstreams (for dividing the team)

| Workstream | Owns phases | Notes |
|---|---|---|
| **A: Agent & environments** | 3, (support 8) | vLLM serving, ALFWorld/WebShop setup, ReAct prompt, logging |
| **B: Cost model & baselines** | 2, 5 | Pricing research, simulator, B1–B6, LLMLingua |
| **C: RL controller** | 4, 6, 7 | Features, reward, PPO, Lagrangian, training runs |
| **D: Evaluation & report** | 8, 9 | Metrics, plots, statistics, ablations, literature/novelty, report |

Phases 2 and 3 run in parallel (B and A), followed by 4 and 5 (C and B), then 6 (C, with D preparing evaluation scripts on the toy env). Phase 7 and the WebShop adapter are stretch work, picked up only after every core gate passes.

## 11. Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-1 | **Reproducibility:** seeds for environments, controller init, PPO sampling and evaluation; versioned configs; each run records its git commit, LLM, tokenizer and pricing profile. |
| NFR-2 | **Fairness:** identical tasks, LLM, decoding, prompt layout, tokenizer, cost model and scorer for every method. Tuning on validation only. |
| NFR-3 | **Honesty:** no fabricated or illustrative numbers presented as measured; failures and budget misses are reported; nothing missing is silently estimated. |
| NFR-4 | **Testability:** CI runs without a GPU, network or paid API, using the mock LLM and toy env. |
| NFR-5 | **Code quality:** ruff + mypy + pytest green on every commit (existing CI). |
| NFR-6 | **Safety:** no secrets in git; API keys come only from env vars; raw benchmark data stays out of git. |
| NFR-7 | **Cost control:** zero paid-API spend by default; any paid run needs an explicit, recorded budget. |

## 12. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| RL rollouts with a real LLM are slow | Too little training signal | Toy env for debugging; short-horizon subsets; vLLM batching; controller acts only before LLM calls; budget training episodes per seed |
| Frozen LLM too weak on ALFWorld/WebShop | No measurable success differences | Phase 3 competence gate; stronger model or better prompt before RL |
| Cache simulator differs from real providers | Cost claims don't hold up | Two contrasting profiles; check against vLLM cache stats; optional paid validation (FR-8.5) |
| The hand-written or CADOC-style rule (B5, B8) or observation masking (B3) matches PPO | H1 not supported over rules | A valid finding; report it, analyse when RL helps (e.g. variable-length observations) |
| Reward hacking (e.g. pruning the task-critical observation) | Low success | Terminal success dominates; per-term logs; trajectory inspection |
| Pruning lengthens episodes | Hidden extra cost | Cost is summed over *all* calls; episode length reported |
| Novelty claim weakens after the final literature check | Positioning changes | Phase 9 re-check; contribution can be framed as the empirical study plus the budget-constrained extension |
| GPU access limited | Phases 6–8 delayed | Smaller model; fewer seeds (report the exact number); prioritise ALFWorld over WebShop |

## 13. Acceptance criteria (project complete)

1. **Core:** Phases 0–6 and 8 pass their exit gates on ALFWorld.
2. ALFWorld is evaluated with all baselines (including B3 observation masking and B8 CADOC-style), both PPO variants, ≥ 2 pricing profiles and ≥ 3 seeds (or the exact reduced number is stated).
   **Stretch, if attempted:** Phase 7 passes or its shortfall is documented; WebShop is evaluated with the same protocol.
3. Success-vs-real-cost Pareto plots, success-vs-tokens plots and H3 behaviour plots can be regenerated from logged per-episode records with one command.
4. The report states for each of H1–H4 whether it was supported, refuted or inconclusive, with confidence intervals.
5. CI is green; the README explains how to reproduce every figure.

## 14. Open questions

- Which GPU and how many GPU-hours are available? This sets the model size and the number of seeds.
- What is the final choice of frozen LLM, after the Phase 3 competence gate?
- Core benchmark: ALFWorld by default. Does the team want a different one?
- Which provider pricing profiles do we use (source and date recorded)?
- Is any budget available for the optional paid-API validation?
- When is the semester deadline? Map phases to calendar weeks once it's known.

## 15. Novelty (claim exactly this, no stronger)

- **To our knowledge, the first RL-learned memory controller for agents trained on a cache-adjusted cost objective.** Prior cache-aware methods (TokenPilot, AKV, CADOC, CAPC) use fixed rules or formulas; prior learned pruners (TRACER, LRE) reward token count.
- **Learning *when* to edit, not just what.** An explicit edit-now-or-not gate. Hypothesis (H3): it learns rare, large compactions that keep the cached prefix stable, while token-based pruners trim every turn. CADOC already decides when to batch edits analytically, so it is the closest competitor here.
- **Budget-constrained, budget-conditioned control (stretch).** A single policy meets a given budget as a constraint at runtime, rather than tuning a threshold or penalty for each budget.
- **Evaluation in real cost.** Show empirically when "fewer tokens" and "lower bill" disagree, which token-only evaluations cannot reveal.

Never write "the first" without "to our knowledge". Re-check arXiv for new prior art before the report (Phase 9).

## 16. Key prior art

| Work | ID | Relevance |
|---|---|---|
| CADOC | arXiv 2609.37012 | Closest. Cache-rebuild cost in the objective, but an analytical (EOQ-style) rule, not learned. Baseline B8. |
| CAPC | arXiv 2607.15516 | Cache-aware, analytical. Cite. |
| TRACER | arXiv 2608.29363 | RL over a frozen agent, raw-token penalty. Token-reward PPO (B7) is our stand-in. |
| LRE | arXiv 2606.20954 | Learned keep/evict scorer, token objective; budget via threshold. |
| Observation masking | arXiv 2508.21433 | Simple and strong. **Must-beat baseline** (B3). |
| Beyond Token Savings | arXiv 2609.32961 | Token savings vs real cost; motivation. |
| TokenPilot | arXiv 2606.17016 | Cache-aware hand-written rules. |
| AKV | arXiv 2610.28577 | Cache-aware hand-written rules. |
| PCRL | IEEE Access, 2024 | IEEE-indexed citation for RL-based prompt compression. |

Taken from `CLAUDE.md`. Verify every bibliographic detail (authors, venue, DOI) against the source before it goes in a report.

## 17. Glossary

- **Prefix caching:** the provider or server reuses computation for the longest prompt prefix it has seen before and bills it at a discount.
- **Cache-adjusted cost:** the real bill after applying cache-read and cache-write prices.
- **Compaction:** one edit that removes or compresses many items at once.
- **CMDP / PPO-Lagrangian:** constrained MDP solved by PPO on `reward − λ·cost`, with λ adapted by dual ascent to meet a budget.
- **Frozen LLM:** the agent's language model; never trained in this project.
