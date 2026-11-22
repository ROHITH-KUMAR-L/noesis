# Novelty check: cache-aware RL memory controller (checked 2026-11-22)

Method: WebSearch (standard + extended), arXiv abstract pages, Crossref API (filter prefix 10.1109 = IEEE)
and Semantic Scholar API for IEEE metadata/abstracts. IEEE Xplore pages themselves returned empty to the
fetcher, so IEEE facts come from Crossref / Semantic Scholar records keyed by the IEEE DOI.
"Abstract says" = read on the abstract page. "Inference" = my reading, not stated by the source.
Search is not exhaustive; arXiv volume in this area in Sep-Nov 2026 is high. Re-run before the report.

## 1. Prior-art IDs in CLAUDE.md: all 8 resolve

| ID | Resolves to | Matches CLAUDE.md description? |
|---|---|---|
| 2609.37012 | CADOC: Cache-Aware Dynamic Object Context for Long-Horizon Agents (Yao, Zhou, Xu; 29 Sep 2026) | Yes. Analytic EOQ-style batching rule; cache-reconstruction cost in objective; no RL in abstract. |
| 2607.15516 | Cache-Aware Prompt Compression: A Two-Tier Cost Model for LLM API Caching (Yan Song; 17 Jul 2026) | Yes. Analytic cost model + query-agnostic compression; no learned policy. |
| 2608.29363 | TRACER: Per-Tool Context Retention for LLM Agents via Consequence-Attributed RL (Lin et al.; 29 Aug 2026) | Mostly. REINFORCE policy over per-tool retention ratios. Objective = task success + total tokens + tool re-invocations (not pure "raw-token penalty"). Abstract does not say agent is frozen. No cache mention. |
| 2606.20954 | Learning What Not to Forget: Long-Horizon Agent Memory from a Few Kilobytes of Learning (Lia, Mazumder; v1 18 Jun 2026) | Partly. Learned keep/evict scorer (LRE) confirmed. "Token objective; budget via threshold" NOT stated in abstract (says "matched-budget comparison" only) - unverified, check the paper. |
| 2508.21433 | The Complexity Trap: Simple Observation Masking Is as Efficient as LLM Summarization for Agent Context Management (Lindenbauer et al.; NeurIPS'25 DL4C workshop per arXiv page) | Yes. |
| 2609.32961 | Beyond Token Savings: A Systematic Study of Context Compression in LLM Agents (Satish et al.; 26 Sep 2026) | MISMATCH in emphasis. Abstract is about what/when/how-much compression, latency and "estimated cost"; it does NOT mention prompt caching or cached pricing. Do not cite it as evidence that caching makes token savings cost more. Use "Context Has a Price" (2611.22238) for that. |
| 2606.17016 | TokenPilot: Cache-Efficient Context Management for LLM Agents (Xu et al.; v2 28 Aug 2026; a search snippet says Findings of EMNLP 2026 - unverified) | Yes, cache-aware. Rule vs learned not stated in abstract ("conservative batch-turn schedule"). |
| 2610.28577 | Cache Engineering for Agents (Fu et al.; 20 Oct 2026); AKV is its serving engine | MISMATCH. Not prompt-level hand-written rules: agents edit the KV cache directly (drop entries, re-position keys, no re-prefill). Also reports "recursive self-improvement of cache policies" (automated policy refinement; mechanism not in abstract, not RL per abstract). Describe it as KV-level cache engineering, not "hand-written rules". |

PCRL: verified. "Discrete Prompt Compression With Reinforcement Learning", Hoyoun Jung, Kyung-Joong Kim,
IEEE Access vol. 12, pp. 72578-72587, 2024, DOI 10.1109/ACCESS.2024.3403426 (Crossref record; arXiv 2308.08758
lists the same related DOI). IEEE Xplore doc 10535182.

## 2. Relevant papers

IEEE = yes only when the DOI prefix 10.1109 was seen in Crossref/Semantic Scholar.

| Title | Venue / year | IEEE | URL | Relevance | How we differ |
|---|---|---|---|---|---|
| Discrete Prompt Compression With Reinforcement Learning (PCRL) | IEEE Access 12, 2024 | yes | https://ieeexplore.ieee.org/document/10535182/ ; https://arxiv.org/abs/2308.08758 | RL policy (include/exclude per token) over black-box LM | Single prompt, token reduction objective, no agent history, no caching |
| Learning what to Remember: Reinforcement Learning for Cost-Aware Memory in LLM Agents (MemRL; Cringasu, Tapus) | IEEE RO-MAN 2026, pp. 1812-1818 | yes | https://doi.org/10.1109/RO-MAN72286.2026.11714580 (metadata via api.crossref.org and api.semanticscholar.org) | Closest IEEE work: REINFORCE write policy, "cost-aware" = storage penalty, ~80% less memory at equal accuracy (paraphrased S2 abstract) | Memory-write (what to store), storage cost, not prompt-cache billing; no edit-timing gate; synthetic tasks |
| AMC-MAS: Adaptive Memory Compression for Long-Horizon Multi-Agent LLM Systems | IEEE ASYU 2026 | yes | https://doi.org/10.1109/asyu71790.2026.11772441 | Title-level match only; abstract not available | Unknown - read before report |
| Context-Aware Autoscaling for Cost-Efficient LLM Inference With Prefix Cache Integration (AdaptiveScale) | IEEE Access 2026 | yes | https://doi.org/10.1109/ACCESS.2026.3677249 | Prefix-cache-aware cost at serving/autoscaling level | Infra side (GPU provisioning), not agent memory policy |
| JPPO: Joint Power and Prompt Optimization for Accelerated LLM Services | IEEE ICC 2025 | yes | https://ieeexplore.ieee.org/document/11161529/ | DRL picks compression ratio (+ wireless power) | Wireless latency objective, no caching, no agent history |
| Online Context Caching for Distributed LLM Serving | IEEE (venue/year unverified; seen only in search result) | yes (ieeexplore URL) | https://ieeexplore.ieee.org/document/11044599/ | KV context-cache placement + request scheduling as online optimization | Serving-side, not agent-side |
| Pegasus: Accelerating LLM Inference with Stateful Prefix Caching | IEEE JCC 2026 (Crossref) | yes | https://doi.org/10.1109/jcc72984.2026.00017 | Prefix caching systems (title only) | Serving-side |
| Context Has a Price: Adding and Removing Context for LLM Coding Agents under Prefix Caching (Yang et al.) | arXiv 2611.22238, 17 Nov 2026 | no | https://arxiv.org/abs/2611.22238 | 20k+ runs; removing old observations saves 19-36% without cache discount but RAISES cost 38-116% at 1/10 cache price; shorter prompts can lengthen episodes; concurrent runs share cache and distort comparisons | Measurement study only; no learned policy. Pre-empts our claim (4) as a finding. |
| Self-GC: Self-Governing Context for Long-Horizon LLM Agents | arXiv 2607.00692 | no | https://arxiv.org/abs/2607.00692 (seen as /html/2607.00692v1) | Side-channel LLM planner chooses fold/mask/prune; commit weighs expected savings vs cache-break; prune threshold 0.3 (from search snippets) | LLM planner + threshold, not RL; not in our prior-art table - add it |
| AutoCompact: Learning When to Compact Context in Long-Horizon Coding Agents | arXiv 2610.02163, 1 Oct 2026 | no | https://arxiv.org/abs/2610.02163 | Learns WHEN to compact: SFT on judge-corrected trajectories + RL; +9.2/+5.0 pts SWE-bench | Agent LLM is trained; reward = task success only per abstract; no cache cost |
| ContextBudget / BACM-RL | arXiv 2604.01664, 2 Apr 2026 | no | https://arxiv.org/abs/2604.01664 | Context mgmt as sequential decision under a budget; decides when and how much to compress; curriculum RL across varying budgets (search snippet: GRPO extension, budget-conditioned state) | Context-size budget (unit not in abstract), no cache pricing, no Lagrangian per abstract; likely trains agent |
| Rewards for Budget-Aware Multi-Turn Agents That Adapt to Arbitrary Test-Time Constraints | arXiv 2610.28409, 20 Oct 2026 | no | https://arxiv.org/abs/2610.28409 | One policy per env/resource trained across budget values (tokens, API calls, steps); overspend-penalty rewards | Fine-tunes agent (Qwen3.5); budgets on tokens/calls/steps, not cache-adjusted $; penalty rewards, not Lagrangian; no context editing |
| SeqRoute: Global Budget-Aware Sequential LLM Routing via Offline RL | arXiv 2605.25424 | no | https://arxiv.org/abs/2605.25424 | Remaining session budget in state; lambda-sweep without retraining | Routing between models, not memory editing; no cache |
| BudgetMem: Learning Query-Aware Budget-Tier Routing for Runtime Agent Memory | arXiv 2602.06025 (ICML 2026 per arXiv page) | no | https://arxiv.org/abs/2602.06025 | RL router picks memory-module budget tiers | Memory construction cost, not prompt-cache billing |
| LACONIC: Length-Aware Constrained RL for LLM | arXiv 2602.14468 | no | https://arxiv.org/abs/2602.14468 (search result) | Lagrangian primal-dual PPO with token budget (output length) | Output length, single global budget |
| Pareto-Dominant Clarification: Post-Training Coding LLMs via PPO-Lagrangian Budget Constraints | arXiv 2610.04089 | no | https://arxiv.org/pdf/2610.04089 (search result) | PPO-Lagrangian with question/turn budgets | Different budget, trains LLM |
| MemAct: Memory as Action: Autonomous Context Curation | arXiv 2510.12635 | no | https://arxiv.org/abs/2510.12635 | End-to-end RL for in-place context edits (delete/insert); -51% context | Trains agent LLM; abstract has no cache term (a search snippet says the paper notes edits lower prefix-cache hit rate - unverified) |
| MEMENTO: Learning Context Management for Long Context Processing Agents | arXiv 2611.00244, 30 Oct 2026 | no | https://arxiv.org/abs/2611.00244 | SFT + RL compaction notes; reward = correctness + note heuristics + calibration | Trains agent; cost not in reward |
| CTXAgent: Self-Controlled Context Management of LLM Agents | arXiv 2610.34425 | no | https://arxiv.org/abs/2610.34425 | Prompted model decides when to compact; notes over-frequent compaction raises other costs | Prompting, no RL, no explicit cache model in abstract |
| Don't Break the Cache: Evaluation of Prompt Caching for Long-Horizon Agentic Tasks | arXiv 2601.06007 | no | https://arxiv.org/abs/2601.06007 | Real API cost across caching strategies (OpenAI/Anthropic/Google) | Evaluation of cache layouts, no pruning policy |
| CliffCompaction: Cost-Efficient Compaction for Long-Horizon Coding Agents | arXiv 2609.26779 | no | https://arxiv.org/abs/2609.26779 | Truncate/drop-only compaction, up to 50% cost cut | Abstract silent on caching and on learning |
| PAIR (Adapting Context Compression ... Counterfactual Continuations) | arXiv 2609.36526 | no | https://arxiv.org/abs/2609.36526 | Frozen agent; adapts compression template using counterfactual rollouts | No RL, no cache |
| TACO-RL | arXiv 2409.13035 | no | https://arxiv.org/abs/2409.13035 (search result) | REINFORCE task-aware prompt compression | Single prompt, no cache |
| Memex(RL), MemSearcher, CompactionRL | arXiv 2603.04257, 2511.02805, 2607.05378 | no | seen in search results only | RL context/memory management (search snippets: Memex reward penalises overflow; CompactionRL is PPO under fixed context budget) | All train the agent; none showed cache-adjusted reward in snippets. Not opened - verify before citing. |

IEEE verdict: IEEE has little directly relevant work. Only PCRL (prompt-level RL compression) and MemRL
(RO-MAN 2026, RL memory writing with storage cost) are close. IEEE prefix-cache papers are serving-side.
Nothing in IEEE found combines RL with prompt-cache-adjusted cost.

## 3. Claim-by-claim verdicts

**(1) First RL-learned agent memory controller trained on a cache-adjusted cost objective - HOLDS WITH NARROWING.**
No source found puts the cache-adjusted (cached vs uncached input pricing) bill into an RL reward.
Cache-aware methods are analytic or rule/planner based: CADOC (EOQ rule), CAPC (cost model), TokenPilot
(schedule), Self-GC (LLM planner + commit threshold), AKV (KV-level engineering + automated policy refinement,
not RL per abstract). Learned methods optimise success plus tokens/context/storage: TRACER, LRE, MemRL (IEEE),
MemAct, AutoCompact, MEMENTO, ContextBudget, PCRL, TACO-RL. Two narrowing points: (a) AKV's "recursive
self-improvement of cache policies" is automated optimisation of cache policies - read its method section
and state precisely how ours differs (RL with a priced reward vs whatever AKV does); (b) state the frozen-LLM,
separate-controller setting, which most learned competitors do not use.

**(2) Learning WHEN to edit - HOLDS ONLY AS A COMBINATION (threatened as stated).**
Learned edit timing exists: AutoCompact ("Learning When to Compact", RL), ContextBudget ("when and how much"),
MemAct, CTXAgent (prompted). Cache-aware edit timing exists: CADOC (analytic), Self-GC (savings-vs-cache-break
commit). What remains new: a learned gate whose training signal includes cache-rebuild cost. "Rare large
compactions" is hypothesis H3, not a claim, until shown.

**(3) Budget-constrained, budget-conditioned single policy - THREATENED; keep only narrow form.**
Budget-conditioned RL context management: ContextBudget (varying context budgets, budget in state per search
snippet). Single policy across budget values: 2610.28409 (tokens/API calls/steps). Remaining budget in state
with lambda sweep: SeqRoute. PPO-Lagrangian for LLM token/turn budgets: LACONIC, 2610.04089. What remains:
the constraint is a per-episode cache-adjusted cost budget, enforced by PPO-Lagrangian, on an external
controller over a frozen LLM. Present as an extension, not a "first".

**(4) Evaluation in real cost reveals tokens vs bill disagree - PRE-EMPTED as a finding.**
"Context Has a Price" (2611.22238, 17 Nov 2026) shows exactly this on coding agents (removing old observations
saves 19-36% of cost with no cache discount but raises cost 38-116% at a 1/10 cache-read price). CAPC also predicts negative returns
for cache-busting compression; "Don't Break the Cache" measures real API cost. Our contribution here is
replication on ALFWorld/WebShop, and extension to learned controllers (token-reward vs cache-reward PPO).

## 4. Recommended wording

- "To our knowledge, this is the first memory controller for LLM agents trained with reinforcement learning on
  a cache-adjusted cost objective, i.e. task success minus the billed input cost after prefix-cache discounts.
  Prior cache-aware context managers (CADOC, CAPC, TokenPilot, Self-GC, AKV) use analytical rules, schedules or
  LLM planners; prior learned context managers (TRACER, LRE, MemAct, AutoCompact, ContextBudget, MemRL) optimise
  success with token, context-length or storage penalties."
- "The controller learns when to edit, as well as what. Learned edit timing exists (AutoCompact, ContextBudget)
  but is trained without cache cost; cache-aware timing exists (CADOC, Self-GC) but is not learned. We test
  whether a learned, cache-priced gate produces rarer, larger edits (H3)."
- Stretch: "We extend the controller with PPO-Lagrangian so a single policy, conditioned on the budget, targets
  a per-episode cache-adjusted cost budget. Budget-conditioned RL for agents exists for context, token and call
  budgets (ContextBudget; Lamb et al. 2610.28409); our budget is in billed cost after caching." (no "first")
- Evaluation: "Consistent with Yang et al. (2611.22238) and CAPC, we report both token and cache-adjusted cost,
  and test whether the gap they observed for fixed strategies also appears for learned controllers on ALFWorld."

## 5. Gaps to address to keep the claims

1. Add to prior-art table and related work: Context Has a Price (2611.22238), Self-GC (2607.00692), AutoCompact
   (2610.02163), ContextBudget (2604.01664), 2610.28409, MemAct (2510.12635), MEMENTO (2611.00244), MemRL (IEEE
   RO-MAN 2026), PCRL with full IEEE citation.
2. Fix CLAUDE.md descriptions: AKV = KV-level cache engineering with automated policy refinement, not
   hand-written prompt rules; "Beyond Token Savings" abstract does not cover caching; LRE "token objective /
   threshold budget" is unverified; TRACER objective = success + tokens + re-invocations.
3. Read AKV method section ("recursive self-improvement") and Self-GC commit rule; if Self-GC's rule is cheap
   to reproduce, consider it as a baseline variant next to `threshold_compaction` (it is the closest
   cache-aware timing rule after CADOC).
4. Claim (2)/H3 needs the no-gate ablation and edit-timing histograms vs CADOC; without them claim (2) is
   unsupported.
5. "Context Has a Price" finds the sign of the effect depends on cache price: keep the pricing-profile
   transfer ablation, and sweep cache-read multiplier. Note their pitfall: concurrent runs sharing cached
   prefixes distort cost - relevant to the vLLM validation (FR-8.5).
6. Their result also predicts observation masking (our must-beat B3) may cost more than full history under
   1/10 cache pricing - report B3 in both token and real cost.
7. Re-run this search before the report (Phase 9); activity in Oct-Nov 2026 is high.
