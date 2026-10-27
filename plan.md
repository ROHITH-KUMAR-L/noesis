# Implementation Plan: Cache-Aware RL Memory Controller for LLM Agents

## 1. Instructions for the AI coding agent

Read [`CLAUDE.md`](CLAUDE.md) first, then this document before changing code. This document is the implementation contract. [`prd.md`](prd.md) holds the product/research requirements, hypotheses and phase overview; this file holds the invariants, interfaces, per-phase work, tests and exit gates. If `CLAUDE.md` disagrees with this file on **scope or claims**, `CLAUDE.md` is newer: flag the conflict rather than silently picking one.

Before implementing:

1. Inspect the actual workspace, code, tests, project metadata and Git status. Do not assume this plan's snapshot is current.
2. Preserve existing conventions and user changes. Do not overwrite or remove unrelated work.
3. Adapt to the existing architecture and document meaningful deviations. Do not create duplicate modules just to match the tree below.
4. Implement one phase at a time. Run its focused tests and satisfy its exit gate before moving on.
5. Do not silently skip a failing test, fabricate results, or describe illustrative values as measured.
6. Keep secrets, API keys, raw benchmark data and absolute local paths out of source control and reports.
7. When finished, report what was implemented, the commands run, test results, known limitations, and the next incomplete milestone.

When a requirement is underspecified, use the defaults here. Ask the user only when a decision changes the research question, needs a paid or external service, or cannot be made safely from the repository.

### 1.1 History of this plan

The first version of this plan (commit `67386a3`) targeted a DQN controller on synthetic QA under a token budget, with LoCoMo transfer. The project was re-scoped to the current problem: PPO, ALFWorld/WebShop agents on a frozen LLM, and a cache-adjusted cost objective, with the token budget kept as a constrained-RL extension. Phase 0 and Phase 1 code (memory engine, tokenizer, serializer) carries over unchanged.

## 2. Project definition

### 2.1 Working title

**Cache-Aware Reinforcement Learning for Memory Management in Multi-Turn LLM Agents**

### 2.2 What we are building

A research prototype in which a frozen LLM agent (ReAct loop) acts in ALFWorld (core) and WebShop (stretch). Before every LLM call, a small **PPO memory controller** may edit the agent's history:

- a **gate** decides `NO_EDIT` or `EDIT` for this turn;
- on `EDIT`, each eligible past item receives **KEEP**, **COMPRESS** (fixed, non-learned compressor, only when the result is shorter) or **PRUNE** (removed from the active prompt; archived for analysis).

The reward is task success minus the **cache-adjusted API cost** of every LLM call, computed by a prefix-cache simulator under configurable provider pricing profiles. The comparison controller is identical but trained on a **token-count** cost. A constrained extension adds dollar and token **budgets** (PPO-Lagrangian, budget-conditioned policy).

Central question:

> At matched task success, does a controller trained on cache-adjusted cost achieve a lower real API cost than token-count-trained controllers and hand-written context-management rules?

### 2.3 Research claim boundaries

- Claims concern **context management for a frozen LLM agent**, not LLM capability.
- Cost claims hold **under stated pricing profiles and the simulator's cache model**; validation against real usage is reported separately (Phase 8).
- PPO superiority is a hypothesis. A reliable negative or inconclusive result is a valid outcome.
- Positioning versus CADOC, CAPC, TRACER, LRE, TokenPilot, AKV and other work must be re-verified (Phase 9) before any novelty claim is reported. Claim exactly the novelty in `CLAUDE.md` / `prd.md` §15, always with "to our knowledge".

### 2.4 Scope priority

- **Core (must finish):** Phases 0–6 and 8 on **ALFWorld** (unless the team decides otherwise).
- **Stretch:** Phase 7 (budget control) and WebShop.
- Don't start stretch work while a core phase's exit gate is still failing.

## 3. Scope and fixed initial decisions

| Decision | Initial implementation |
|---|---|
| Language/runtime | Python ≥ 3.11 (see `pyproject.toml`) |
| Learning algorithm | PPO (clipped objective, GAE, masked categorical heads) |
| Decision point | Once per agent turn, immediately before the LLM call |
| Action space | Hierarchical: gate {NO_EDIT, EDIT}; per eligible item {KEEP, COMPRESS, PRUNE} with validity masks |
| Memory unit | One item per agent step component: assistant thought/action, tool observation |
| Protected content | System prompt, environment task instruction, few-shot examples |
| Compression | Fixed compressor (rule-based initially; LLMLingua-2 optional) applied at most once per item |
| Frozen LLM | Open instruct model (default ~7–8B, e.g. Qwen2.5-7B-Instruct) via vLLM, prefix caching on, temperature 0 |
| Tokenizer | The frozen LLM's own HF tokenizer + chat template for all cost/cache accounting; tiktoken / test tokenizer kept for unit tests |
| Cost | Prefix-cache simulator + pricing profiles from config (never hardcoded prices) |
| Environments | Toy tool env (CI/debug), ALFWorld (core), WebShop (stretch) |
| Baselines | Full history, sliding window, observation masking (must-beat), LLMLingua, threshold compaction rule, CADOC-style analytical rule, random, always-KEEP, token-reward PPO |
| Main metric | Task success vs real cost per episode (Pareto frontier) |
| Reproducibility | Seeds, fixed splits, versioned configs, per-episode JSONL records |

Retrieval is not in the action space.

## 4. System behavior and invariants

### 4.1 End-to-end flow

```text
env.reset(task) → protected prefix (system + task instruction)
loop per turn t:
    history items + features ──► controller (gate, per-item actions)
    memory engine applies actions (archive keeps originals)
    agent serializer: protected prefix + active items (in order)
    cache simulator: cached prefix vs live entries → CallCost
    frozen LLM → thought/action  → new ASSISTANT item
    env.step(action) → observation → new TOOL_RESULT item
until success | failure | step limit | context/budget violation
episode record: success/score, Σ costs, tokens, cache stats, edits, latency
```

### 4.2 Invariants every implementation must preserve

1. A pruned item is absent from every subsequent prompt. Archiving it for diagnostics must not make it available to the LLM.
2. System instructions, the environment's task instruction and few-shot examples are outside the controller's action space and cannot be pruned or compressed.
3. Ground-truth or simulator-only information (gold trajectories, success flags before termination, expert plans, reference answers, `is_critical` labels) never enters policy observations, features, memory metadata or prompts.
4. All methods in a comparison use the same tasks, LLM and decoding settings, prompt layout, tokenizer, compressor (where applicable), cost model and pricing profile, and scorer.
5. Cost is computed from the **exact serialized token sequence** sent to the LLM, using the LLM's tokenizer and chat template. Memory-only tokens are reported separately.
6. Prompts are never silently truncated. A context-limit overflow is a recorded episode failure.
7. COMPRESS is valid only if the result is shorter under the configured tokenizer. Compression is not assumed to preserve meaning; task success measures that.
8. Every episode has a finite horizon (environment step limit) and an explicit terminal reason.
9. Every reported metric comes from actual run data. Unmeasured costs or latencies are marked unavailable, never estimated as measured.
10. Randomness is controlled by explicit seeds: environment/task sampling, controller init, PPO action sampling, minibatch order and evaluation.
11. Cost is summed over **all** LLM calls in an episode, so a policy that makes the agent take more steps pays for them.
12. Prices live only in versioned pricing-profile configs that record their source and date.

## 5. Repository layout

Existing modules are marked ✅. Others are planned and created when their phase lands.

```text
.
├── README.md  prd.md  plan.md  pyproject.toml  Dockerfile  docker-compose.yaml
├── configs/
│   ├── pricing/               # provider profiles (YAML, with source + date)
│   ├── env/                   # alfworld.yaml, webshop.yaml, toy.yaml
│   ├── ppo.yaml  lagrangian.yaml  baselines.yaml  evaluation.yaml
├── data/                      # raw/ gitignored; environment assets downloaded by scripts
├── src/memory_rl/
│   ├── smoke.py               ✅
│   ├── memory/                ✅ models, operations, store, prompt, tokenizer
│   ├── cost/                  # Phase 2: pricing.py, cache_sim.py, meter.py
│   ├── llm/                   # Phase 3: backend.py (vLLM/OpenAI-compatible), mock.py, hf_tokenizer.py
│   ├── envs/                  # Phase 3: base.py, toy.py, alfworld.py, webshop.py, agent_loop.py, agent_prompt.py
│   ├── controller/            # Phase 4/6/7: env.py, features.py, reward.py, policy.py, ppo.py, train.py, lagrangian.py
│   ├── baselines/             # Phase 5: full_history.py, sliding_window.py, obs_masking.py, llmlingua.py, threshold_compaction.py, cadoc_rule.py, random_policy.py
│   └── evaluation/            # Phase 8: records.py, run_eval.py, metrics.py, plots.py, stats.py
├── tests/                     ✅ test_smoke, test_memory_operations, test_tokenizer_and_prompt
│                              # planned: test_cost_model, test_agent_harness, test_controller_env,
│                              #          test_no_information_leakage, test_baselines, test_ppo, test_lagrangian
├── scripts/                   # environment setup, vLLM launch, figure regeneration
├── checkpoints/  results/     # generated; gitignored
```

Heavy dependencies (torch, vllm, alfworld, webshop, llmlingua, transformers) go in optional extras (`[rl]`, `[envs]`, `[llmlingua]`) and are added only when their phase needs them. CI must keep working without them.

## 6. Data and API contracts

### 6.1 Memory item ✅

Implemented in `memory/models.py`. Fields: `memory_id`, `text`, `original_text`, `token_count`, `turn_index`, `memory_type`, `status`, `compression_count`, `dependencies`, `metadata`. `metadata` may hold only inference-available attributes; simulator-only labels are rejected at construction time (`SIMULATOR_ONLY_LABELS`).

For agents: assistant thought/action → `MemoryType.ASSISTANT`; environment observation → `MemoryType.TOOL_RESULT`. Protected content is never stored as an item.

### 6.2 Tokenizer and prompt serializers ✅ / Phase 3

- `Tokenizer` protocol (`name`, `version`, `count`) with `TestTokenizer` and `TiktokenTokenizer` ✅. Phase 3 adds an HF tokenizer adapter matching the frozen LLM. It also exposes `encode()` returning token IDs, which the cache simulator needs for prefix comparison.
- `PromptSerializer` ✅ (memory block, then question) stays for QA-style tests.
- Phase 3 adds `AgentPromptSerializer`: chat messages in the order **[system] → [task instruction + few-shot] → history items in turn order**, rendered with the LLM's chat template. It returns token IDs, `prefix_tokens` (protected), `memory_tokens`, `total_prompt_tokens`, and `included_ids`. Stable content comes first so caching works; an edit to item *i* changes only tokens from *i* onward.
- Fail loudly when the tokenizer cannot load. Never fall back to character or word counts.

### 6.3 Memory operations ✅

Implemented in `memory/operations.py` and `memory/store.py`: KEEP, COMPRESS and PRUNE return `OperationResult`; `valid_actions()` gives the mask in canonical order; the store keeps the active view, the archive and a hash-only provenance log. Phase 4 adds the rule that COMPRESS is applied **at most once per item** (mask it when `compression_count ≥ 1`), so a cache-friendly "compress once" behaviour is possible and repeated recompression cannot thrash the cache.

### 6.4 Cache cost model (Phase 2)

```python
@dataclass(frozen=True)
class PricingProfile:
    name: str
    input_per_mtok: float          # $ per 1M uncached input tokens
    output_per_mtok: float
    cache_read_multiplier: float   # cached input billed at input * m_read
    cache_write_multiplier: float  # newly cached tokens billed at input * m_write (1.0 for automatic)
    min_cacheable_tokens: int
    block_size: int                # cached length rounded down to a multiple of this
    ttl_seconds: float
    mode: Literal["automatic", "breakpoint"]
    source: str                    # URL
    retrieved: str                 # ISO date

@dataclass(frozen=True)
class CallCost:
    prompt_tokens: int; cached_tokens: int; written_tokens: int
    uncached_tokens: int; output_tokens: int
    input_cost: float; cache_read_cost: float; cache_write_cost: float
    output_cost: float; total_cost: float

class PrefixCacheSimulator:
    def __init__(self, profile: PricingProfile) -> None: ...
    def reset(self) -> None: ...                        # new episode = cold cache (default)
    def call(self, prompt_ids: Sequence[int], output_tokens: int, now: float) -> CallCost: ...
```

Semantics:

- `cached = max over live entries of LCP(prompt_ids, entry)`, rounded down to `block_size`, set to 0 if below `min_cacheable_tokens`. Entries expire `ttl_seconds` after their last use.
- `automatic` mode: after the call, the full prompt becomes a live entry; `written_tokens = prompt − cached`, billed at `m_write` (normally 1.0).
- `breakpoint` mode: the cache entry is the prompt up to the last configured breakpoint (default: end of the latest history item before the newest observation). Written tokens are billed at `m_write`.
- `uncached = prompt − cached − written` (never negative). `total = Σ terms`.
- Cache scope is per episode by default, with an option for warm cross-episode caching of the shared protected prefix. The choice is recorded in run metadata.
- `TokenCostMeter` has the same interface and returns `total_cost = prompt_tokens` (optionally `+ w_out·output_tokens`) for the token-reward variant.

### 6.5 Agent environment contract (Phase 3)

```python
class AgentEnv(Protocol):
    name: str
    def reset(self, task_id: str, seed: int) -> AgentObservation: ...   # task instruction + first observation
    def step(self, action_text: str) -> tuple[AgentObservation, float, bool, dict]: ...
    def task_ids(self, split: Literal["train", "valid", "test"]) -> Sequence[str]: ...
```

The agent loop (`envs/agent_loop.py`) owns the memory store, calls `memory_policy.decide(state)` before each LLM call, applies the actions, serializes, prices through the cost meter, calls the LLM and steps the environment. Terminal reasons: `success`, `env_done`, `step_limit`, `context_overflow`, `budget_violation`, `fatal_error`.

### 6.6 Controller MDP (Phase 4)

- **State/observation** (versioned schema `features.SCHEMA_VERSION`), per eligible item *i*: normalised age, normalised position, `log(1 + token_count)`, type one-hot, compressed flag, compression gain (tokens saved if compressed, 0 if masked), relevance to the task instruction (lexical/embedding similarity, inference-available), **suffix tokens after *i* / prompt tokens** (the re-billing cost of editing *i*). Global: turn/max_turns, prompt tokens/context limit, last cached fraction, pricing multipliers, plus Phase 7 budget features. All values must be finite and bounded, and the dimension fixed per schema.
- **Action:** `gate ∈ {0: NO_EDIT, 1: EDIT}`; if EDIT, `a_i ∈ {KEEP, COMPRESS, PRUNE}` per eligible item with masks from `valid_actions`. On NO_EDIT, every item is KEEP.
- **Log-probability:** `log π(gate) + 1[EDIT] · Σ_i log π(a_i)`.
- **Transition:** one MDP step per agent turn (controller decision, then one LLM call, then one env step).

### 6.7 Reward (Phase 4)

```text
r_t      = −λ · cost_t / Z                    (per LLM call)
r_T     += R_task                             (terminal: ALFWorld success ∈ {0,1}; WebShop score ∈ [0,1])
cost_t   = CallCost.total_cost  (cache variant)  |  prompt tokens (token variant)
Z        = mean full-history episode cost on a fixed calibration subset of TRAIN tasks (per env × profile × variant)
```

- Rewards for context overflow and budget violation: `R_task = 0` plus an optional configured penalty, logged as its own term.
- No reward for edits in themselves. Each term is logged separately per episode.
- λ is swept (for example {0.1, 0.3, 1, 3}) for both variants. Frontiers are compared on **real cost** whatever the training cost was.

### 6.8 Episode record (Phases 3–8)

One JSONL line per episode: `method`, `variant`, `knob`, `env`, `split`, `task_id`, `seed`, `llm`, `tokenizer`, `pricing_profile`, `success`, `score`, `turns`, `llm_calls`, `terminal_reason`, `prompt_tokens_total`, `cached_tokens_total`, `written_tokens_total`, `output_tokens_total`, cost per term and total, `edits`, `forced_edits`, `actions_{keep,compress,prune}`, `tokens_removed`, `edit_turns`, `latency_s`, `budget`, `budget_compliant`, `git_commit`, `config_hash`. Per-call details go in a sidecar file. Prompts are stored as hashes plus counts only.

## 7. Detailed implementation phases

## Phase 0: Repository setup ✅ DONE

Package, `uv` install, multi-stage Docker image, compose, CI (smoke, then ruff, then mypy, then pytest; Docker build, size budget, image smoke), README. The exit gate passed.

## Phase 1: Memory engine and token accounting ✅ DONE

Typed items, archive store, hash-only provenance, KEEP/COMPRESS/PRUNE with structured results and masks, shared serializer, tiktoken and test tokenizers with no silent fallback. The exit gate passed (`tests/test_memory_operations.py`, `tests/test_tokenizer_and_prompt.py`).

## Phase 2: Cache-aware cost model

### Work

1. `cost/pricing.py`: `PricingProfile`, YAML loader with validation (non-negative prices, multipliers ≥ 0, `block_size ≥ 1`, non-empty `source`/`retrieved`).
2. `configs/pricing/`: at least two contrasting profiles filled in from provider pricing pages, with source and date. Clearly labelled illustrative profiles are allowed for tests only.
3. `cost/cache_sim.py`: `PrefixCacheSimulator` (§6.4) on token IDs, with a simulated clock for TTL.
4. `cost/meter.py`: `CacheAwareCostMeter` and `TokenCostMeter` with a common `CostMeter` protocol, plus a per-episode accumulator.

### Required tests (`tests/test_cost_model.py`)

- Cold first call: `cached = 0`, all tokens written or uncached, exact $.
- Append-only second call: cached equals the previous prompt, floored to the block size.
- Editing item *i* in a 10-item history: cached length ends exactly at the first changed token. Early edits re-bill more than late edits (monotonic).
- Prefix below `min_cacheable_tokens` is not cached; block rounding is correct; TTL expiry gives a cold call.
- `breakpoint` vs `automatic` produce the documented different write and read accounting.
- Cost terms add up to the total; everything is finite; results are deterministic.
- `TokenCostMeter` ignores caching.
- A loader without `source` or `retrieved` is rejected.

### Exit gate

Hand-computed cost tables for a scripted five-call episode match exactly under both profiles. A worked example in the docstring shows that "prune a little every turn" can cost more than "never prune".

## Phase 3: Frozen-LLM agent harness

### Work

1. `llm/backend.py`: `LLMBackend` protocol (`generate(messages, **decoding) → LLMResponse` with text, output tokens and latency). Implementations: OpenAI-compatible HTTP client for a local vLLM server; `llm/mock.py` scripted/regex mock for tests.
2. `llm/hf_tokenizer.py`: HF tokenizer adapter with `encode`/`count` plus chat-template rendering, identity recorded (model name and revision).
3. `envs/agent_prompt.py`: `AgentPromptSerializer` (§6.2).
4. `envs/toy.py`: small deterministic tool environment (for example "look up item IDs, then combine them") where early observations are needed later and others are distractors, with variable-length observations. Runs in CI.
5. `envs/alfworld.py` (core) and `envs/webshop.py` (stretch, after core gates pass): adapters to `AgentEnv`, ReAct prompt with fixed few-shot examples, step limits from config. Setup scripts download assets into `data/raw/` (gitignored). Verify licenses, versions and split sizes, and record them.
6. `envs/agent_loop.py`: the loop of §4.1 with a `MemoryPolicy` hook (always-KEEP by default) and JSONL episode records (§6.8).
7. Optional extras in `pyproject.toml`: `[envs]`, `[llm]`. Smoke probes for them stay non-fatal.

### Required tests (`tests/test_agent_harness.py`)

- The mock LLM plus the toy env complete an episode deterministically. The record schema validates.
- The serializer puts protected content first; pruning an item removes it from the next prompt; protected content is never an item.
- Context overflow produces `terminal_reason = context_overflow` with no truncation.
- Cost is accumulated over every call; cache stats come from the simulator.
- Leakage: task gold data or expert plans never appear in prompts, items or metadata.

### Exit gate

- CI runs the toy env end-to-end with the mock LLM.
- **Competence gate:** the frozen LLM with full history reaches a non-trivial success rate on ALFWorld *validation* (and WebShop's, when the stretch work is attempted) (record the numbers; if below ~20%, change the model or prompt before Phase 6). Report the full-history cost per episode under each profile.

## Phase 4: Controller MDP and reward

### Work

1. `controller/features.py`: the §6.6 feature schema with `SCHEMA_VERSION` and documented scales.
2. `controller/env.py`: wraps the agent loop as an RL env yielding `(obs, masks)` per turn and taking `(gate, item_actions)`. It enforces masks, rejects invalid actions with a clear error, and terminates with an explicit reason when no action is valid.
3. `controller/reward.py`: §6.7 for both variants, the `Z` calibration script, and per-term logging.
4. COMPRESS at most once per item (§6.3).

### Required tests (`tests/test_controller_env.py`, `tests/test_no_information_leakage.py`)

- Fixed observation dimension, finite values, no simulator-only fields (checked by constructing tasks with labels and asserting absence).
- Masks: pruned or protected items are never candidates; COMPRESS is masked when it is unavailable or already applied.
- Reward accounting: Σ per-term logs = episode return; the terminal step does not bootstrap.
- Always-KEEP = full-history cost; aggressive PRUNE in the toy env loses a needed fact and fails, and that is measured.
- The cache-variant cost of "prune one item every turn" > "never prune" on a constructed toy case, while the token-variant cost is lower. This demonstrates H2 mechanically.

### Exit gate

1,000 random-policy toy episodes with no hangs, NaN/inf, invalid transitions, inconsistent token counts or state leakage between resets, plus manual inspection of 5 logged episodes.

## Phase 5: Baselines

### Work

`baselines/` implementing the `MemoryPolicy` hook:

| Baseline | Rule | Knob |
|---|---|---|
| Full history | never edit | — |
| Sliding window | keep the last *K* turns (pairs of thought/action and observation) | *K* |
| Observation masking (**must-beat**) | replace observations older than *M* turns with a fixed placeholder; keep thoughts/actions | *M* |
| LLMLingua | compress observations older than *M* once with LLMLingua-2 at rate *r* (counts compressor latency) | *M*, *r* |
| Threshold compaction | no edits until the prompt exceeds *T* tokens, then one compaction (compress, then prune the oldest observations down to *T/2*) | *T* |
| CADOC-style rule | analytical (EOQ-style) rule that includes cache-rebuild cost and decides when to batch edits; re-implemented from the paper, formula checked against the source, deviations documented | CADOC's parameters |
| Random valid | random gate and actions under the masks | p(edit) |

Plus a common runner `evaluation/run_eval.py --method … --split valid` writing §6.8 records.

### Required tests (`tests/test_baselines.py`)

Window boundary behaviour; masking leaves protected content and actions untouched; threshold compaction triggers exactly at *T*; every baseline respects masks; records validate. LLMLingua tests are skipped only when the optional extra is not installed, with an explicit skip reason.

### Exit gate

Every baseline × knob runs on a fixed ALFWorld validation subset (and WebShop's, if stretch) and produces valid records and aggregate metrics. The CADOC-style rule's re-implementation is documented, with its source equations cited.

## Phase 6: PPO controller

### Model

- Item encoder: MLP (2 × 128, ReLU) over per-item features, with global features concatenated.
- Context: mean+max pooling over item embeddings (a 2-layer transformer encoder is an ablation).
- Heads: gate logits from pooled context; per-item 3-way logits with masked softmax (invalid logits set to −∞, so probability is exactly 0); value from pooled context.

### Algorithm

- PPO-clip (ε = 0.2), GAE (γ = 1.0 episodic, λ_GAE = 0.95), entropy bonus on gate and item heads, value loss coefficient 0.5, Adam lr 3e-4, grad-norm clip 0.5, 4 epochs, minibatches over turns. These are starting values for pipeline validation, not tuned results.
- Rollouts: N parallel episodes per update (vLLM batches LLM calls), stored as `(obs, masks, gate, item_actions, logp, value, reward, done)`.
- Variants: `cost=cache` and `cost=token`, each over the λ sweep, ≥ 3 seeds.
- Validation every K updates, deterministic (argmax) policy, on `valid` tasks only. Checkpoint selection on validation success at the target cost region.
- Checkpoints: weights, optimiser state, config, `SCHEMA_VERSION`, LLM and tokenizer identity, pricing profile, seed, step. Incompatible loads fail with an actionable message.
- Fail explicitly on non-finite loss, weights or advantages.

### Required tests (`tests/test_ppo.py`)

Shapes and finiteness; masked actions are never sampled (exploration included); log-prob of a joint action = gate + Σ items; the GAE terminal case; checkpoint save → load round-trip; the schema mismatch error; a seeded short run is reproducible on CPU.

### Exit gate

On the toy env, PPO beats random and always-KEEP on the reward with a clear margin over ≥ 3 seeds. A small ALFWorld subset run trains end-to-end, writes metrics and checkpoints, and evaluates from a checkpoint without training code.

## Phase 7: Budget-constrained extension (stretch)

Start only after every core exit gate (Phases 2–6) passes.

### Work

1. `controller/lagrangian.py`: CMDP with soft dollar budget B. Reward for the policy: `R_task − λ_B · cost/Z`. Dual update after each PPO iteration: `λ_B ← clip(λ_B + η_λ (Ĉ_B − B), 0, λ_max)`, with one multiplier per budget bin.
2. Hard per-call token cap `B_tok`: if the next prompt would exceed `B_tok`, the gate is forced to EDIT (`forced_edits += 1`). If the post-edit prompt still exceeds it, the episode ends with `terminal_reason = budget_violation`.
3. Budget conditioning: sample B (or B_tok) per episode from a configured range; add `B/Z`, the remaining-budget fraction and the cap headroom to the global features (bump `SCHEMA_VERSION`).
4. Comparisons: one budget-conditioned policy vs policies trained per budget vs baselines tuned per budget, all at three budgets (for example 25/50/75% of full-history cost, and caps at 25/50/75% of the median full-history final prompt length).

### Required tests (`tests/test_lagrangian.py`)

λ rises when cost is over budget and falls (to a floor of 0) when it is under; forced-edit logic; a budget violation is recorded rather than hidden; budget features stay within bounds.

### Exit gate

Budget compliance ≥ 95% at the three validation budgets, or the shortfall is reported with analysis. H3 behaviour plots (edit frequency and size under a hard cap) are produced for both variants.

## Phase 8: Final evaluation

### Protocol

1. Freeze configs, checkpoints and knob grids from validation. Record them before touching test data.
2. Run every method × knob × (budget) × pricing profile × seed on the **same** test tasks (core: ALFWorld `valid_unseen`; stretch: WebShop test split as verified in Phase 3).
3. Baselines don't depend on prices, so their stored per-call token traces can be re-priced under every profile without re-running. A PPO policy trained under profile A and re-priced under profile B is the **transfer** condition and is labelled as such; the main result for profile B uses policies trained under B.
4. Aggregate: §8 metrics of `prd.md`, paired bootstrap CIs over tasks, per-seed variation.
5. Plots: success vs real cost (Pareto, main), success vs tokens, cache-hit rate, edits per episode and edit timing, budget compliance.
6. Cost-model validation: compare simulated cached tokens with vLLM prefix-cache hit metrics. The optional paid-API check (a small number of episodes, provider-reported cached-token usage vs simulator) runs only with explicit user approval and a recorded spending cap.

### Positive-claim criteria (fix them on validation before seeing test)

- **H1 supported:** at the success level of full history (within 3 pp), cache-aware PPO's cost is lower than every baseline's frontier at that success, explicitly including observation masking and the CADOC-style rule, with a paired CI excluding 0.
- **H2 supported:** on at least one environment × profile, the token-reward controller has fewer input tokens and higher real cost than cache-aware PPO or full history, with a CI.
- **H3 supported:** the cache-aware controller has significantly fewer edits per episode and more tokens removed per edit than the token-reward controller. Its edit timing is also reported next to the CADOC-style rule's.
- **H4 supported (stretch):** Phase 7 gate on test.

Otherwise report the measured outcome and narrow the conclusion. Do not change thresholds after seeing test results.

## Phase 9: Analysis, ablations and report

- Ablations (one factor at a time): no gate (flat per-item); no suffix-tokens feature; no COMPRESS; transformer context encoder; pricing-profile transfer (train A, test B); a different frozen LLM; cold vs warm cross-episode cache.
- Failure analysis: overflow, budget violation, step limit, wrong action after a relevant observation was pruned.
- Related-work and novelty re-check on arXiv (CADOC, CAPC, TRACER, LRE, TokenPilot, AKV, observation masking, "Beyond Token Savings", PCRL, and any newer work): compare actions, objective, reward, training, baselines and evaluation. Verify every bibliographic detail against the source.
- Deliverables: code and tests, configs, logs, checkpoint metadata, per-episode records, one-command figure regeneration, the report, slides and a demo episode visualisation.

## 8. Cross-cutting edge-case checklist

**Memory and prompts:** empty history; one-item history; duplicate texts with distinct IDs; very long single observations; Unicode; a compressor returning empty, unchanged or longer text; pruning an item that a later step depends on; chat-template tokens counted in cost; tokenizer mismatch between cost model and LLM (must fail).

**Cache model:** first call; identical consecutive prompts; an edit at the first token after the protected prefix; a prefix below the minimum; block boundary ±1; TTL expiry mid-episode; a breakpoint placed after an edited item; the output tokens of call *k* becoming part of the prompt of call *k+1* (they are cacheable only if the provider caches them; follow the profile).

**Environment and learning:** no eligible items (first turn); all items pruned; the step limit reached; LLM produces an invalid action; LLM server error (retry policy, recorded); reset clears the store, cache simulator and labels; masks all-invalid apart from the gate; NaN in features or advantages; reward incentives to prune everything or keep everything; leakage through metadata, IDs, filenames, caches or logs.

**Evaluation:** zero successes (cost per success = ∞); different prompt layouts across methods (forbidden); test reuse during selection (forbidden); missing provider usage (mark unavailable); LLM nondeterminism at temperature 0 (record and report).

## 9. Project-wide success criteria

The prototype is **implementation-complete** when:

1. Memory, cost model, harness and controller satisfy §4 invariants, and focused tests pass in CI (no GPU needed).
2. Episodes are finite, reproducible and leakage-tested on the toy env and both benchmarks.
3. All baselines and both PPO variants use the same harness, records and runner.
4. Training saves compatible checkpoints and validation metrics.
5. Final evaluation runs every method on the same test tasks, budgets, profiles and seeds.
6. Figures and tables regenerate from per-episode records with one command.
7. Failures, limitations, versions and unmeasured costs are disclosed.

The research hypotheses count as **supported** only if the Phase 8 criteria are met on held-out data with uncertainty reported.

## 10. Milestone gates

| Milestone | Deliverable | Must pass before proceeding |
|---|---|---|
| M0 ✅ | Runnable package, Docker, CI | Smoke + CI green |
| M1 ✅ | Memory engine and token accounting | Phase 1 tests |
| M2 | Cache cost model | Hand-computed cost cases |
| M3 | Agent harness and competence gate | Toy env in CI; non-trivial full-history success |
| M4 | Controller MDP and reward | 1,000 random episodes; leakage tests |
| M5 | Baselines | All baselines (incl. CADOC-style) on ALFWorld validation |
| M6 | PPO (cache and token variants) | Beats random/KEEP on toy; ALFWorld subset end-to-end |
| M7 (stretch) | Budget-constrained extension | ≥ 95% compliance or a documented gap |
| M8 | Final evaluation | Same-task, multi-seed, multi-profile results |
| M9 | Report | Claims match measured evidence |

Phases 2 and 3 can proceed in parallel, as can 4 and 5. Core is M0–M6 and M8 on ALFWorld; WebShop and Phase 7 are stretch, started only once every core gate passes.

## 11. Developer workflow and commands

```bash
python -m memory_rl.smoke
ruff check .
mypy
python -m pytest
```

Planned entry points (they don't exist until their phase lands):

```bash
python -m memory_rl.cost.cache_sim --demo                                      # Phase 2
python -m memory_rl.envs.agent_loop --env toy --llm mock                       # Phase 3
python -m memory_rl.evaluation.run_eval --config configs/baselines.yaml        # Phase 5
python -m memory_rl.controller.train --config configs/ppo.yaml                 # Phase 6
python -m memory_rl.controller.train --config configs/lagrangian.yaml          # Phase 7
python -m memory_rl.evaluation.run_eval --config configs/evaluation.yaml       # Phase 8
python -m memory_rl.evaluation.plots --results results/<run>                   # Phase 8
```

Commands must: return nonzero on invalid config or failure; log config, seeds, versions and output paths; refuse to overwrite results without an explicit flag; support a `--smoke` mode for CI; read API keys only from environment variables.

## 12. Explicit non-goals

- Training or fine-tuning the LLM.
- A production memory service or UI.
- Retrieval in the controller's action space.
- Paid-API usage during training, or without explicit approval.
- Claiming lower token count alone as success.
- Hardcoded prices or scores; treating starting hyperparameters as tuned optima.
