# Implementation Plan: Budget-Constrained RL for Task-Critical Memory

## 1. Instructions for the AI coding agent

Read this document before changing code. Treat it as the implementation contract for the first research prototype.

Before implementing:

1. Inspect the actual workspace, existing code, tests, project metadata, and Git status. This plan was written from a workspace snapshot that contained `plan.md`; the implementation agent must verify the current state rather than assume it is unchanged.
2. Preserve existing project conventions and user changes. Do not overwrite or remove unrelated work.
3. If the repository already has an architecture, adapt this plan to it and document meaningful deviations. Do not create duplicate modules just to match the example tree below.
4. Implement one phase at a time. Run its focused tests and satisfy its exit criteria before moving on.
5. Do not silently skip a failing test, fabricate benchmark results, or describe illustrative values as measured results.
6. Keep secrets, private benchmark data, credentials, and local machine paths out of source control and reports.
7. At completion, report what was implemented, commands run, test results, known limitations, and the next incomplete milestone.

When a requirement is underspecified, use the defaults in this plan. Ask the user only when a decision materially changes the research question, requires a paid/external service, or cannot be made safely from the repository context.

## 2. Project definition

### 2.1 Working title

**Budget-Constrained Reinforcement Learning for Task-Critical Memory Retention in Multi-Step AI Agents**

### 2.2 What we are building

Build a reproducible research prototype that manages a history of messages, facts, decisions, and tool results under a constrained input-context budget. A small DQN controller processes memory units and chooses one operation for each eligible unit:

- **KEEP** the current representation.
- **COMPRESS** it using a separate, fixed compression method when a shorter valid representation is available.
- **PRUNE** it from the active context while retaining its archived record for experiment analysis.

A separate, fixed answerer receives the resulting active context and the current task. It is not the memory controller. The controller must not receive answer keys, reference answers, or synthetic criticality labels as inference-time features.

The project is a research system, not a production conversational assistant. Its central question is:

> At a matched active-context token budget, can a learned memory controller preserve downstream task success better than strong non-RL memory-management baselines?

The initial controlled environment uses synthetic tasks with known answers and required facts. Once that environment, baselines, and evaluation are reliable, the same pipeline can be adapted to a public conversational-memory benchmark.

### 2.3 Research claim boundaries

- The intended claim concerns **memory management**, not general language-model capability.
- Claims about multi-step tool-using agents require explicit tasks where earlier tool results or decisions are needed later. Conversational question-answering results alone do not establish that broader claim.
- DQN superiority is a hypothesis, not an implementation requirement. A reliable, reproducible negative or inconclusive result is a valid research outcome.
- Do not claim novelty until relevant work (including Mem-α and Memory-R1) and benchmark protocols have been reviewed.

## 3. Scope and fixed initial decisions

| Decision | Initial implementation |
|---|---|
| Language/runtime | Python 3.11 unless the existing repository sets a supported version |
| Learning algorithm | DQN with replay buffer and target network |
| Decision unit | One memory item at a time, in a finite, deterministic episode order |
| Action set | KEEP, COMPRESS, PRUNE; invalid actions are masked |
| Compression | Fixed deterministic compressor initially; it is not learned by DQN |
| Training data | Generated synthetic episodes with known ground truth |
| Answering | Deterministic exact-match answerer for synthetic tasks; one fixed answer model per external evaluation |
| Initial baselines | Full history, sliding window, relevance/recency heuristic, fixed summarization |
| Initial benchmark candidate | LoCoMo, subject to verifying its current release, license, split, and scoring rules |
| Main metric | Task success at matched actual prompt-token usage |
| Reproducibility | Configured seeds, fixed splits, versioned tokenizer/model, per-example results |

Do not add retrieval to the initial DQN action space. A retrieval-based memory strategy may be added later as a separate baseline, not mixed into the first experiment.

## 4. System behavior and invariants

### 4.1 End-to-end flow

```text
conversation/task history
        ↓
memory items + immutable archive
        ↓
token accounting and observable features
        ↓
policy (baseline or DQN) chooses operations
        ↓
active memory + current task
        ↓
fixed answerer
        ↓
answer scoring + efficiency metrics + experiment record
```

### 4.2 Invariants every implementation must preserve

1. A pruned item is absent from the active prompt. Keeping an archive for diagnostics must not make it available to the answerer.
2. System instructions and the current task/question are outside the controller's action space and cannot be pruned.
3. Ground-truth labels may be used by the synthetic simulator for rewards and diagnostics only. They must not enter policy observations, feature computation, or answer prompts.
4. All methods in a comparison use the same task examples, tokenizer, prompt template, answerer, scoring rules, and budget definition.
5. Count tokens for the complete serialized answerer input, including fixed instructions, formatting, and the current question. Also report memory-only tokens separately.
6. Never silently truncate a prompt and label it as full history or as a successful budget-compliant example.
7. A compression operation is valid only if the resulting text is shorter under the configured tokenizer. Compression does not imply semantic preservation; answer success and critical fact retention must be measured independently.
8. Every episode has a finite decision horizon and a defined terminal state.
9. Every recorded metric is derived from actual run data. Missing cost or latency measurements are marked unavailable, not estimated as measured values.
10. Randomness is controlled by explicit seeds for task generation, policy exploration, replay sampling, and evaluation where applicable.

## 5. Proposed repository layout

Adapt to existing conventions before creating files. If starting from an empty project, use a Python `src/` layout:

```text
.
├── README.md
├── pyproject.toml
├── configs/
│   ├── environment.yaml
│   ├── dqn.yaml
│   └── evaluation.yaml
├── data/
│   ├── README.md
│   ├── raw/                 # ignored; never commit restricted data
│   ├── processed/
│   └── synthetic/
├── src/
│   └── memory_rl/
│       ├── memory/
│       │   ├── models.py
│       │   ├── store.py
│       │   ├── tokenizer.py
│       │   └── operations.py
│       ├── environment/
│       │   ├── tasks.py
│       │   ├── features.py
│       │   ├── reward.py
│       │   └── memory_env.py
│       ├── controller/
│       │   ├── dqn.py
│       │   ├── replay_buffer.py
│       │   └── train.py
│       ├── answering/
│       │   ├── interface.py
│       │   └── synthetic.py
│       ├── baselines/
│       │   ├── full_history.py
│       │   ├── sliding_window.py
│       │   ├── heuristic.py
│       │   └── summarization.py
│       └── evaluation/
│           ├── metrics.py
│           ├── run_eval.py
│           └── report.py
├── tests/
│   ├── test_memory_operations.py
│   ├── test_tokenizer_and_prompt.py
│   ├── test_environment.py
│   ├── test_no_information_leakage.py
│   ├── test_baselines.py
│   └── test_dqn.py
├── scripts/
├── checkpoints/             # generated; normally ignored
└── results/                 # generated; normally ignored or selectively published
```

Keep modules small and responsibilities separate. Avoid importing benchmark-specific code into the core memory engine. Add dependencies only when the selected implementation needs them; do not install every package mentioned in older drafts preemptively.

## 6. Data and API contracts

### 6.1 Memory item

Use a typed data model (a dataclass is sufficient unless the project already uses another validation library) with at least:

| Field | Meaning |
|---|---|
| `memory_id` | Stable unique ID within an episode |
| `text` | Current active representation |
| `original_text` | Immutable original text for analysis; never shown if item is pruned |
| `token_count` | Count for current representation, recalculated after edits |
| `turn_index` | Position in the source history |
| `memory_type` | User message, assistant message, tool result, fact, decision, etc. |
| `status` | Active, compressed, or pruned |
| `compression_count` | Number of successful compressions |
| `dependencies` | Optional IDs of linked items; do not infer or expose labels unavailable at inference |
| `metadata` | Non-sensitive, inference-available attributes only |

Keep simulator-only labels such as `is_critical`, `importance`, `required_fact_ids`, and `reference_answer` in a separate task/label object. Enforce this separation in constructors, feature functions, and tests.

### 6.2 Tokenizer and prompt builder

Provide one tokenizer interface used by every method. It must:

- Be selected to match the answer model where possible.
- Have an explicit stable name/version recorded in experiment metadata.
- Count empty strings, Unicode text, punctuation, numbers, and formatted prompts consistently.
- Fail explicitly if its configured tokenizer cannot be loaded; do not silently fall back to character or word counts.
- Support a deterministic test tokenizer for isolated unit tests if a model tokenizer is unavailable.

Build prompts through a single shared serializer. Calculate and expose:

- `memory_tokens`: tokens of the serialized active memory only.
- `fixed_prompt_tokens`: instructions, separators, and current task/question.
- `total_prompt_tokens`: token count of the exact final serialized input.
- `answer_tokens`: generated output tokens, when available.

If a provider reports usage, record it alongside local counts and document discrepancies. Never mix token definitions in a single budget-compliance metric.

### 6.3 Memory operations

Implement operations in the memory engine, not in the policy:

- `KEEP`: item remains active and unchanged.
- `COMPRESS`: call the configured fixed compressor, validate output, update text/count/status/history only on success. If output is empty, unchanged, longer, or invalid, report compression as unavailable/failed and preserve the original. The policy's action mask should normally exclude unavailable compression.
- `PRUNE`: mark item pruned and omit it from active prompt construction. Preserve original record and log old token count as potential savings.

Each operation should return a structured result describing whether it changed the item, token delta, reason, and resulting status. Do not hide failure by returning a success-shaped result.

### 6.4 Task and episode contracts

A synthetic task includes:

- Ordered history items.
- Current question/task.
- Reference answer and answer-checking rule.
- Required fact IDs, kept outside controller observations.
- Task family and difficulty metadata.
- Episode seed or stable identifier.

The environment should expose a Gymnasium-like interface even if it does not depend on Gymnasium:

```python
observation = env.reset(task, seed=seed)
observation, reward, terminated, info = env.step(action)
```

Return explicit terminal status and diagnostic `info`. Reject invalid actions with a clear exception or repository-standard error path; do not silently reinterpret them.

## 7. Detailed implementation phases

## Phase 0 — Repository reconnaissance and executable setup

### Work

1. Inspect current source, project metadata, supported Python versions, test runner, formatter/linter, Git status, and ignore rules.
2. Record the actual architecture and reuse existing conventions.
3. If the repository has no packaging setup, create a minimal `pyproject.toml` and an installable `src/` package. Select compatible versions only as needed; pin the environment after a working baseline is established.
4. Add ignore rules for virtual environments, secrets, raw/restricted data, checkpoints, caches, and generated results. Do not ignore source configs or small reproducibility metadata.
5. Add a README with the research question, installation, planned commands, and a clear statement that the DQN result is not yet established.

### Exit criteria

- The package imports in the selected environment.
- A smoke command runs and exits nonzero on failure.
- No credentials or restricted data are required for the synthetic path.
- Existing tests and project conventions remain intact.

## Phase 1 — Memory engine and trustworthy token accounting

### Work

1. Implement the typed memory model and archive/active-store behavior.
2. Implement the tokenizer interface, shared prompt serializer, and exact token metrics.
3. Implement KEEP, COMPRESS, and PRUNE with structured operation results.
4. Keep immutable task instructions and current question outside action candidates.
5. Add provenance logging sufficient to reproduce each transformation without exposing pruned content to the answerer.

### Required tests

- KEEP preserves text, ID, status, and token count.
- Successful compression changes active text, preserves original text, increments the compression count, and recomputes tokens.
- Compression that is not shorter or is invalid leaves the original active text untouched and reports a non-success.
- PRUNE removes the item from serialized active context but retains its archive record.
- Repeated prune/compress attempts are masked or rejected according to the operation contract.
- Active token total equals the sum of active item counts.
- Full prompt count includes system instructions, separators, memory, and current task.
- System instructions and current task are never candidates for mutation.
- Token counting is deterministic for a fixed tokenizer/version, including Unicode and empty input.

### Exit criteria

Load a sample history, execute each valid operation, serialize the prompt, and verify exact expected counts. Do not begin RL training before this gate passes.

## Phase 2 — Synthetic tasks and leakage-safe environment

### Work

1. Generate varied examples for:
   - Exact fact recall (names, dates, amounts, identifiers).
   - Multiple simultaneous constraints.
   - Multi-step dependencies where a prior result matters later.
   - Distractor-heavy histories with useful facts at varied positions.
2. Vary values, order, history length, distractors, and templates. Use split-by-template/scenario so test examples are not near-duplicates of training examples.
3. Keep reference answers and required-fact labels in simulator-only fields.
4. Use one memory item per message for the first version. Preserve conversation order and stable IDs.
5. Define a finite one-pass decision sequence. In each step the policy acts on the current eligible item; advancing to the next item is automatic. An action cannot be repeated indefinitely on the same item.
6. Include current task/question in observations only to the degree it is available at runtime. Do not include the answer, hidden required IDs, or future labels.
7. Define observable features with documented scales, for example:
   - age/turn position, normalized;
   - current item token count and relative cost;
   - item type encoding;
   - compression count;
   - remaining and total active memory tokens relative to budget;
   - episode progress;
   - an inference-available relevance estimate from current question and item text;
   - dependency metadata only where it is genuinely observable.
8. Document each feature and assert the observation vector has fixed dimension, finite values, and no simulator-only fields.

### Action validity

- KEEP and PRUNE are valid for active eligible items.
- COMPRESS is valid only when the fixed compressor can produce a shorter representation.
- Pruned items and protected prompt content are never candidates.
- Return a validity mask; both exploratory and greedy selection must respect it.
- If no action is valid, terminate with a clear reason rather than looping.

### Budget and episode termination

- Define the target budget explicitly as a fraction of original **memory** tokens for controlled experiments.
- Track full prompt tokens separately; report compliance against the exact budget definition being used.
- Do not enforce budget by silently deleting or truncating information behind the policy. A DQN episode may end over budget, incur the configured penalty, and be counted as a violation.
- Enforce the model context limit as a hard answerer safety boundary. If the final prompt does not fit, mark the example as a context/budget failure and do not silently truncate.
- Episode terminates after all candidates have one decision, when an explicit stop condition is reached, or on a documented fatal environment error.

### Exit criteria

- Random valid policies can execute at least 1,000 short generated episodes without hangs, invalid transitions, NaN/inf observations or rewards, inconsistent token counts, or state leakage between resets.
- A deliberately constructed task demonstrates that a critical fact can be lost and measured as lost.
- Tests prove hidden labels do not appear in observations, feature inputs, or answer prompts.

## Phase 3 — Reward design and environment verification

### Initial reward

Use an explicit, configurable reward with separately logged components:

```text
episode return =
    task_success_reward
  - token_usage_cost
  - required_information_loss_penalty
  - budget_violation_penalty
  - measured_or_configured_operation_cost
```

The precise numeric weights belong in config and must be recorded per run. Start with a large positive task-success signal and a substantial penalty for losing required information or exceeding budget. Avoid paying the agent merely for pruning; token savings are useful only when considered with task outcome.

For the controlled synthetic environment, the simulator may calculate required-information loss after an operation for training rewards and diagnostics. This label must remain unavailable to the policy at inference. Avoid double-counting task failure and critical-loss penalties without explicitly documenting why both are retained.

Use a potential-based or otherwise clearly justified per-step shaping scheme if needed for learning. Keep terminal task outcome authoritative and log each reward term separately so reward hacking can be diagnosed. Do not tune reward weights on the final test set.

### Verification cases

- Always KEEP retains all available history and normally uses more tokens.
- Aggressive PRUNE saves tokens and fails a task when its only required fact is removed.
- A compression that drops an exact number/date can reduce tokens but must fail the exact task; it must not receive an unearned successful-compression reward.
- A feasible task can meet budget without forced hidden repairs.
- Unreachable budgets are reported as violations, not disguised as success.
- All rewards are finite; terminal transitions have no future-value bootstrap.

### Exit criteria

Run at least 1,000 random episodes, verify reward component accounting and termination, and inspect examples manually before implementing DQN training.

## Phase 4 — Baselines and standardized evaluation interface

All policies must consume equivalent task inputs and return a common result structure. At minimum include method, task ID, seed, memory tokens, total prompt tokens, target budget, compliance, operation counts, final answer, correctness, critical information retained/lost (diagnostic only), latency, and error/skip reason.

### Required baselines

1. **Unconstrained full history:** retain all history and report if it exceeds target budget/context. Do not label it budget-feasible when it is not.
2. **Budget-feasible full history:** report separately; applicable only when full history fits.
3. **Sliding window:** add most recent items in reverse chronological order while respecting the budget, then restore chronological order in prompt serialization.
4. **Heuristic pruning:** rank by the same inference-available relevance/recency/cost information exposed to DQN, retain within budget, and record dropped items.
5. **Fixed summarization:** summarize all history or chunks with one fixed compressor/model; count summaries and any retained originals. Do not use ground-truth labels.

Retrieval is a later optional baseline with its own explicitly documented storage/retrieval budget and should not be presented as directly equivalent without accounting for retrieval costs.

### Fairness rules

- Same per-example target and fixed prompt overhead for each method.
- Same tokenizer, serializer, answerer/model version, decoding settings, answer limit, and scorer.
- Same evaluation examples and final test split.
- Baseline hyperparameters tuned only on validation data.
- Include compression/model-call time and cost when applicable.

### Exit criteria

Every required baseline runs over a fixed synthetic validation set and produces schema-valid per-example records and aggregate metrics. Unit tests verify sorting, budget boundary behavior, and protected prompt fields.

## Phase 5 — DQN implementation and training

### Model and transitions

- Feed a fixed-length normalized feature vector to a small feed-forward network, initially two hidden layers of 128 ReLU units and three Q-values.
- Store `(state, action, reward, next_state, terminated, next_action_mask)` transitions; retaining the next mask prevents bootstrapping from invalid actions.
- Use a replay buffer, minibatch sampling, a target network, Huber loss, Adam, terminal masking, and optional gradient clipping.
- Select actions epsilon-greedily only among valid actions.
- Decay epsilon on a configured, recorded step/episode schedule with a nonzero floor.
- Seed model initialization, exploration, replay sampling, task generation, and evaluation independently but reproducibly.
- Save checkpoints with model/optimizer state as appropriate, config, feature schema/version, tokenizer identity, global step, and seed. Reject incompatible checkpoint feature schemas clearly.

### Training loop responsibilities

1. Load and validate configuration.
2. Generate/sample a training task and reset the environment with a reproducible seed.
3. Select a valid action; execute it; record transition and reward components.
4. Train after the configured warmup and batch-size conditions are met.
5. Update target network on the configured schedule.
6. Run periodic validation without exploration and without training updates.
7. Select checkpoints on validation metrics only; never inspect final test results during training.
8. Save episode-level metrics, action distribution, budget compliance, task outcome, critical-loss diagnostics, and training parameters.
9. Fail explicitly on non-finite loss/weights or malformed transitions.

### Initial debug configuration

Use small, editable config values such as 1,000 training episodes, at most 30 items per episode, an action horizon no longer than the number of eligible items, a 50% memory-token budget, 128x128 hidden layers, learning rate 0.0003, gamma 0.99, batch size 64, replay capacity 50,000, epsilon 1.0 to 0.05, and target update every 500 training steps. These are starting values for pipeline validation, not scientific recommendations or evidence of adequate training.

### DQN tests

- Q-network input/output shapes and finite outputs.
- Replay buffer capacity, sampling, and deterministic seeded behavior.
- Epsilon-greedy never selects masked actions, including exploration.
- Target computation handles terminal transitions and invalid next actions.
- A minimal training run saves and reloads a compatible checkpoint.
- Incompatible feature schema/configuration fails with an actionable error.

### Exit criteria

A small run completes end-to-end, writes reproducible metrics/checkpoints, and can be evaluated without training code or access to simulator-only labels.

## Phase 6 — Learning sanity checks and controlled test

Run and compare:

- Random valid policy.
- Always KEEP control.
- Aggressive PRUNE control.
- Trained DQN.
- Required non-RL baselines.

Use separate train, validation, and test sets. Split by template/scenario (not just random rows) to measure generalization. Validate on new values, longer histories, varied critical-fact positions, and new distractor patterns.

Run at least five independent training seeds for the first reported DQN result when resources permit. Report per-seed results and variation; do not hide unstable runs. If compute limits prevent five, report the exact number and treat conclusions as preliminary.

Inspect:

- Reward curve and each reward component.
- Validation task success and prompt-token use.
- Budget violation rate and forced/invalid-action counts.
- KEEP/COMPRESS/PRUNE distribution and policy collapse.
- Critical fact loss and compression-induced answer errors.
- Accuracy with full history versus managed history.
- Failure attribution: memory loss, bad compression, answerer failure, scoring error, or environment bug.

Never use final test performance to choose a checkpoint, features, reward weights, or hyperparameters.

## Phase 7 — Final evaluation protocol

### Metrics (report mean and suitable uncertainty)

| Metric | Definition |
|---|---|
| Task success | Correct examples / eligible examples; state scorer and excluded/error examples |
| Memory-token reduction | `1 - managed_memory_tokens / original_memory_tokens` |
| Prompt-token reduction | Same calculation using complete serialized input tokens |
| Budget compliance | Fraction of examples whose measured prompt meets the stated budget |
| Critical information retention | Required facts still answerable/present; diagnostic for labeled synthetic data |
| Cost per successful task | Total measured inference/compression cost divided by successful tasks |
| End-to-end latency | Controller + compression + answerer time, with measurement boundary stated |
| Action distribution | Counts/fractions by action and failed/unavailable compression |

Do not treat critical-item retention as a replacement for end-task accuracy. Report zero-success cost as undefined/infinite with an explicit convention, not as zero.

### Budget sweep and statistical comparison

- Evaluate at minimum three target memory budgets, for example 25%, 50%, and 75% of original memory tokens, plus unconstrained full history as a reference.
- Run every method on the same test examples at each budget.
- Plot task success against **actual total prompt tokens**, and show target budget/compliance separately.
- Use paired comparisons over shared examples. Report confidence intervals; where feasible, bootstrap over examples and separately show variation across training seeds.
- Report all results, including failures and budget misses. Do not claim a win based only on lower token count.

### Evidence needed for a positive practical claim

Pre-register/select the claim threshold on validation data before opening final test results. A reasonable initial interpretation is:

- “Quality preserved”: DQN's task success is no more than 3 percentage points below unconstrained full history at the evaluated budget, with uncertainty reported.
- “Improves the trade-off”: at matched actual prompt-token usage, DQN exceeds the strongest non-RL baseline on task success with a paired confidence interval supporting a positive difference; or it uses fewer tokens at statistically indistinguishable success.
- If these conditions are not met, report the measured result and narrow the conclusion. Do not alter thresholds after seeing test outcomes.

These thresholds are proposed interpretation criteria, not guaranteed model performance.

## Phase 8 — External benchmark transfer

Proceed only after the synthetic evaluation passes its gates.

1. Verify the current LoCoMo release, dataset license/terms, splits, scoring, dependencies, and any restrictions. Record the exact version/source and checksums when permitted.
2. Keep benchmark raw files outside source control unless the license explicitly permits redistribution.
3. Write an adapter from benchmark records to the standard task format; preserve turn ordering and metadata required by official scoring.
4. Use one memory unit per message initially unless the benchmark structure requires another granularity; document any transformation.
5. Do not expose reference answers or annotations to the controller, compressor, or answerer.
6. Use the official evaluation/scorer when possible and validate a sample manually.
7. Initially evaluate zero-shot transfer from synthetic training. If any benchmark-specific training/fine-tuning is later done, report it as a separate condition with allowed splits.
8. Include the exact same prompt/token budget and answerer controls across methods.
9. Report that conversational QA results do not on their own prove general multi-step-agent performance.

MemoryAgentBench and LongMemEval are optional follow-on candidates. Assess fit and protocol before adding them; more benchmarks do not compensate for an unreliable primary evaluation.

## Phase 9 — Research analysis and final deliverables

### Related work and ablations

Review Mem-α, Memory-R1, and the chosen benchmark papers/repositories. Compare problem framing, actions, reward, budget, training regime, baselines, and evaluation before making novelty claims.

After the main result is reliable, run controlled ablations where resources allow:

- Remove COMPRESS (KEEP/PRUNE only).
- Remove critical-loss reward term.
- Remove relevance or recency feature one at a time.
- Change budget.
- Change compression method.
- Change training data/episode count.

Hold other settings fixed as much as possible and report the exact changed factor.

### Final artifacts

1. Source code and tests for memory engine, environment, policies, DQN, and evaluation.
2. Configs and reproducibility instructions.
3. Training logs, checkpoint metadata, and learning curves.
4. Per-example predictions, token counts, success labels, latency/cost where measured.
5. Aggregate tables and task-success-versus-prompt-token plot.
6. Ablation and failure analyses.
7. Research report covering question, prior work, methods, baselines, protocol, results, limitations, and conclusion.
8. A clear statement of dataset/model versions and any restrictions.

## 8. Cross-cutting edge-case checklist

The implementation and tests must explicitly handle:

### Memory and operations

- Empty history, empty item text, one-item history, duplicate text with distinct IDs.
- Unicode, long numeric identifiers, dates, negative/decimal values, punctuation, and exact constraints.
- A compressor returning empty, unchanged, longer, malformed, or semantically lossy output.
- Repeated compression and repeated prune attempts.
- Multiple items whose combined tokens exceed budget although each item fits individually.
- A critical fact repeated in multiple items: losing one copy is not necessarily losing the fact.
- Dependencies between facts/intermediate tool results; do not claim dependency preservation unless tested.
- Stable archive versus active prompt separation.

### Budgets and prompts

- Fixed prompt alone exceeds budget/context limit.
- Exact equality at token budget and one-token overage.
- Tokenizer/version mismatch between controller, prompt builder, and answerer.
- Prompt formatting adds tokens beyond the sum of item tokens.
- Full history does not fit and must be reported as over budget.
- No silent truncation, hidden fallback tokenizer, or excluded failed examples.

### Environment and learning

- No valid compression action; no remaining eligible actions; already-satisfied budget.
- Unreachable budget and all-pruned state.
- Reset after terminal state without retaining previous episode inventory or labels.
- Terminal transition masking and finite horizon.
- All-invalid action mask, invalid index, NaN/inf feature or reward.
- Reproducibility under fixed seeds and documented nondeterminism on hardware.
- Reward incentives that favor deleting everything or keeping everything.
- Data leakage through metadata, candidate order, task templates, caches, filenames, or logs.

### Evaluation and benchmark

- Empty result set, answerer exception, missing provider usage, zero successful tasks.
- Different token counts or prompt templates across methods.
- Test-set reuse during model selection.
- Benchmark split/license changes or missing official scorer.
- LLM judge disagreement and answerer stochasticity.
- Latency/cost accounting that excludes compression or controller overhead.

## 9. Project-wide success criteria

The prototype is **implementation-complete** when all of the following hold:

1. Memory operations and prompt construction satisfy the invariants in Section 4 and focused tests pass.
2. Synthetic episodes are reproducible, finite, and leakage-tested.
3. Random, control, baseline, and DQN policies use the same task/result interfaces.
4. The training pipeline saves compatible checkpoints and validation metrics.
5. The final evaluation can run all methods over the same held-out examples and budget sweep.
6. Per-example records allow aggregate metrics and plots to be regenerated.
7. Failure cases, limitations, versions, and unmeasured costs are disclosed.

The research hypothesis is **supported** only if the predeclared trade-off criteria in Phase 7 are met on held-out evaluation with uncertainty reported. Otherwise the project can still be complete, but conclusions must accurately say the result was negative, mixed, or inconclusive.

## 10. Suggested execution order and milestone gates

| Milestone | Deliverable | Must pass before proceeding |
|---|---|---|
| M0: inspect/setup | Runnable package and smoke test | Existing project state preserved |
| M1: memory engine | Typed items, tokenizer, prompt builder, operations | Memory/token tests pass |
| M2: simulator | Synthetic task generator and finite environment | 1,000 random episodes; leakage tests pass |
| M3: baselines | Standardized full-history, window, heuristic, summarizer | Same-example/budget evaluation works |
| M4: DQN | Training loop, replay, checkpoints, logs | Small reproducible train/evaluate run |
| M5: controlled study | Seeds, test split, budget sweep, report | No test leakage; metrics reproducible |
| M6: transfer | Benchmark adapter and evaluation | License/protocol validated |
| M7: research report | Results, ablations, failures, limitations | Claims match measured evidence |

Prioritize correctness of the environment and fair evaluation over network complexity or number of benchmarks. If time is limited, complete M0–M5 with a high-quality synthetic study before attempting M6.

## 11. Recommended developer workflow and commands

Use commands consistent with the repository. If none exist, provide equivalents for:

```bash
python -m pytest
python -m memory_rl.controller.train --config configs/dqn.yaml
python -m memory_rl.evaluation.run_eval --config configs/evaluation.yaml
```

Expose configuration through files/CLI rather than editing source constants. Training and evaluation commands should:

- Return nonzero on invalid config or failed runs.
- Log config, seed, software/model/tokenizer versions, and output paths.
- Avoid overwriting existing results without explicit opt-in.
- Support a small smoke mode for CI and local development.
- Keep raw benchmark data and secrets external to the repository.

## 12. Explicit non-goals for the first version

- A production memory service or multi-user persistence layer.
- Autonomous tool execution or modification of external systems.
- Training an LLM to generate summaries as part of the DQN.
- Retrieval mixed into the first controller's action space.
- Claims of universal benchmark or real-world generalization.
- Claiming that lower token count alone demonstrates success.
- Hardcoding hypothetical scores or treating suggested hyperparameters as established optima.

