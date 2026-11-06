# Project state

Snapshot of where the work stands, so it can be picked up later. Last updated 2026-11-06,
on branch `feat/full-pipeline`. `main` is still at `fba7de8`.

**Bottom line:** Phases 0–6 are implemented, tested and committed. The demo web UI is built and
committed, but no one has looked at it in a browser yet. The final evaluation (Phase 8) has not
run. Every number below comes from the **toy KeyHunt environment with a scripted mock LLM**.
None of it is an ALFWorld result or a measured provider bill.

## How to run

```bash
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev,web]"
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # [rl] extra
python -m memory_rl.smoke && ruff check . && mypy && python -m pytest   # 140 pass, 1 skip

python -m memory_rl.cost.cache_sim --demo                       # cost simulator demo
python -m memory_rl.envs.agent_loop --episodes 3                # agent episodes (mock LLM)
python -m memory_rl.evaluation.run_eval --config configs/baselines.yaml --overwrite
scripts/train_sweep.sh checkpoints                              # 18 PPO runs, ~15 min on CPU

cd web && npm install && npm run build && cd ..                 # build the UI
python -m memory_rl.web                                         # http://127.0.0.1:8000
```

Dev mode for the UI: run `python -m memory_rl.web` in one shell and `cd web && npm run dev` in
another (port 5173, which proxies `/api` to 8000).

Real LLM (not tested yet): copy `.env.example` to `.env` and fill in `NOESIS_LLM_BASE_URL`,
`NOESIS_LLM_API_KEY`, `NOESIS_LLM_MODEL` and `NOESIS_LLM_MAX_CALLS`. Then add
`--llm openai --allow-paid` to `agent_loop` or `run_eval`. The call cap is enforced. No paid call
has been made.

## Done

| Commit | Stage | What it contains |
|---|---|---|
| `470dbd4` | Repair | Fixed `fba7de8`, the old DQN-plan synthetic-QA env: 3 failing tests, ruff and mypy errors, and numpy used without being declared. It is kept aside as a legacy module. |
| `43cffe8` | Phase 2: cost model | `cost/`: `PricingProfile` plus a validated YAML loader, and `PrefixCacheSimulator` with automatic (LCP, block rounding) and breakpoint (whole-entry hit) modes, a minimum cacheable length, and TTL. Also `CacheAwareCostMeter` / `TokenCostMeter`. `configs/pricing/` holds `anthropic-sonnet-5-5`, `openai-gpt-5-4`, `openai-gpt-6-1-sol` (from the provider pages, read 2026-11-04) and an illustrative test profile. The five-call cost tables were computed by hand and match. |
| `6199e39` | Phase 3: harness | `envs/toy.py`: the KeyHunt env (search/open/submit; long noisy listings; keys needed again at the end). `envs/agent_prompt.py`: a prefix-stable serializer with per-item token spans. `envs/agent_loop.py`: a stepwise `AgentEpisode` with masks, compress-at-most-once, and context overflow recorded as a failure. `llm/`: an OpenAI-compatible client reading `.env` (call cap, retries, `--allow-paid`) and a scripted mock agent. `evaluation/records.py`: JSONL records. |
| `1aedfa0` | Phase 4: controller MDP | `controller/features.py`: schema `keyhunt-ctrl-v1` with 64 slots × 12 item features (including `suffix_share`, the re-billing cost of editing an item) and 11 global features. `controller/env.py`: gate plus masked per-item actions. `controller/reward.py`: −λ·cost/Z + task reward, cache and token variants. Exit gate: 1,000 random episodes run cleanly. The H2 mechanism shows on 20/20 tasks: trimming already-sent history costs more dollars with fewer tokens. |
| `02e39df` | Phase 5: baselines | `baselines/rules.py`: sliding window, observation masking, threshold compaction, the CADOC rule (arXiv:2609.37012, Eq. 11–14, with deviations in `docs/baselines.md`) and random. LLMLingua-2 is an optional extra. `evaluation/run_eval.py` and `metrics.py` provide the runner, task-bootstrap CIs and paired differences. `configs/env/toy.yaml` sets the benchmark: 3–6 keys and a **5,000-token context budget**. |
| `c6e44cf` | Phase 6: PPO | `controller/policy.py`: MLP encoder, mean+max pooling, a gate head (EDIT masked when there are no candidates), a masked item head and a value head, initialised biased toward KEEP. `controller/ppo.py` (GAE, clip, non-finite checks) and `controller/train.py` (train split only, greedy validation on `valid`, checkpoint selection there). `configs/ppo.yaml`, `scripts/train_sweep.sh`. CI installs CPU torch. |
| (this commit) | Web UI | FastAPI backend in `src/memory_rl/web/`: `/api/meta`, `/api/episode` (runs a live episode), `/api/playground` and `/api/results`. React + Vite front end in `web/` with three views: episode replay (two policies side by side, the per-call "cache ledger" strip, and the context the model saw), the cache playground and the results Pareto chart with CIs. The colours passed the dataviz palette validator. |

### Validation-split observations

These are **not final results**: validation split, toy env, mock LLM, 30 tasks, 3 seeds each.
They are kept here to guide the next session.

Baselines, Anthropic Sonnet 5.5 profile (`results/baselines_valid/`, 60 tasks × 3 seeds):

| Method | Success | Mean cost per episode |
|---|---|---|
| Full history | 0.53 (overflows on long tasks) | $0.0160 |
| Threshold compaction T=3500 | 1.00 | ≈ $0.018 |
| CADOC rule M=2 | 1.00 | $0.0212 |
| Observation masking M=0/1 | 1.00 | $0.0229, with the fewest tokens |
| Sliding window | 0.00–0.38 | — |

PPO best checkpoints (validation, 30 tasks, seeds 0/1/2):

| Variant | λ | Success | Real cost | Edits per episode |
|---|---|---|---|---|
| cache | 0.3 | 1.00 / 1.00 / 1.00 | $0.0177 / 0.0175 / 0.0177 | 2.0 / 3.7 / 3.2 |
| cache | 1 | 1.00 / 1.00 / 1.00 | $0.0175 / 0.0178 / 0.0179 | 3.7 / 3.1 / 1.9 |
| cache | 3 | 0.60 for all seeds (never learned to edit) | $0.0158 | 0 |
| token | 0.3 | 1.00 for all seeds | $0.0199 / 0.0227 / 0.0197 | 8.5 |
| token | 1 | 1.00 for all seeds | $0.0200 / 0.0194 / 0.0197 | 8.5 |
| token | 3 | 1.00 for all seeds | $0.0189 / 0.0183 / 0.0193 | 8.5 |

So far these numbers point the way the hypotheses predict. The cache-aware PPO edits rarely and
costs less. The token-reward PPO edits every turn and pays more in real dollars (H2/H3 on
validation). This is **not confirmed** until the Phase 8 test run, with paired CIs.

## Remaining

1. **Phase 8: final evaluation** (core, next step)
   - Freeze the choices made on validation: which λ for each variant (cache λ=0.3 or 1; token
     λ=1 or 3), the baseline knobs, and the checkpoints. Write them into
     `configs/evaluation.yaml` *before* touching test.
   - Run every method × knob on the **test** split (100 tasks, seeds 0–2) under all three
     profiles. `run_eval` loads PPO with `--method ppo_cache --checkpoint …`. Its loop over
     several checkpoints may need a small extension so the 3 seeds land in one run.
   - Compute paired bootstrap CIs (`metrics.paired_difference_ci`) for H1, H2 and H3 against
     observation masking and the CADOC rule. Produce the plots (the UI's Results view reads
     `results/*/summary.json`). Add `evaluation/plots.py` for static figures if the report
     needs them.
   - Report profile transfer (train on Anthropic, evaluate on the OpenAI profiles) labelled as
     *transfer*.
2. **UI visual check**: the UI has not been seen in a browser yet. The Playwright MCP needs
   Chrome, and a Playwright Chromium download into the session scratchpad was started. Take
   screenshots of all three views at desktop and phone width, fix layout issues, then rebuild.
3. **README / CLAUDE.md status update**: the phase table, the new commands, the toy benchmark
   setting, the demo instructions and the `.env` usage.
4. **Real-LLM check (user)**: fill in `.env` and run a few `--llm openai --allow-paid` episodes.
   The mock's behaviour (noting keys in its thoughts) was designed to resemble a ReAct agent, so
   check that a real model behaves similarly. Compare provider-reported `cached_tokens` with the
   simulator.
5. **Phase 7: budget control (stretch, not started)**: PPO-Lagrangian, a forced edit under a
   per-call token cap, and budget-conditioned features. The global feature slots already exist.
   `SCHEMA_VERSION` must be bumped.
6. **Not done / out of reach today**:
   - ALFWorld adapter and competence gate. A real LLM is needed, and the 6 GB GPU cannot host
     the planned 7B model.
   - WebShop (stretch).
   - LLMLingua baseline run (optional extra not installed).
   - Phase 9 ablations, arXiv novelty re-check, report.
   - Reading CADOC appendices A, B and F against the PDF.

## Design decisions worth remembering

- **Toy benchmark setting** (`configs/env/toy.yaml`): with no context pressure, "never edit"
  is near-optimal under cheap cache reads (Sonnet 5.5 reads cost 0.05×). The 5K budget makes
  full history fail long tasks, so whether, what and *when* to compact all matter. The
  unconstrained setting (16K) should be reported as a secondary condition.
- **Mock agent** notes each key in its thought (`"REC-x shows the key for p: K-1234."`), the way
  ReAct agents use a scratchpad. Without this, observation masking was unfairly 0% successful.
- **PPO init**: the item head starts biased toward KEEP (`keep_bias=1.0`, chosen on validation
  from {1.0, 1.5, 2.0}). With a uniform head, exploratory edits pruned a third of the history
  and the gate collapsed to never editing.
- **Breakpoint cache mode** hits only on whole entries (Anthropic-style), which is stricter than
  plan.md §6.4's LCP wording. The deviation is documented in `cost/cache_sim.py`.
- `checkpoints/`, `results/`, `web/node_modules/` and `web/dist/` are gitignored. Re-create them
  with the commands above.
