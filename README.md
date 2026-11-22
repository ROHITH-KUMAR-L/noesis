# Noesis: Cache-Aware RL Memory Controller for LLM Agents

Research prototype for our AIML EL project. A frozen LLM agent works in **ALFWorld** (core)
and **WebShop** (stretch). Before every LLM call, a small **PPO** controller decides whether to edit the
agent's history at all, and if so chooses **KEEP / COMPRESS / PRUNE** for each past step.

The twist: the reward is **task success minus the real API cost after prompt-caching
discounts**, not the raw token count. Pruning an old message breaks the cached prefix and
re-bills everything after it, so *fewer tokens does not always mean a smaller bill*. An
extension (stretch) adds **budget-constrained** control (PPO-Lagrangian, budget-conditioned policy).

> **Status: Phases 0–2 and 4 done; 3, 5 and 6 done on the toy env only (their ALFWorld parts are
> pending). 140 tests pass, 1 skipped. Next: a frozen LLM that passes the ALFWorld competence gate.**
> PPO has only been trained on the toy KeyHunt env with a mock LLM, judged on the validation
> split. No test-split, ALFWorld or real-LLM result exists, so all hypotheses (prd.md §4) are
> untested. Nothing here may be reported as a measured result until it comes from a real run.
> Details and next steps: [`state.md`](state.md).

| Document | Purpose |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | Agent guidance; authoritative on **scope, novelty claims and prior art** |
| [`prd.md`](prd.md) | Requirements: problem, goals, hypotheses, metrics, **phases and work split** |
| [`plan.md`](plan.md) | Implementation contract: invariants, interfaces, per-phase tests and exit gates |

## Requirements

- Python **3.11+**; use **3.11** to match CI. On 3.12+, uv resolves numpy ≥ 2.5, whose type
  stubs make `mypy` (checking as 3.11) fail on `ndarray.max()`/`.min()`.
- [`uv`](https://docs.astral.sh/uv/) for installs
- Docker + `docker compose` (optional, for the image workflow)
- Later phases: a GPU for the vLLM-served frozen LLM (see prd.md §9)

Install into your virtualenv (editable, with dev tools):

```bash
uv venv .venv --python 3.11      # or reuse an existing env
uv pip install -e ".[dev,web]"
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # [rl] extra, CPU
```

Without installing, prefix commands with `PYTHONPATH=src`.

## Commands

Run in this order; each exits nonzero on failure (CI does the same):

```bash
python -m memory_rl.smoke     # smoke: installation is usable
ruff check .                  # lint
mypy                          # typecheck
python -m pytest              # tests
```

Entry points (see plan.md §11; `configs/evaluation.yaml` lands with Phase 8):

```bash
python -m memory_rl.cost.cache_sim --demo                                    # cost simulator demo
python -m memory_rl.envs.agent_loop --env toy --llm mock                     # agent episodes
python -m memory_rl.evaluation.run_eval --config configs/baselines.yaml      # baselines
python -m memory_rl.controller.train --config configs/ppo.yaml               # PPO (needs torch)
scripts/train_sweep.sh checkpoints                                           # 18 PPO runs
python -m memory_rl.web                    # demo UI on :8000; build first: cd web && npm i && npm run build
```

## Phases

| Phase | Name | Status |
|---|---|---|
| 0 | Setup (package, Docker, CI) | ✅ |
| 1 | Memory engine + token accounting | ✅ |
| 2 | Cache-aware cost model (pricing profiles, prefix-cache simulator) | ✅ |
| 3 | Frozen-LLM agent harness (vLLM, mock LLM, toy env, ALFWorld; WebShop is stretch) | ◐ toy env + mock/OpenAI-compatible client; ALFWorld not started |
| 4 | Controller MDP + cache/token reward | ✅ |
| 5 | Baselines (window, obs masking, LLMLingua, threshold compaction, CADOC-style, …) | ◐ toy env; ALFWorld runs pending; LLMLingua not wired into `run_eval` |
| 6 | PPO controller (cache-reward and token-reward variants) | ◐ toy env (validation); ALFWorld subset run pending |
| 7 | Budget-constrained extension (PPO-Lagrangian, budget-conditioned), *stretch* | ☐ |
| 8 | Final evaluation (success vs real-cost Pareto) | ☐ |
| 9 | Ablations, novelty re-check, report | ☐ |

## Docker

The image is intentionally minimal: `python:3.11-slim` plus a self-contained venv, built
in a multi-stage Dockerfile. Heavy ML dependencies (torch, vLLM, environments) are not in
the image.

```bash
docker compose build
docker compose run --rm app                # smoke check inside the container
docker compose run --rm app python -m ...  # any project command
```

- Named volumes persist `uv`, Hugging Face and Torch caches across rebuilds.
- `./configs`, `./data` and `./results` are bind-mounted; raw benchmark data and secrets
  are never copied into the image (see `.dockerignore`).
- CI enforces a 400 MiB image size budget and runs the image's smoke check.

## Layout

```text
pyproject.toml     packaging, ruff/mypy/pytest config
src/memory_rl/     memory/, cost/, llm/, envs/, controller/, baselines/, evaluation/, web/
                   (plan.md §5); environment/ + answering/ are the legacy DQN QA env (tests only)
web/               React + Vite demo UI (FastAPI backend is src/memory_rl/web/)
tests/             pytest suite; no test may be skipped to make a phase "pass"
configs/           run configs and pricing profiles (YAML); edit these, not constants
data/              raw/ is gitignored; never commit benchmark data
results/           generated output, ignored except .gitkeep
```

## Ground rules

- Never present illustrative or placeholder numbers as measured results.
- Simulator-only labels stay out of policy observations, features and prompts.
- No silent prompt truncation; context overflow and over-budget are recorded failures.
- Prices live only in pricing-profile configs with a source and date.
- Tune on validation only; never on the test split.
- No paid-API spending without explicit approval and a recorded cap.
- Keep credentials, benchmark data, checkpoints, caches and local paths out of git.
