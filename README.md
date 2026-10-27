# Noesis: Cache-Aware RL Memory Controller for LLM Agents

Research prototype for our AIML EL project. A frozen LLM agent works in **ALFWorld** and
**WebShop**. Before every LLM call, a small **PPO** controller decides whether to edit the
agent's history at all, and if so chooses **KEEP / COMPRESS / PRUNE** for each past step.

The twist: the reward is **task success minus the real API cost after prompt-caching
discounts**, not the raw token count. Pruning an old message breaks the cached prefix and
re-bills everything after it, so *fewer tokens does not always mean a smaller bill*. An
extension adds **budget-constrained** control (PPO-Lagrangian, budget-conditioned policy).

> **Status: Phases 0–1 done (memory engine + token accounting, 32 tests passing).
> Phase 2 (cache cost model) is next. No RL training has run and no result exists yet.**
> All hypotheses (prd.md §4) are untested. Nothing here may be reported as a measured
> result until it comes from a real run.

| Document | Purpose |
|---|---|
| [`prd.md`](prd.md) | Requirements: problem, goals, hypotheses, metrics, **phases and work split** |
| [`plan.md`](plan.md) | Implementation contract: invariants, interfaces, per-phase tests and exit gates |

## Requirements

- Python **3.11+**
- [`uv`](https://docs.astral.sh/uv/) for installs
- Docker + `docker compose` (optional, for the image workflow)
- Later phases: a GPU for the vLLM-served frozen LLM (see prd.md §9)

Install into your virtualenv (editable, with dev tools):

```bash
uv venv .venv                     # or reuse an existing env
uv pip install -e ".[dev]"
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

Planned entry points, which don't exist until their phase lands (see plan.md §11):

```bash
python -m memory_rl.cost.cache_sim --demo                                # Phase 2
python -m memory_rl.envs.agent_loop --env toy --llm mock                 # Phase 3
python -m memory_rl.controller.train --config configs/ppo.yaml           # Phase 6
python -m memory_rl.evaluation.run_eval --config configs/evaluation.yaml # Phase 8
```

## Phases

| Phase | Name | Status |
|---|---|---|
| 0 | Setup (package, Docker, CI) | ✅ |
| 1 | Memory engine + token accounting | ✅ |
| 2 | Cache-aware cost model (pricing profiles, prefix-cache simulator) | ⏭ next |
| 3 | Frozen-LLM agent harness (vLLM, mock LLM, toy env, ALFWorld, WebShop) | ☐ |
| 4 | Controller MDP + cache/token reward | ☐ |
| 5 | Baselines (window, obs masking, LLMLingua, threshold compaction, …) | ☐ |
| 6 | PPO controller (cache-reward and token-reward variants) | ☐ |
| 7 | Budget-constrained extension (PPO-Lagrangian, budget-conditioned) | ☐ |
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
src/memory_rl/     memory/ (done); cost/, llm/, envs/, controller/, baselines/,
                   evaluation/ land with their phases (plan.md §5)
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
