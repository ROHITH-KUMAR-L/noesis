# Budget-Constrained RL for Task-Critical Memory Retention

Research prototype: a small DQN controller decides, for each memory unit, whether to
**KEEP**, **COMPRESS**, or **PRUNE** it, under a fixed input-token budget. A separate,
fixed answerer answers the task from the resulting active context.

> **Status: Phase 0 (setup) only. No training has run and no DQN result exists yet.**
> DQN beating the non-RL baselines is a *hypothesis*, not an established outcome —
> see `plan.md` §2.3 and §9. Nothing in this repository may be reported as a measured
> result until it comes from a real run.

`plan.md` is the implementation contract: phases, invariants, per-phase tests, and
gates. Read the phase you are working on before changing code.

## Requirements

- Python **3.11+**
- `uv` for package installs (the project venv has **no `pip`**)
- Docker + `docker compose` for the image workflow

Local development runs against the existing **`myenv`** virtualenv — do not create a
second one:

```bash
source /home/rohith/.venvs/myenv/bin/activate.fish      # fish
uv pip install --python /home/rohith/.venvs/myenv/bin/python -e ".[dev]"
```

## Commands

Run these in order; each must exit nonzero on failure.

```bash
python -m memory_rl.smoke     # smoke: installation is usable
ruff check .                  # lint
mypy                          # typecheck
python -m pytest              # tests
```

Focused runs:

```bash
python -m pytest tests/test_smoke.py
python -m pytest -k smoke
```

Planned entrypoints (they do not exist until their phase lands — do not assume they run):

```bash
python -m memory_rl.controller.train    --config configs/dqn.yaml         # Phase 5
python -m memory_rl.evaluation.run_eval --config configs/evaluation.yaml  # Phase 7
```

## Docker

The image is intentionally minimal: `python:3.11-slim` plus a self-contained venv,
built in a multi-stage Dockerfile (classic builder works; CI adds buildx caching).
Heavy ML dependencies stay in `myenv`, not in the image.

```bash
docker compose build                       # build (reuses the layer + GHA/buildx cache)
docker compose run --rm app                # smoke check inside the container
docker compose run --rm app python -m ...  # any project command
```

- **Named volumes** persist `uv`, Hugging Face, and Torch caches across rebuilds.
- **Named network** `noesis` is defined in `docker-compose.yaml`.
- `./configs`, `./data`, and `./results` are bind-mounted; raw benchmark data and
  secrets are never copied into the image (see `.dockerignore`).
- CI enforces an image size budget (400 MiB) and runs the image's smoke check.

## Layout

```text
pyproject.toml     packaging, ruff/mypy/pytest config (single source of truth)
src/memory_rl/     package (smoke.py today; modules per plan.md §5 as phases land)
tests/             pytest suite; no test may be skipped to make a phase "pass"
configs/           run configs (YAML); edit these, not source constants
data/              raw/ is gitignored; never commit restricted or benchmark data
results/           generated output, ignored except .gitkeep
```

## Ground rules

- Never present illustrative or placeholder numbers as measured results.
- Ground-truth labels stay out of policy observations, features, and prompts
  (`tests/test_no_information_leakage.py`).
- No silent prompt truncation; over-budget means a recorded violation.
- Tune on validation only; never on the final test set.
- Keep credentials, restricted data, checkpoints, caches, and absolute local paths
  out of git.
