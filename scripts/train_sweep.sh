#!/usr/bin/env bash
# Train every PPO variant x lambda x seed in parallel (CPU, one thread each).
# Usage: scripts/train_sweep.sh [out_dir]   (default: checkpoints)
set -euo pipefail
OUT="${1:-checkpoints}"
mkdir -p "$OUT/logs"
for variant in cache token; do
  for lam in 0.3 1 3; do
    for seed in 0 1 2; do
      python -m memory_rl.controller.train --config configs/ppo.yaml \
        --variant "$variant" --lam "$lam" --seed "$seed" --out "$OUT" \
        > "$OUT/logs/${variant}_lam${lam}_s${seed}.log" 2>&1 &
    done
  done
done
wait
echo "sweep done: $OUT"
