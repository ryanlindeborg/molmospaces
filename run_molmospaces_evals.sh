#!/bin/bash
set -uo pipefail

export WANDB_PROJECT_NAME=molmospaces_tiptop
CONFIG=molmo_spaces.evaluation.configs.evaluation_configs:TiptopPolicyEvalConfig
NUM_WORKERS=12
BENCH_ROOT=assets/benchmarks/molmospaces-bench-v1/procthor-10k
LOG_DIR=logs
mkdir -p "$LOG_DIR"

RUN_NAME="pick_$(date +%Y%m%d_%H%M%S)"
WANDB_RUN_NAME="$RUN_NAME" \
  python molmo_spaces/evaluation/eval_main.py "$CONFIG" \
  --benchmark_dir "$BENCH_ROOT/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231" \
  --num_workers "$NUM_WORKERS" 2>&1 | tee "$LOG_DIR/$RUN_NAME.log"

# Let the CPU cool down between runs.
sleep 300

RUN_NAME="pickandplace_$(date +%Y%m%d_%H%M%S)"
WANDB_RUN_NAME="$RUN_NAME" \
  python molmo_spaces/evaluation/eval_main.py "$CONFIG" \
  --benchmark_dir "$BENCH_ROOT/FrankaPickandPlaceDroidMiniBench/FrankaPickandPlaceDroidMiniBench_20260111_json_benchmark" \
  --num_workers "$NUM_WORKERS" 2>&1 | tee "$LOG_DIR/$RUN_NAME.log"

echo "Done"