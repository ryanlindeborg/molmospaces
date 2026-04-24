#!/bin/bash
set -uo pipefail

export WANDB_PROJECT_NAME=molmospaces_tiptop
export WANDB_ENTITY=willshen-mit
CONFIG=molmo_spaces.evaluation.configs.evaluation_configs:TiptopPolicyEvalConfig
#CONFIG=molmo_spaces.evaluation.configs.evaluation_configs:DummyBenchmarkEvalConfig
NUM_WORKERS=12
BENCH_ROOT=assets/benchmarks/molmospaces-bench-v2
LOG_DIR=logs
mkdir -p "$LOG_DIR"

RUN_NAME="pick-v1.5_$(date +%Y%m%d_%H%M%S)"
WANDB_RUN_NAME="$RUN_NAME" \
  python molmo_spaces/evaluation/eval_main.py "$CONFIG" \
  --benchmark_dir "$BENCH_ROOT/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231" \
  --num_workers "$NUM_WORKERS" 2>&1 | tee "$LOG_DIR/$RUN_NAME.log"

#RUN_NAME="pickandplace_$(date +%Y%m%d_%H%M%S)"
#WANDB_RUN_NAME="$RUN_NAME" \
#  python molmo_spaces/evaluation/eval_main.py "$CONFIG" \
#  --benchmark_dir "$BENCH_ROOT/FrankaPickandPlaceDroidMiniBench/FrankaPickandPlaceDroidMiniBench_20260111_json_benchmark" \
#  --num_workers "$NUM_WORKERS" 2>&1 | tee "$LOG_DIR/$RUN_NAME.log"

echo "Done"
