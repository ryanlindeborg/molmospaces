#!/bin/bash
set -uo pipefail

export WANDB_PROJECT_NAME=molmospaces_tiptop
export WANDB_ENTITY=willshen-mit
CONFIG=molmo_spaces.evaluation.configs.evaluation_configs:TiptopPolicyEvalConfig
#CONFIG=molmo_spaces.evaluation.configs.evaluation_configs:DummyBenchmarkEvalConfig
NUM_WORKERS=12
BENCHMARK_ROOT_V2=assets/benchmarks/molmospaces-bench-v2
BENCHMARK_ROOT_V1=assets/benchmarks/molmospaces-bench-v1
LOG_DIR=logs
mkdir -p "$LOG_DIR"

#pick-v1
RUN_NAME="pick-v1_$(date +%Y%m%d_%H%M%S)"
WANDB_RUN_NAME="$RUN_NAME" \
  python molmo_spaces/evaluation/eval_main.py "$CONFIG" \
  --benchmark_dir "$BENCHMARK_ROOT_V1/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231" \
  --num_workers "$NUM_WORKERS" 2>&1 | tee "$LOG_DIR/$RUN_NAME.log"

#pnp-v1
RUN_NAME="pnp-v1_$(date +%Y%m%d_%H%M%S)"
WANDB_RUN_NAME="$RUN_NAME" \
  python molmo_spaces/evaluation/eval_main.py "$CONFIG" \
  --benchmark_dir "$BENCHMARK_ROOT_V1/procthor-10k/FrankaPickandPlaceDroidMiniBench/FrankaPickandPlaceDroidMiniBench_20260111_json_benchmark" \
  --num_workers "$NUM_WORKERS" 2>&1 | tee "$LOG_DIR/$RUN_NAME.log"

#pick-v2-classic
RUN_NAME="pick-v2-classic_$(date +%Y%m%d_%H%M%S)"
WANDB_RUN_NAME="$RUN_NAME" \
  python molmo_spaces/evaluation/eval_main.py "$CONFIG" \
  --benchmark_dir "$BENCHMARK_ROOT_V2/procthor-objaverse/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark" \
  --num_workers "$NUM_WORKERS" 2>&1 | tee "$LOG_DIR/$RUN_NAME.log"

#pick-v2-filament
RUN_NAME="pick-v2-filament_$(date +%Y%m%d_%H%M%S)"
WANDB_RUN_NAME="$RUN_NAME" \
  python molmo_spaces/evaluation/eval_main.py "$CONFIG" \
  --benchmark_dir "$BENCHMARK_ROOT_V2/procthor-objaverse/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark" \
  --num_workers "$NUM_WORKERS" 2>&1 | tee "$LOG_DIR/$RUN_NAME.log"

#pick-v2-rand-cam
RUN_NAME="pick-v2-rand-cam_$(date +%Y%m%d_%H%M%S)"
WANDB_RUN_NAME="$RUN_NAME" \
  python molmo_spaces/evaluation/eval_main.py "$CONFIG" \
  --benchmark_dir "$BENCHMARK_ROOT_V2/procthor-objaverse/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark" \
  --num_workers "$NUM_WORKERS" 2>&1 | tee "$LOG_DIR/$RUN_NAME.log"

#pnp-v2
RUN_NAME="pnp-v2_$(date +%Y%m%d_%H%M%S)"
WANDB_RUN_NAME="$RUN_NAME" \
  python molmo_spaces/evaluation/eval_main.py "$CONFIG" \
  --benchmark_dir "$BENCHMARK_ROOT_V2/procthor-objaverse/FrankaPickandPlaceHardBench/FrankaPickandPlaceHardBench_20260206_json_benchmark" \
  --num_workers "$NUM_WORKERS" 2>&1 | tee "$LOG_DIR/$RUN_NAME.log"

#pnp_next_to-v2
RUN_NAME="pnp_next_to-v2_$(date +%Y%m%d_%H%M%S)"
WANDB_RUN_NAME="$RUN_NAME" \
  python molmo_spaces/evaluation/eval_main.py "$CONFIG" \
  --benchmark_dir "$BENCHMARK_ROOT_V2/procthor-objaverse/FrankaPickandPlaceNextToHardBench/FrankaPickandPlaceNextToHardBench_20260305_json_benchmark" \
  --num_workers "$NUM_WORKERS" 2>&1 | tee "$LOG_DIR/$RUN_NAME.log"

#pnp_color-v2
RUN_NAME="pnp_color-v2_$(date +%Y%m%d_%H%M%S)"
WANDB_RUN_NAME="$RUN_NAME" \
  python molmo_spaces/evaluation/eval_main.py "$CONFIG" \
  --benchmark_dir "$BENCHMARK_ROOT_V2/procthor-objaverse/FrankaPickandPlaceColorHardBench/FrankaPickandPlaceColorHardBench_20260304_json_benchmark" \
  --num_workers "$NUM_WORKERS" 2>&1 | tee "$LOG_DIR/$RUN_NAME.log"

#pick-v1.5
RUN_NAME="pick-v1.5_$(date +%Y%m%d_%H%M%S)"
WANDB_RUN_NAME="$RUN_NAME" \
  python molmo_spaces/evaluation/eval_main.py "$CONFIG" \
  --benchmark_dir "$BENCHMARK_ROOT_V2/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231" \
  --num_workers "$NUM_WORKERS" 2>&1 | tee "$LOG_DIR/$RUN_NAME.log"

echo "Done"
