#!/bin/bash
set -uo pipefail

# Resume support: each benchmark's RUN_NAME defaults to "<bench>_<timestamp>",
# but can be overridden by setting WANDB_RUN_NAME externally. This points the
# eval at the same output dir as a previous run; the runner skips any house
# whose trajectories h5 still exists, so only deleted (partial) houses re-roll.
# Only one benchmark can be resumed per invocation (toggle just that one in RUN).
# Example:
#   WANDB_RUN_NAME=pnp-v1_20260428_103426 ./run_molmospaces_evals.sh

export CUDA_VISIBLE_DEVICES=2
#export MUJOCO_EGL_DEVICE_ID=0
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
#echo "MUJOCO_EGL_DEVICE_ID=$MUJOCO_EGL_DEVICE_ID"

export WANDB_PROJECT_NAME=molmospaces_tiptop
export WANDB_ENTITY=willshen-mit
CONFIG=molmo_spaces.evaluation.configs.evaluation_configs:TiptopPolicyEvalConfig
#CONFIG=molmo_spaces.evaluation.configs.evaluation_configs:DummyBenchmarkEvalConfig
NUM_WORKERS=18
BENCHMARK_ROOT_V1=assets/benchmarks/molmospaces-bench-v1
BENCHMARK_ROOT_V2=assets/benchmarks/molmospaces-bench-v2
LOG_DIR=logs
mkdir -p "$LOG_DIR"

# Toggle which benchmarks to run (1 = run, 0 = skip)
declare -A RUN=(
  # ms-bench
  [close-v1]=0
  [open-v1]=0
  [pick-v1.1]=0
  [pnp-v1]=1
  # mb-bench
  [pick-v1.5]=0
  [pick-v2-classic]=0
  [pick-v2-filament]=0
  [pick-v2-rand-cam]=0
  [pnp-v2]=0
  [pnp_next_to-v2]=0
  [pnp_color-v2]=0
)

should_run() { [[ "${RUN[$1]:-0}" == "1" ]]; }

# ===== ms-bench =====

#close-v1 (Close)
if should_run close-v1; then
  RUN_NAME="${WANDB_RUN_NAME:-close-v1_$(date +%Y%m%d_%H%M%S)}"
  WANDB_RUN_NAME="$RUN_NAME" \
    python molmo_spaces/evaluation/eval_main.py "$CONFIG" \
    --benchmark_dir "$BENCHMARK_ROOT_V1/ithor/FrankaCloseDataGenConfig/FrankaCloseDataGenConfig_20260123_json_benchmark" \
    --num_workers "$NUM_WORKERS" --wandb_project "$WANDB_PROJECT_NAME" 2>&1 | tee "$LOG_DIR/$RUN_NAME.log"
fi

#open-v1 (Open)
if should_run open-v1; then
  RUN_NAME="${WANDB_RUN_NAME:-open-v1_$(date +%Y%m%d_%H%M%S)}"
  WANDB_RUN_NAME="$RUN_NAME" \
    python molmo_spaces/evaluation/eval_main.py "$CONFIG" \
    --benchmark_dir "$BENCHMARK_ROOT_V1/ithor/FrankaOpenDataGenConfig/FrankaOpenDataGenConfig_20260123_json_benchmark" \
    --num_workers "$NUM_WORKERS" --wandb_project "$WANDB_PROJECT_NAME" 2>&1 | tee "$LOG_DIR/$RUN_NAME.log"
fi

#pick-v1.1 (Pick)
if should_run pick-v1.1; then
  RUN_NAME="${WANDB_RUN_NAME:-pick-v1.1_$(date +%Y%m%d_%H%M%S)}"
  WANDB_RUN_NAME="$RUN_NAME" \
    python molmo_spaces/evaluation/eval_main.py "$CONFIG" \
    --benchmark_dir "$BENCHMARK_ROOT_V1/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231" \
    --num_workers "$NUM_WORKERS" --wandb_project "$WANDB_PROJECT_NAME" 2>&1 | tee "$LOG_DIR/$RUN_NAME.log"
fi

#pnp-v1 (Pick and Place)
if should_run pnp-v1; then
  RUN_NAME="${WANDB_RUN_NAME:-pnp-v1_$(date +%Y%m%d_%H%M%S)}"
  WANDB_RUN_NAME="$RUN_NAME" \
    python molmo_spaces/evaluation/eval_main.py "$CONFIG" \
    --benchmark_dir "$BENCHMARK_ROOT_V1/procthor-10k/FrankaPickandPlaceDroidMiniBench/FrankaPickandPlaceDroidMiniBench_20260111_json_benchmark" \
    --num_workers "$NUM_WORKERS" --wandb_project "$WANDB_PROJECT_NAME" 2>&1 | tee "$LOG_DIR/$RUN_NAME.log"
fi

# ===== mb-bench =====

#pick-v1.5 (Pick-MSProc)
if should_run pick-v1.5; then
  RUN_NAME="${WANDB_RUN_NAME:-pick-v1.5_$(date +%Y%m%d_%H%M%S)}"
  WANDB_RUN_NAME="$RUN_NAME" \
    python molmo_spaces/evaluation/eval_main.py "$CONFIG" \
    --benchmark_dir "$BENCHMARK_ROOT_V2/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231" \
    --num_workers "$NUM_WORKERS" --wandb_project "$WANDB_PROJECT_NAME" 2>&1 | tee "$LOG_DIR/$RUN_NAME.log"
fi

#pick-v2-classic
if should_run pick-v2-classic; then
  RUN_NAME="${WANDB_RUN_NAME:-pick-v2-classic_$(date +%Y%m%d_%H%M%S)}"
  WANDB_RUN_NAME="$RUN_NAME" \
    python molmo_spaces/evaluation/eval_main.py "$CONFIG" \
    --benchmark_dir "$BENCHMARK_ROOT_V2/procthor-objaverse/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark" \
    --num_workers "$NUM_WORKERS" --wandb_project "$WANDB_PROJECT_NAME" 2>&1 | tee "$LOG_DIR/$RUN_NAME.log"
fi

#pick-v2-filament
if should_run pick-v2-filament; then
  RUN_NAME="${WANDB_RUN_NAME:-pick-v2-filament_$(date +%Y%m%d_%H%M%S)}"
  WANDB_RUN_NAME="$RUN_NAME" \
    python molmo_spaces/evaluation/eval_main.py "$CONFIG" \
    --benchmark_dir "$BENCHMARK_ROOT_V2/procthor-objaverse/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark" \
    --use-filament \
    --num_workers "$NUM_WORKERS" --wandb_project "$WANDB_PROJECT_NAME" 2>&1 | tee "$LOG_DIR/$RUN_NAME.log"
fi

#pick-v2-rand-cam
if should_run pick-v2-rand-cam; then
  RUN_NAME="${WANDB_RUN_NAME:-pick-v2-rand-cam_$(date +%Y%m%d_%H%M%S)}"
  WANDB_RUN_NAME="$RUN_NAME" \
    python molmo_spaces/evaluation/eval_main.py "$CONFIG" \
    --benchmark_dir "$BENCHMARK_ROOT_V2/procthor-objaverse/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark" \
    --use-filament \
    --camera_names randomized_zed2_analogue_1 wrist_camera_zed_mini \
    --num_workers "$NUM_WORKERS" --wandb_project "$WANDB_PROJECT_NAME" 2>&1 | tee "$LOG_DIR/$RUN_NAME.log"
fi

#pnp-v2
if should_run pnp-v2; then
  RUN_NAME="${WANDB_RUN_NAME:-pnp-v2_$(date +%Y%m%d_%H%M%S)}"
  WANDB_RUN_NAME="$RUN_NAME" \
    python molmo_spaces/evaluation/eval_main.py "$CONFIG" \
    --benchmark_dir "$BENCHMARK_ROOT_V2/procthor-objaverse/FrankaPickandPlaceHardBench/FrankaPickandPlaceHardBench_20260206_json_benchmark" \
    --use-filament \
    --num_workers "$NUM_WORKERS" --wandb_project "$WANDB_PROJECT_NAME" 2>&1 | tee "$LOG_DIR/$RUN_NAME.log"
fi

#pnp_next_to-v2
if should_run pnp_next_to-v2; then
  RUN_NAME="${WANDB_RUN_NAME:-pnp_next_to-v2_$(date +%Y%m%d_%H%M%S)}"
  WANDB_RUN_NAME="$RUN_NAME" \
    python molmo_spaces/evaluation/eval_main.py "$CONFIG" \
    --benchmark_dir "$BENCHMARK_ROOT_V2/procthor-objaverse/FrankaPickandPlaceNextToHardBench/FrankaPickandPlaceNextToHardBench_20260305_json_benchmark" \
    --use-filament \
    --num_workers "$NUM_WORKERS" --wandb_project "$WANDB_PROJECT_NAME" 2>&1 | tee "$LOG_DIR/$RUN_NAME.log"
fi

#pnp_color-v2
if should_run pnp_color-v2; then
  RUN_NAME="${WANDB_RUN_NAME:-pnp_color-v2_$(date +%Y%m%d_%H%M%S)}"
  WANDB_RUN_NAME="$RUN_NAME" \
    python molmo_spaces/evaluation/eval_main.py "$CONFIG" \
    --benchmark_dir "$BENCHMARK_ROOT_V2/procthor-objaverse/FrankaPickandPlaceColorHardBench/FrankaPickandPlaceColorHardBench_20260304_json_benchmark" \
    --use-filament \
    --num_workers "$NUM_WORKERS" --wandb_project "$WANDB_PROJECT_NAME" 2>&1 | tee "$LOG_DIR/$RUN_NAME.log"
fi

echo "Done"
