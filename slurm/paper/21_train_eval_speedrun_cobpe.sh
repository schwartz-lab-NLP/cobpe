#!/usr/bin/env bash
#SBATCH --job-name=speedrun-cobpe
#SBATCH --gres=gpu:8
#SBATCH --mem=80g
#SBATCH -c32
#SBATCH --time=48:0:0
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

set -euo pipefail

if [[ -n "${COBPE_REPO_ROOT_OVERRIDE:-}" ]]; then
  COBPE_REPO_ROOT="$COBPE_REPO_ROOT_OVERRIDE"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/slurm/paper/_env.sh" ]]; then
  COBPE_REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  COBPE_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
fi
export COBPE_REPO_ROOT
SCRIPT_DIR="$COBPE_REPO_ROOT/slurm/paper"
source "$SCRIPT_DIR/_env.sh"
setup_paper_env

MODIFIER_CONDITIONING_MODE="${MODIFIER_CONDITIONING_MODE:-concat_gated_refine}"
MODIFIER_GATE_MODE="${MODIFIER_GATE_MODE:-per_group}"
case "$MODIFIER_GATE_MODE" in
  scalar|per_group) ;;
  *) echo "ERROR: MODIFIER_GATE_MODE must be scalar or per_group." >&2; exit 1 ;;
esac

mkdir -p logs "$OUTPUT_BASE_DIR"
LOCAL_PARQUET_DIR="$(stage_dir_to_local "$LOCAL_PARQUET_DIR" base_data_climbmix)"
export NANOCHAT_TOKENIZER_DIR="$TOKENIZER_COBPE_DIR"

MODEL_NAME="${MODEL_NAME:-d24}"
DEPTH="${DEPTH:-24}"
SPEEDRUN_ARCHITECTURE_PRESET="${SPEEDRUN_ARCHITECTURE_PRESET:-speedrun}"
if [[ -z "${SPEEDRUN_TAG_SUFFIX:-}" ]]; then
  case "$SPEEDRUN_ARCHITECTURE_PRESET" in
    speedrun) SPEEDRUN_TAG_SUFFIX="_speedrun" ;;
    speedrun_cutoff_20260312) SPEEDRUN_TAG_SUFFIX="_cutoff_20260312" ;;
    *) SPEEDRUN_TAG_SUFFIX="_${SPEEDRUN_ARCHITECTURE_PRESET}" ;;
  esac
fi
MODEL_TAG="${MODEL_TAG:-cobpe_${MODEL_NAME}${SPEEDRUN_TAG_SUFFIX}}"
WORLD_SIZE="${WORLD_SIZE:-8}"
COBPE_ARCHITECTURE_PRESET="${COBPE_ARCHITECTURE_PRESET:-${SPEEDRUN_ARCHITECTURE_PRESET}_cobpe}"

print_paper_run_paths "$MODEL_TAG"

uv run --no-sync torchrun --standalone --nproc_per_node="$WORLD_SIZE" -m scripts.base_train \
  --wandb-run "$MODEL_TAG" \
  --model-tag "$MODEL_TAG" \
  --resume-from-step "${RESUME_FROM_STEP:--1}" \
  --output-base-dir "$OUTPUT_BASE_DIR" \
  --local-parquet-dir "$LOCAL_PARQUET_DIR" \
  --depth "$DEPTH" \
  --aspect-ratio "${ASPECT_RATIO:-64}" \
  --head-dim "${HEAD_DIM:-128}" \
  --max-seq-len "${MAX_SEQ_LEN:-2048}" \
  --architecture-preset "$COBPE_ARCHITECTURE_PRESET" \
  --modifier-head "$MODIFIER_CONDITIONING_MODE" \
  --modifier-gates "$MODIFIER_GATE_MODE" \
  --device-batch-size "${DEVICE_BATCH_SIZE:-16}" \
  --total-batch-size "${TOTAL_BATCH_SIZE:-1048576}" \
  --target-param-data-ratio "${TARGET_PARAM_DATA_RATIO:-6.5}" \
  --muon-final-momentum "${MUON_FINAL_MOMENTUM:-0.97}" \
  --eval-every "${EVAL_EVERY:--1}" \
  --eval-tokens "${EVAL_TOKENS:-41943040}" \
  --core-metric-every -1 \
  --sample-every -1 \
  --window-pattern "${WINDOW_PATTERN:-L}" \
  --save-every "${SAVE_EVERY:-4000}"

if [[ "${RUN_EVAL:-1}" == "1" ]]; then
  typeset -a DOWNSTREAM_PRESET_ARGS
  DOWNSTREAM_PRESET_ARGS=()
  if [[ -n "${DOWNSTREAM_PRESET:-}" ]]; then
    DOWNSTREAM_PRESET_ARGS=(--downstream-preset "$DOWNSTREAM_PRESET")
  fi
  STEP="$(resolve_latest_step "$OUTPUT_BASE_DIR/base_checkpoints/$MODEL_TAG")"
  uv run --no-sync torchrun --standalone --nproc_per_node="$WORLD_SIZE" -m scripts.base_eval \
    --eval "${EVAL_MODES:-core,sample,bpb}" \
    --model-tag "$MODEL_TAG" \
    --step "$STEP" \
    --output-base-dir "$OUTPUT_BASE_DIR" \
    --local-parquet-dir "$LOCAL_PARQUET_DIR" \
    --max-per-task "${CORE_MAX_PER_TASK:--1}" \
    --device-batch-size "${DEVICE_BATCH_SIZE:-16}" \
    "${DOWNSTREAM_PRESET_ARGS[@]+"${DOWNSTREAM_PRESET_ARGS[@]}"}"
fi
