#!/usr/bin/env bash

# This file is sourced into interactive shells as well as used by batch
# wrappers. Do not enable shell-wide strict options here: a harmless failed
# command in the user's interactive session should not terminate the job.

ENV_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

paper_repo_root() {
  if [[ -n "${COBPE_REPO_ROOT_OVERRIDE:-}" ]]; then
    printf '%s\n' "$COBPE_REPO_ROOT_OVERRIDE"
  else
    cd -- "$ENV_SCRIPT_DIR/../.." && pwd -P
  fi
}

setup_paper_env() {
  export COBPE_REPO_ROOT="$(paper_repo_root)"
  export DATA_BASE_DIR="${DATA_BASE_DIR:-$COBPE_REPO_ROOT/experiments}"
  export EXP_DIR="${EXP_DIR:-paper}"
  export OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-$DATA_BASE_DIR/runs/$EXP_DIR}"
  # Keep heavy parquet data independently relocatable from experiment outputs.
  # PARQUET_DATA_DIR is the simple public override; LOCAL_PARQUET_DIR remains
  # supported for backwards compatibility. This applies to every paper .sh
  # wrapper because they all source this file.
  local parquet_dir_was_explicit=0
  if [[ -n "${PARQUET_DATA_DIR:-}" ]]; then
    parquet_dir_was_explicit=1
    export LOCAL_PARQUET_DIR="$PARQUET_DATA_DIR"
  else
    export LOCAL_PARQUET_DIR="${LOCAL_PARQUET_DIR:-$DATA_BASE_DIR/base_data_climbmix}"
  fi
  # nanochat.dataset uses NANOCHAT_BASE_DIR/base_data_climbmix as its download
  # destination. If parquet data was explicitly relocated, keep downloads there
  # too unless the caller provided a separate NANOCHAT_BASE_DIR.
  if [[ -z "${NANOCHAT_BASE_DIR:-}" ]]; then
    if (( parquet_dir_was_explicit )); then
      export NANOCHAT_BASE_DIR="$(dirname -- "$LOCAL_PARQUET_DIR")"
    else
      export NANOCHAT_BASE_DIR="$DATA_BASE_DIR"
    fi
  fi
  export TOKENIZER_CACHE_DIR="${TOKENIZER_CACHE_DIR:-$DATA_BASE_DIR/tokenizer_cache}"
  export TOKENIZER_BPE_DIR="${TOKENIZER_BPE_DIR:-$DATA_BASE_DIR/tokenizer_bpe}"
  export TOKENIZER_COBPE_DIR="${TOKENIZER_COBPE_DIR:-$DATA_BASE_DIR/tokenizer_cobpe}"
  export NANOCHAT_TOKENIZER_DIR="${NANOCHAT_TOKENIZER_DIR:-$TOKENIZER_BPE_DIR}"

  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
  export WANDB_MODE="${WANDB_MODE:-disabled}"

  # Respect each tool's usual per-user cache defaults unless overridden.
  if [[ -n "${CARGO_HOME:-}" ]]; then
    export PATH="$CARGO_HOME/bin:$PATH"
  fi
  # Slurm jobs execute the already-bootstrapped environment. Avoid an implicit
  # sync at job launch so the shared environment keeps its selected Torch
  # profile and is not mutated concurrently by multiple jobs.
  export UV_NO_SYNC=1

  mkdir -p "$OUTPUT_BASE_DIR" "$TOKENIZER_CACHE_DIR"
  cd "$COBPE_REPO_ROOT"
  if [[ -n "${VENV_PATH:-}" && -f "$VENV_PATH/bin/activate" ]]; then
    source "$VENV_PATH/bin/activate"
  elif [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
  fi
}

print_paper_run_paths() {
  local model_tag="${1:-<unset>}"
  local checkpoint_dir="${OUTPUT_BASE_DIR%/}/base_checkpoints/${model_tag}"
  printf 'Resolved experiment paths:\n'
  printf '  repo_root=%s\n' "${COBPE_REPO_ROOT:-<unset>}"
  printf '  output_base_dir=%s\n' "${OUTPUT_BASE_DIR:-<unset>}"
  printf '  checkpoint_dir=%s\n' "$checkpoint_dir"
  printf '  tokenizer_dir=%s\n' "${NANOCHAT_TOKENIZER_DIR:-<unset>}"
  printf '  parquet_dir=%s\n' "${LOCAL_PARQUET_DIR:-<unset>}"
  printf '  model_tag=%s\n' "$model_tag"
  printf '  resume_from_step=%s\n' "${RESUME_FROM_STEP:--1}"
  if [[ -d "$checkpoint_dir" ]]; then
    printf '  checkpoint_dir_exists=1\n'
  else
    printf '  checkpoint_dir_exists=0\n'
  fi
}

# Sourcing this file is the complete interactive setup command. Wrapper jobs
# may call setup_paper_env again harmlessly for backwards compatibility.
setup_paper_env

stage_dir_to_local() {
  local src="$1"
  local tag="$2"
  if [[ -z "$src" || ! -d "$src" || "${STAGE_LOCAL:-1}" != "1" || -z "${SLURM_TMPDIR:-}" ]]; then
    printf '%s' "$src"
    return
  fi
  local dst="${SLURM_TMPDIR%/}/cobpe_${SLURM_JOB_ID:-$$}/${tag}"
  mkdir -p "$dst"
  rsync -a --delete "${src%/}/" "${dst%/}/"
  printf '%s' "$dst"
}

preposition_list_for_profile() {
  case "${1:-core}" in
    core)
      printf '%s\n' by at of to in on with for from
      ;;
    expanded)
      printf '%s\n' by at of to in on with for from about across after against around before between during inside into onto over through under without
      ;;
    full)
      printf '%s\n' about above across after against along amid among around at before behind below beneath beside between beyond by despite during for from in inside into near of on onto outside over through throughout to toward towards under underneath until upon with within without per via
      ;;
    *)
      echo "ERROR: PREPOSITION_PROFILE must be core, expanded, or full." >&2
      return 1
      ;;
  esac
}

resolve_latest_step() {
  local checkpoint_dir="$1"
  local step
  local model_file
  local best_step=-1
  for model_file in "$checkpoint_dir"/model_*.pt; do
    [[ -f "$model_file" ]] || continue
    step="${model_file##*/model_}"
    step="${step%.pt}"
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    [[ -f "$checkpoint_dir/meta_${step}.json" ]] || continue
    if (( step > best_step )); then
      best_step="$step"
    fi
  done
  if (( best_step >= 0 )); then
    printf '%s\n' "$best_step"
    return 0
  fi
  echo "ERROR: no complete model/meta checkpoint found in $checkpoint_dir" >&2
  return 1
}
