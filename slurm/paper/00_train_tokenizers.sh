#!/usr/bin/env bash
#SBATCH --job-name=cobpe-tokenizers
#SBATCH --mem=256g
#SBATCH -c16
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

mkdir -p logs
LOCAL_PARQUET_DIR="$(stage_dir_to_local "$LOCAL_PARQUET_DIR" base_data_climbmix)"

TOK_VOCAB_SIZE="${TOK_VOCAB_SIZE:-32768}"
TOK_VOCAB_BUFFER_SIZE="${TOK_VOCAB_BUFFER_SIZE:-2048}"
TOK_MAX_CHARS="${TOK_MAX_CHARS:-2000000000}"
TOK_DOC_CAP="${TOK_DOC_CAP:-10000}"
TOK_SEED="${TOK_SEED:-42}"

uv run --no-sync python -m scripts.tok_train \
  --output-base-dir "$DATA_BASE_DIR" \
  --local-parquet-dir "$LOCAL_PARQUET_DIR" \
  --vocab-size "$TOK_VOCAB_SIZE" \
  --max-chars "$TOK_MAX_CHARS" \
  --doc-cap "$TOK_DOC_CAP" \
  --seed "$TOK_SEED" \
  --tokenizer-algorithm bpe \
  --tokenizer-dirname "$(basename "$TOKENIZER_BPE_DIR")"

uv run --no-sync python -m scripts.tok_train \
  --output-base-dir "$DATA_BASE_DIR" \
  --local-parquet-dir "$LOCAL_PARQUET_DIR" \
  --vocab-size "$TOK_VOCAB_SIZE" \
  --vocab-buffer-size "$TOK_VOCAB_BUFFER_SIZE" \
  --max-chars "$TOK_MAX_CHARS" \
  --doc-cap "$TOK_DOC_CAP" \
  --seed "$TOK_SEED" \
  --normalization-variant space_cap \
  --case-normalization all_letters_except_all_caps \
  --tokenizer-algorithm bpe \
  --tokenizer-dirname "$(basename "$TOKENIZER_COBPE_DIR")"

# The CoBPE tokenizer is intentionally still TOK_VOCAB_SIZE +
# TOK_VOCAB_BUFFER_SIZE here. Job 01 must select decomposition survivors,
# replace removed tokens from the buffer, and publish the exact runtime vocab.
