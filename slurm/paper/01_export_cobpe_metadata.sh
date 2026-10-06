#!/usr/bin/env bash
#SBATCH --job-name=cobpe-metadata
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

PREPOSITION_PROFILE="${PREPOSITION_PROFILE:-core}"
PREPOSITION_LIST=()
while IFS= read -r preposition; do
  PREPOSITION_LIST+=("$preposition")
done < <(preposition_list_for_profile "$PREPOSITION_PROFILE")

extra_flags=()
[[ "${DECOMPOSE_DEMO:-0}" == "1" ]] && extra_flags+=(--decompose-demonstrative-determiners)
[[ "${DECOMPOSE_QUANT:-0}" == "1" ]] && extra_flags+=(--decompose-quantifier-determiners)
[[ "${OVERWRITE_COBPE_METADATA:-0}" == "1" || "${OVERWRITE_COBPE_METADATA_CACHE:-0}" == "1" ]] && extra_flags+=(--overwrite --overwrite-metadata-cache)

uv run --no-sync python -m scripts.export_compositional_metadata \
  --tokenizer-dir "$TOKENIZER_COBPE_DIR" \
  --metadata-cache-dir "$TOKENIZER_CACHE_DIR" \
  --local-parquet-dir "$LOCAL_PARQUET_DIR" \
  --preposition-profile "$PREPOSITION_PROFILE" \
  --preposition-list "${PREPOSITION_LIST[@]}" \
  --multi-token-modifier-placement split_by_role \
  --decompose-articles \
  --decompose-possessive-determiners \
  --decompose-prepositions \
  --decompose-punctuation \
  --word-boundary-safety \
  "${extra_flags[@]+"${extra_flags[@]}"}"

# Export is also the mandatory CoBPE vocabulary-finalization step. It fails
# unless the published tokenizer, token_bytes, and metadata use TOK_VOCAB_SIZE.
