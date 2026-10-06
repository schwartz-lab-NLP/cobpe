# Reproduce the paper experiments

This guide describes the public tokenizer and language-model workflows. It
assumes the environment from the [README](../README.md) is installed and the
commands below are run from the repository root.

## Configure data and outputs

Copy `.env.example` to `.env`, set paths for your machine, then export its
values in the shell that launches a job:

```bash
set -a
source .env
set +a
```

The Slurm wrappers and Python scripts use these variables:

```bash
export DATA_BASE_DIR=/path/to/experiments
export LOCAL_PARQUET_DIR=/path/to/climbmix-parquet
export TOKENIZER_CACHE_DIR="$DATA_BASE_DIR/tokenizer_cache"
export TOKENIZER_BPE_DIR="$DATA_BASE_DIR/tokenizer_bpe"
export TOKENIZER_COBPE_DIR="$DATA_BASE_DIR/tokenizer_cobpe"
```

The Slurm wrappers also accept `PARQUET_DATA_DIR` as an alias for
`LOCAL_PARQUET_DIR`. Training and tokenizer scripts read local parquet shards.
Download the desired number of training shards plus the validation shard with:

```bash
NANOCHAT_BASE_DIR="$DATA_BASE_DIR" \
LOCAL_PARQUET_DIR="$LOCAL_PARQUET_DIR" \
uv run python -m nanochat.dataset --data-dir "$LOCAL_PARQUET_DIR" --num-files 8
```

The example downloads eight training shards plus the validation shard, for
setup and smaller runs. Use a larger count for the paper training horizon;
`--num-files -1` downloads the full training set. Set `HF_TOKEN` if Hugging
Face access requires authentication. Set `WANDB_API_KEY` to log to Weights &
Biases; logging is disabled by default in the paper wrappers.

## Train the tokenizers

Train the BPE baseline and the normalized tokenizer used to build CoBPE:

```bash
bash slurm/paper/00_train_tokenizers.sh
```

Build the CoBPE metadata and finalize its vocabulary before training a CoBPE
model:

```bash
bash slurm/paper/01_export_cobpe_metadata.sh
```

The export step selects decomposition survivors, compacts the tokenizer to the
configured vocabulary size, regenerates its token-byte data, and writes
`compositional.json`. CoBPE model loading requires these finalized artifacts.
The default paper configuration uses the core preposition profile.

## Main matched-compute experiments

The paper's main comparison uses a 24-layer vanilla model, a maximum sequence
length of 2048, and 20 training tokens per parameter. The paper reports
8 L40S GPUs, 780M parameters, and 15.6B training tokens for this model.
The training horizon option is named `TARGET_PARAM_DATA_RATIO` in the wrapper
and `--target-param-data-ratio` in the Python CLI; these values specify tokens
per parameter.

```bash
EXP_DIR=vanilla/d24 MODEL_NAME=d24 DEPTH=24 \
TARGET_PARAM_DATA_RATIO=20 WORLD_SIZE=8 \
bash slurm/paper/10_train_eval_vanilla_bpe.sh
```

```bash
EXP_DIR=vanilla/d24 MODEL_NAME=d24 DEPTH=24 \
TARGET_PARAM_DATA_RATIO=20 WORLD_SIZE=8 \
bash slurm/paper/11_train_eval_vanilla_cobpe.sh
```

The default CoBPE head is `gated-refinement`. The released head uses one gate
per modifier group, conditioned on the hidden state. For the simpler heads,
set `MODIFIER_CONDITIONING_MODE=lexical-bias` or
`MODIFIER_CONDITIONING_MODE=gated-concat`. See the
[head options](modifier_base_conditioned_head.md#head-options) for CLI and
wrapper settings. Give each variant a distinct `EXP_DIR` and `MODEL_TAG`.

The paper also evaluates a 28-layer, 1.3B-parameter variant trained on 26B
tokens. Use `MODEL_NAME=d28 DEPTH=28` and the same vanilla wrappers to select
that model scale.

## SuperBPE baseline

The paper uses SuperBPE with transition point `t = 27k`. Its trainer requires a
separate Python environment containing the `tokenizers-superbpe` fork, which
is not installed by `uv sync`. Set `SUPERBPE_PYTHON_BIN` to that environment's
Python executable, then train the tokenizer on the same ClimbMix sample:

```bash
uv run python -m cobpe train-tokenizer \
  --output-base-dir "$DATA_BASE_DIR" \
  --local-parquet-dir "$LOCAL_PARQUET_DIR" \
  --tokenizer-dirname tokenizer_superbpe \
  --tokenizer-algorithm superbpe \
  --superbpe-transition-merges 27000 \
  --superbpe-python-bin "$SUPERBPE_PYTHON_BIN" \
  --vocab-size 32768 --max-chars 2000000000 --doc-cap 10000 --seed 42
```

Train and evaluate the 780M baseline with the vanilla BPE wrapper, selecting
the SuperBPE tokenizer and a separate output directory:

```bash
TOKENIZER_BPE_DIR="$DATA_BASE_DIR/tokenizer_superbpe" \
EXP_DIR=vanilla/superbpe_d24 MODEL_TAG=superbpe_vanilla_d24 \
MODEL_NAME=d24 DEPTH=24 TARGET_PARAM_DATA_RATIO=20 WORLD_SIZE=8 \
bash slurm/paper/10_train_eval_vanilla_bpe.sh
```

## Nanochat speedrun comparison

The speedrun comparison uses a 24-layer model with hidden size 1536,
intermediate size 6144, head dimension 128, sequence length 2048, and 8 L40S
GPUs. Its device batch size is 16 and total batch size is 1,048,576 tokens.
The paper measures training time to a CORE score of 25.65. The reported runs
stopped at horizons of 8.0 tokens per parameter for BPE and 6.5 for CoBPE.
The wrappers train to these fixed horizons and then evaluate; they do not stop
automatically at the CORE target.

The reported results reach CORE 25.91 for BPE and 26.00 for CoBPE, with 5.84B
and 4.74B training tokens, respectively. These measurements use the paper's
8 L40S GPU setup.

```bash
EXP_DIR=speedrun/d24 MODEL_NAME=d24 DEPTH=24 \
TARGET_PARAM_DATA_RATIO=8.0 DEVICE_BATCH_SIZE=16 \
TOTAL_BATCH_SIZE=1048576 WORLD_SIZE=8 \
bash slurm/paper/20_train_eval_speedrun_bpe.sh
```

```bash
EXP_DIR=speedrun/d24 MODEL_NAME=d24 DEPTH=24 \
TARGET_PARAM_DATA_RATIO=6.5 DEVICE_BATCH_SIZE=16 \
TOTAL_BATCH_SIZE=1048576 WORLD_SIZE=8 \
bash slurm/paper/21_train_eval_speedrun_cobpe.sh
```

The wrappers use the `speedrun` architecture preset for BPE and the
corresponding CoBPE preset for CoBPE. They train and then run CORE, sample, and
bits-per-byte evaluation. On a Slurm cluster, use `sbatch` instead of running
the wrappers directly.
Set `RUN_EVAL=0` to train without running the evaluation job at the end.

## Evaluation from a saved checkpoint

The wrappers evaluate the latest complete checkpoint by default. To evaluate a
specific checkpoint directly, set `NANOCHAT_TOKENIZER_DIR` to the matching
tokenizer and call the shared evaluation entry point:

```bash
NANOCHAT_TOKENIZER_DIR="$TOKENIZER_BPE_DIR" \
uv run torchrun --standalone --nproc_per_node=8 -m scripts.base_eval \
  --eval sample,bpb,downstream \
  --model-tag bpe_vanilla_d24 \
  --step <STEP> \
  --output-base-dir "$DATA_BASE_DIR/runs/vanilla/d24" \
  --local-parquet-dir "$LOCAL_PARQUET_DIR"
```

For CoBPE, use `NANOCHAT_TOKENIZER_DIR="$TOKENIZER_COBPE_DIR"` and the
matching CoBPE model tag. The evaluator checks the tokenizer metadata when it
loads a CoBPE checkpoint.

## Tokenizer compression

Measure bytes per token on the same held-out UTF-8 text file for both
finalized tokenizers:

```bash
for tokenizer in "$TOKENIZER_BPE_DIR" "$TOKENIZER_COBPE_DIR"; do
  uv run python -m cobpe evaluate-tokenizer \
    --tokenizer-dir "$tokenizer" --text-file /path/to/heldout.txt
done
```

The command reports bytes, tokens, bytes per token, and exact roundtrip fidelity
for each file. Repeat `--text-file` to evaluate multiple documents.

## Not included in this release

The full vocabulary-size and modifier-family sweep for Figure 1, the
CNN/DailyMail copying benchmark for Table 4, and the UD-derived multilingual
inventory extraction and evaluation for Table 5 are not included. See the
[paper](https://arxiv.org/abs/2610.05597) for their protocols and results.
