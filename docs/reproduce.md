# Reproduce the paper experiments

This guide describes the public tokenizer and language-model workflows. It
assumes the environment from the [README](../README.md) is installed and the
required ClimbMix parquet shards are available locally.

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
`LOCAL_PARQUET_DIR`. Training and tokenizer scripts read local parquet shards. Download the desired
number of ClimbMix training shards plus the validation shard with:

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

The default CoBPE modifier head is `concat_gated_refine` with per-group gates
conditioned on the hidden state. Set `MODIFIER_CONDITIONING_MODE=base_bias` or
`MODIFIER_CONDITIONING_MODE=concat_gated` for those alternatives. The
`MODIFIER_GATE_MODE=scalar` setting selects one shared gate instead of the
default per-group gates. Model depth, experiment directory, batch size, and
world size can be overridden as shown above. Give each head or gate ablation a
distinct `EXP_DIR` and `MODEL_TAG` to keep its checkpoint separate.

The paper also evaluates a 28-layer, 1.3B-parameter variant trained on 26B
tokens. Use `MODEL_NAME=d28 DEPTH=28` and the same vanilla wrappers to select
that model scale.

## Nanochat speedrun comparison

The speedrun comparison uses a 24-layer model with hidden size 1536,
intermediate size 6144, head dimension 128, sequence length 2048, and 8 L40S
GPUs. Its device batch size is 16 and total batch size is 1,048,576 tokens.
The training horizon settings are 8.0 tokens per parameter for BPE and 6.5
for CoBPE; the different values set the training horizon used in the reported
comparison.
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
