"""
Train model. From root directory of the project, run as:

python -m scripts.base_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
from dataclasses import asdict
from contextlib import contextmanager

import wandb
import torch
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig, Linear
from nanochat.modifier_cli import add_modifier_training_args
from nanochat.lr_schedule import (
    LR_SCHEDULE_CHOICES,
    build_lr_schedule_info,
    get_lr_multiplier_for_step,
    get_muon_momentum_for_step,
)
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanochat.tokenizer import get_tokenizer, get_token_bytes, get_tokenizer_fingerprint
from nanochat.checkpoint_manager import (
    save_checkpoint,
    load_checkpoint,
    patch_missing_model_keys,
    validate_checkpoint_tokenizer_fingerprint,
    get_exact_resume_compatibility_mismatches,
    list_checkpoint_steps,
    load_checkpoint_meta,
    is_checkpoint_step_complete,
)
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.generation import decode_generated_batch, encode_prompt
from nanochat.flash_attention import HAS_FA3
from cobpe.integrations.nanochat.runtime import configure_output_base_dir
from scripts.base_eval import evaluate_core
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", "--wandb-run", dest="run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
parser.add_argument("--output-base-dir", type=str, default="", help="root directory for checkpoints/eval artifacts (overrides NANOCHAT_BASE_DIR)")
parser.add_argument("--local-parquet-dir", type=str, default="", help="directory containing raw parquet shards for train/val")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU and torchao)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--model-dim", type=int, default=-1, help="explicit hidden size override (-1 = auto from depth/aspect-ratio)")
parser.add_argument("--n-head", type=int, default=-1, help="explicit number of query heads override (-1 = auto)")
parser.add_argument("--n-kv-head", type=int, default=-1, help="explicit number of KV heads override (-1 = n-head)")
parser.add_argument("--mlp-dim", type=int, default=-1, help="explicit MLP intermediate size override (-1 = 4*hidden)")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--architecture-preset", type=str, default="speedrun", choices=["speedrun", "speedrun_cobpe", "vanilla"], help="architecture preset controlling model-side speedrun tricks")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=quarter context (e.g. 'SSL')")
parser.add_argument(
    "--cobpe-smear-backout-scope",
    type=str,
    default=None,
    choices=["full", "base"],
    help="for CoBPE, apply smear/backout to the full representation or base-token stream only (default comes from architecture preset)",
)
add_modifier_training_args(parser)
parser.add_argument("--require-cobpe", action="store_true", help="fail unless the tokenizer is compositional/CoBPE")
parser.add_argument("--forbid-cobpe", action="store_true", help="fail if the tokenizer is compositional/CoBPE")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=10.5, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.008, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.28, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--warmup-steps", type=int, default=40, help="number of steps for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="final LR as fraction of initial LR")
parser.add_argument("--muon-final-momentum", type=float, default=0.90, help="Muon momentum target at the end of warmdown for linear_warmdown schedule")
parser.add_argument(
    "--lr-schedule",
    type=str,
    default="linear_warmdown",
    choices=LR_SCHEDULE_CHOICES,
    help="learning-rate schedule: default linear warmdown, or DeepSeek-style 80/10/10 multi-step decay",
)
def parse_resume_from_step(value):
    if str(value).lower() == "auto":
        return "auto"
    return int(value)

parser.add_argument(
    "--resume-from-step",
    type=parse_resume_from_step,
    default=-1,
    help="resume from an exact step, or 'auto' for the newest compatible checkpoint (-1 = disable)",
)
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=80*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
args = parser.parse_args()
if args.require_cobpe and args.forbid_cobpe:
    raise ValueError("--require-cobpe and --forbid-cobpe are mutually exclusive")
configure_output_base_dir(args.output_base_dir)
user_config = vars(args).copy()  # for logging
# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project=os.environ.get("WANDB_PROJECT", "cobpe"), name=args.run, config=user_config)

# Flash Attention status
from nanochat.flash_attention import USE_FA3
using_fa3 = USE_FA3
if using_fa3:
    print0("✓ Using Flash Attention 3 (Hopper GPU detected), efficient, new and awesome.")
else:
    print0("!" * 80)
    if HAS_FA3 and COMPUTE_DTYPE != torch.bfloat16:
        print0(f"WARNING: Flash Attention 3 only supports bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback")
    else:
        print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    if args.window_pattern != "L":
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
tokenizer_fingerprint = get_tokenizer_fingerprint()
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")
compositional_mode = bool(
    hasattr(tokenizer, "has_compositional_mode") and tokenizer.has_compositional_mode()
)
if args.require_cobpe and not compositional_mode:
    raise RuntimeError("--require-cobpe was set, but the loaded tokenizer is plain BPE.")
if args.forbid_cobpe and compositional_mode:
    raise RuntimeError("--forbid-cobpe was set, but the loaded tokenizer is CoBPE/compositional.")
modifier_group_sizes = tuple(tokenizer.get_modifier_group_sizes()) if compositional_mode else ()
user_config["compositional_mode"] = compositional_mode
user_config["modifier_group_sizes"] = list(modifier_group_sizes)
if compositional_mode:
    backend_name = "rust" if getattr(tokenizer, "rust_backend", None) is not None else "python"
    print0(f"CoBPE compositional mode enabled with modifier groups: {list(modifier_group_sizes)}")
    print0(f"CoBPE tokenizer backend: {backend_name}")

# -----------------------------------------------------------------------------
# Initialize the Model

def _resolve_model_shape(depth: int, apply_overrides: bool):
    base_dim = depth * args.aspect_ratio
    default_model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    model_dim = args.model_dim if apply_overrides and args.model_dim > 0 else default_model_dim
    if model_dim <= 0:
        raise ValueError(f"model_dim must be > 0, got {model_dim}")

    if apply_overrides and args.n_head > 0:
        num_heads = args.n_head
        if model_dim % num_heads != 0:
            raise ValueError(
                f"Invalid head config: model_dim={model_dim} not divisible by n_head={num_heads}"
            )
        resolved_head_dim = model_dim // num_heads
        if args.head_dim > 0 and resolved_head_dim != args.head_dim:
            raise ValueError(
                "Conflicting head settings: "
                f"model_dim/n_head={resolved_head_dim} but --head-dim={args.head_dim}. "
                "Either omit --n-head, or set --head-dim to match."
            )
    else:
        if args.head_dim <= 0:
            raise ValueError(f"head_dim must be > 0, got {args.head_dim}")
        if model_dim % args.head_dim != 0:
            raise ValueError(
                f"Invalid head config: model_dim={model_dim} not divisible by head_dim={args.head_dim}"
            )
        num_heads = model_dim // args.head_dim
        resolved_head_dim = args.head_dim

    if apply_overrides and args.n_kv_head > 0:
        num_kv_heads = args.n_kv_head
    else:
        num_kv_heads = num_heads
    if num_kv_heads <= 0:
        raise ValueError(f"n_kv_head must be > 0, got {num_kv_heads}")
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"Invalid GQA config: n_head={num_heads} must be divisible by n_kv_head={num_kv_heads}"
        )

    n_inner = args.mlp_dim if apply_overrides and args.mlp_dim > 0 else 4 * model_dim
    if n_inner <= 0:
        raise ValueError(f"mlp_dim must be > 0, got {n_inner}")

    return {
        "base_dim": base_dim,
        "model_dim": model_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": resolved_head_dim,
        "n_inner": n_inner,
    }


def build_model_meta(depth, apply_overrides=True):
    """Build a model on meta device for a given depth (shapes/dtypes only, no data)."""
    resolved = _resolve_model_shape(depth, apply_overrides=apply_overrides)
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth,
        n_head=resolved["num_heads"],
        n_kv_head=resolved["num_kv_heads"],
        n_embd=resolved["model_dim"],
        n_inner=resolved["n_inner"],
        architecture_preset=args.architecture_preset,
        window_pattern=args.window_pattern,
        modifier_group_sizes=modifier_group_sizes,
        modifier_conditioning_mode=str(args.modifier_conditioning_mode),
        modifier_gate_mode=str(args.modifier_gate_mode),
        cobpe_smear_backout_scope=args.cobpe_smear_backout_scope,
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta

# Build the model, move to device, init the weights
model = build_model_meta(args.depth, apply_overrides=True) # 1) Build on meta device (only shapes/dtypes, no data)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device) # 2) All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights() # 3) All tensors get initialized

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resume_auto = args.resume_from_step == "auto"
resuming = args.resume_from_step != -1
print0("Resolved training output paths:")
print0(f"  output_base_dir={os.path.abspath(base_dir)}")
print0(f"  checkpoint_dir={os.path.abspath(checkpoint_dir)}")
print0(f"  model_tag={output_dirname}")
print0(f"  resume_request={args.resume_from_step}")
print0(f"  tokenizer_dir={os.environ.get('NANOCHAT_TOKENIZER_DIR', '<default>')}")
print0(f"  local_parquet_dir={args.local_parquet_dir or '<default>'}")
if resuming and not resume_auto:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(
        checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank
    )
    validate_checkpoint_tokenizer_fingerprint(meta_data)
    patched_resume_keys = patch_missing_model_keys(model_data, model_config)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after the copy
else:
    optimizer_data = None
    meta_data = None
    patched_resume_keys = []

# -----------------------------------------------------------------------------
# FP8 training initialization and management (this has to be done before torch.compile)

# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        # our custom fp8 is simpler than torchao, written for exact API compatibility
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
        # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        # Filter: dims must be divisible by 16 (FP8 hardware requirement) large enough
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            # Keep output projections in BF16. FP8 conversion of the shared LM
            # head or modifier heads changes the numerically sensitive loss path.
            if fqn == "lm_head" or (compositional_mode and fqn.startswith("cobpe.")):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            if min(mod.in_features, mod.out_features) < 128:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped = num_linear - num_fp8
        print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small)")

# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    import torch.nn as nn

    # Find all Float8Linear modules and their locations
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # No FP8 modules, nothing to do
        return

    # Swap Float8Linear -> Linear (our custom class that casts weights to match input dtype)
    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device=fp8_module.weight.device,
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# Compile the model

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe

# -----------------------------------------------------------------------------
# Scaling laws and muP extrapolations to determine the optimal training horizon, batch size, learning rates, weight decay.

# Get the parameter counts of our model
param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# 1) Use scaling laws to determine the optimal training horizon in tokens
# The compute-optimal models satisfy the Tokens:Params ratio of --target-param-data-ratio (derived experimentally via scaling laws analysis).
# We've already initialized the model so we have Params. Optimal Tokens is now simply target-param-data-ratio * Params
def get_scaling_params(m):
    # Scale the training horizon using transformer matrices and the base-token head.
    params_counts = m.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    return scaling_params
num_scaling_params = get_scaling_params(model)
target_tokens = int(args.target_param_data_ratio * num_scaling_params) # optimal tokens for the model we are about to train

# Our reference model is d12, this is where a lot of hyperparameters are tuned and then transfered to higher depths (muP style)
d12_ref = build_model_meta(12, apply_overrides=False) # creates the model on meta device
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref) # compute-optimal d12 training horizon in tokens (measured empirically)
B_REF = 2**19 # optimal batch size at d12 ~= 524,288 tokens (measured empirically)

# 2) Now that we have the token horizon, we can calculate the optimal batch size
# We follow the Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
# The optimal batch size grows as approximately D^0.383, so e.g. if D doubles from d12 to d24, B should grow by 2^0.383 ≈ 1.3x.
total_batch_size = args.total_batch_size # user-provided override is possible
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # clamp to nearest power of 2 for efficiency
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# 3) Knowing the batch size, we can now calculate a learning rate correction (bigger batch size allows higher learning rates)
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF # B/B_ref
if batch_ratio != 1.0:
    # SGD: linear scaling with batch size is standard (not used in nanochat)
    # AdamW: sqrt scaling is standard: η ∝ √(B/B_ref)
    # Muon: we will use the same scaling for Muon as for AdamW: η ∝ √(B/B_ref) (not studied carefully, assumption!)
    batch_lr_scale = batch_ratio ** 0.5 # η ∝ √(B/B_ref)
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

# 4) Knowing the batch size and the token horizon, we can now calculate the appropriate weight decay scaling
# We adopt the T_epoch framework from https://arxiv.org/abs/2405.13698
# Central idea of the paper is that T_epoch = B/(η·λ·D) should remain constant.
# Above, we used learning rate scaling η ∝ √(B/B_ref). So it's a matter of ~10 lines of math to derive that to keep T_epoch constant, we need:
# λ = λ_ref · √(B/B_ref) · (D_ref/D)
# Note that these papers study AdamW, *not* Muon. We are blindly following AdamW theory for scaling hoping it ~works for Muon too.
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
optimizer = model.setup_optimizer(
    # AdamW hyperparameters
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    # Muon hyperparameters
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
)

if resuming and not resume_auto:
    if patched_resume_keys:
        raise RuntimeError(
            "Checkpoint requires model-state migration. "
            f"Patched model keys {patched_resume_keys}, but optimizer resume is unsafe because parameter shapes changed."
        )
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
# GradScaler for fp16 training (bf16/fp32 don't need it — bf16 has the same exponent range as fp32)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Calculate the number of iterations we will train for and set up the various schedulers

# num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    # Override num_iterations to a specific value if given
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # Calculate the number of iterations from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # Calculate the number of iterations from the target param data ratio (the most common use case)
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations # the actual number of tokens we will train for
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}") # e.g. Chinchilla was ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

lr_schedule = build_lr_schedule_info(
    lr_schedule=args.lr_schedule,
    num_iterations=num_iterations,
    warmup_steps=args.warmup_steps,
    warmdown_ratio=args.warmdown_ratio,
    final_lr_frac=args.final_lr_frac,
)
warmup_iters = lr_schedule.warmup_iters
warmdown_iters = lr_schedule.warmdown_iters
fixed_lr_start_step = lr_schedule.fixed_lr_start_step
fixed_lr_end_step = lr_schedule.fixed_lr_end_step
print0(
    f"LR schedule: {args.lr_schedule} | warmup_iters={warmup_iters:,} "
    f"| fixed_lr_window=[{fixed_lr_start_step:,}, {fixed_lr_end_step:,}]"
)
if lr_schedule.decay_step_1 is not None:
    print0(
        f"LR decay milestones: step1={lr_schedule.decay_step_1:,} "
        f"(x{lr_schedule.decay_step_1_lr_frac:.4f}), "
        f"step2={lr_schedule.decay_step_2:,} (x{lr_schedule.decay_step_2_lr_frac:.4f})"
    )

resume_lr_schedule = {
    "type": args.lr_schedule,
    "warmup_iters": warmup_iters,
    "warmdown_iters": warmdown_iters,
    "fixed_lr_start_step": fixed_lr_start_step,
    "fixed_lr_end_step": fixed_lr_end_step,
    "decay_step_1": lr_schedule.decay_step_1,
    "decay_step_2": lr_schedule.decay_step_2,
    "decay_step_1_lr_frac": lr_schedule.decay_step_1_lr_frac,
    "decay_step_2_lr_frac": lr_schedule.decay_step_2_lr_frac,
    "muon_final_momentum": args.muon_final_momentum,
}

if resume_auto:
    discovered_steps = list_checkpoint_steps(checkpoint_dir)
    complete_steps = [
        step
        for step in discovered_steps
        if is_checkpoint_step_complete(
            checkpoint_dir,
            step,
            expected_world_size=ddp_world_size,
            require_optimizer=True,
        )
    ]
    print0(
        "Auto-resume checkpoint inventory: "
        f"discovered_steps={discovered_steps or '<none>'} | "
        f"complete_for_world_size_{ddp_world_size}={complete_steps or '<none>'}"
    )
    skipped_resume_steps = []
    for candidate_step in discovered_steps:
        if not is_checkpoint_step_complete(
            checkpoint_dir,
            candidate_step,
            expected_world_size=ddp_world_size,
            require_optimizer=True,
        ):
            continue
        candidate_meta = load_checkpoint_meta(checkpoint_dir, candidate_step)
        candidate_mismatches = get_exact_resume_compatibility_mismatches(
            meta_data=candidate_meta,
            expected_num_iterations=num_iterations,
            expected_lr_schedule=resume_lr_schedule,
            expected_ddp_world_size=ddp_world_size,
            expected_total_batch_size=total_batch_size,
            expected_device_batch_size=args.device_batch_size,
            expected_max_seq_len=args.max_seq_len,
            expected_train_data_signature_hash=None,
        )
        if candidate_mismatches:
            skipped_resume_steps.append((candidate_step, candidate_mismatches))
            continue
        args.resume_from_step = candidate_step
        meta_data = candidate_meta
        print0(f"Auto-resume selected compatible checkpoint at step {candidate_step}")
        break
    else:
        details = "\n".join(
            f"  - step {step}: {mismatches[0]}"
            for step, mismatches in skipped_resume_steps
        )
        print0(
            "WARNING: --resume-from-step auto found no compatible complete checkpoint; "
            "starting a fresh run."
            + (f"\nSkipped checkpoints:\n{details}" if details else "")
        )
        # The model was initialized before automatic checkpoint discovery, so
        # a missing compatible checkpoint can safely fall back to fresh
        # training. Keep explicit numeric resume requests strict above.
        resume_auto = False
        resuming = False
        args.resume_from_step = -1
        meta_data = None
        optimizer_data = None

    if resume_auto:
        model_data, optimizer_data, loaded_meta = load_checkpoint(
            checkpoint_dir,
            args.resume_from_step,
            device,
            load_optimizer=True,
            rank=ddp_rank,
        )
        validate_checkpoint_tokenizer_fingerprint(loaded_meta)
        patched_resume_keys = patch_missing_model_keys(model_data, model_config)
        orig_model.load_state_dict(model_data, strict=True, assign=True)
        del model_data
        meta_data = loaded_meta

if resuming:
    resume_mismatches = get_exact_resume_compatibility_mismatches(
        meta_data=meta_data,
        expected_num_iterations=num_iterations,
        expected_lr_schedule=resume_lr_schedule,
        expected_ddp_world_size=ddp_world_size,
        expected_total_batch_size=total_batch_size,
        expected_device_batch_size=args.device_batch_size,
        expected_max_seq_len=args.max_seq_len,
        expected_train_data_signature_hash=None,
    )
    if resume_mismatches:
        details = "\n".join(f"  - {mismatch}" for mismatch in resume_mismatches)
        raise RuntimeError(
            "Checkpoint is not compatible with the current training command; refusing unsafe resume:\n"
            + details
        )

if resume_auto:
    if patched_resume_keys:
        raise RuntimeError(
            "Auto-selected checkpoint requires model-state migration, but optimizer resume would be unsafe: "
            f"{patched_resume_keys}"
        )
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# Learning rate schedule (linear warmup, constant, linear warmdown)
def get_lr_multiplier(it):
    return get_lr_multiplier_for_step(it, lr_schedule)

# Momentum scheduler for Muon optimizer (warms up to 0.97, then decays to muon_final_momentum during LR warmdown)
def get_muon_momentum(it):
    return get_muon_momentum_for_step(it, lr_schedule, muon_final_momentum=args.muon_final_momentum)

# Weight decay scheduler for Muon optimizer (cosine decay to zero over the course of training)
def get_weight_decay(it):
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val after resume selection.
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
local_parquet_dir = args.local_parquet_dir.strip() or None
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer,
    args.device_batch_size,
    args.max_seq_len,
    split="train",
    device=device,
    resume_state_dict=dataloader_resume_state_dict,
    local_parquet_dir=local_parquet_dir,
    with_modifiers=compositional_mode,
)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer,
    args.device_batch_size,
    args.max_seq_len,
    split="val",
    device=device,
    local_parquet_dir=local_parquet_dir,
    with_modifiers=compositional_mode,
)
x, y, dataloader_state_dict = next(train_loader)

# -----------------------------------------------------------------------------
# Training loop

# Loop state (variables updated by the training loop)
if not resuming:
    step = 0
    val_bpb = None # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Go!
while True:
    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model):
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes, tokenizer=tokenizer)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model):
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = encode_prompt(tokenizer, prompt)
            with disable_fp8(orig_model):
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(decode_generated_batch(tokenizer, sample)[0])
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            optimizer.state_dict(), # optimizer state
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "num_iterations": num_iterations,
                "lr_schedule": {
                    "type": args.lr_schedule,
                    "warmup_iters": warmup_iters,
                    "warmdown_iters": warmdown_iters,
                    "fixed_lr_start_step": fixed_lr_start_step,
                    "fixed_lr_end_step": fixed_lr_end_step,
                    "decay_step_1": lr_schedule.decay_step_1,
                    "decay_step_2": lr_schedule.decay_step_2,
                    "decay_step_1_lr_frac": lr_schedule.decay_step_1_lr_frac,
                    "decay_step_2_lr_frac": lr_schedule.decay_step_2_lr_frac,
                    "muon_final_momentum": args.muon_final_momentum,
                },
                "model_config": model_config_kwargs,
                "tokenizer_fingerprint": tokenizer_fingerprint,
                "user_config": user_config, # inputs to the training script
                "training_env": {
                    "device_type": device_type,
                    "gpu_device_name": gpu_device_name if device_type == "cuda" else "",
                    "ddp_world_size": int(ddp_world_size),
                },
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        if compositional_mode:
            x_ids, x_mods = x
            y_ids, y_mods = y
            loss = model(x_ids, y_ids, modifier_ids=x_mods, target_modifier_ids=y_mods)
        else:
            loss = model(x, y)
        train_loss = loss.detach() # for logging
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
    # step the optimizer
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    if scaler is not None:
        scaler.unscale_(optimizer)
        # In distributed training, all ranks must agree on whether to skip the step.
        # Each rank may independently encounter inf/nan gradients, so we all-reduce
        # the found_inf flag (MAX = if any rank found inf, all ranks skip).
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item() # .item() is a CPU-GPU sync point
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging (CPU action only)
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        wandb_run.log(log_data)

    # state update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # Periodic collection avoids repeated scans of the persistent model graph.
    if first_step_of_run:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # immediately freeze all currently surviving objects and exclude them from GC
        gc.disable() # Collect explicitly at the interval below.
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very, very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Base model training", data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Scaling params ratio": total_batch_size * num_iterations / num_scaling_params,
        "DDP world size": ddp_world_size,
        "warmup_steps": warmup_iters,
        "lr_schedule": args.lr_schedule,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
