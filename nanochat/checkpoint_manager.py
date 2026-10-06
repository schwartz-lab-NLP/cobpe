"""
Utilities for saving and loading model/optim/state checkpoints.
"""
import os
import re
import glob
import json
import logging
import tempfile
from typing import Any, Optional
import torch

from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig, materialize_gpt_config_kwargs
from nanochat.tokenizer import get_tokenizer, get_tokenizer_fingerprint
from nanochat.common import setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)


def validate_checkpoint_tokenizer_fingerprint(meta_data, tokenizer_dir=None):
    """Reject a tokenizer that differs from the artifact used to train a checkpoint."""

    saved = meta_data.get("tokenizer_fingerprint") if isinstance(meta_data, dict) else None
    if saved is None:
        log0("Checkpoint has no tokenizer fingerprint; compatibility is limited to legacy shape checks.")
        return
    current = get_tokenizer_fingerprint(tokenizer_dir)
    saved_digest = saved.get("sha256") if isinstance(saved, dict) else str(saved)
    if saved_digest != current["sha256"]:
        raise ValueError(
            "Tokenizer fingerprint mismatch: "
            f"checkpoint={saved_digest} current={current['sha256']}. "
            "Set NANOCHAT_TOKENIZER_DIR to the exact tokenizer artifact used for training."
        )

def patch_missing_model_config_keys(model_config_kwargs):
    """Add default values for new config keys missing in old checkpoints."""
    mode = model_config_kwargs.get("modifier_conditioning_mode", "concat_gated_refine")
    has_modifier_groups = bool(model_config_kwargs.get("modifier_group_sizes", ()))
    if (
        has_modifier_groups
        and mode in {"concat_gated", "concat_gated_refine"}
        and "modifier_gate_mode" not in model_config_kwargs
    ):
        raise ValueError(
            "Unsupported legacy CoBPE checkpoint: modifier_gate_mode is missing for a gated modifier head. "
            "Its gate weights may have been eight-row parameters with only row 0 active, so scalar and "
            "per-group behavior cannot be inferred safely."
        )
    # Old models were trained with full context (no sliding window)
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"
        log0(f"Patching missing window_pattern in model config to 'L'")
    before_keys = set(model_config_kwargs.keys())
    materialized = materialize_gpt_config_kwargs(model_config_kwargs)
    # `modifier_refine_dim=0` was serialized by older training runs but has no
    # effect in the retained modifier heads.
    model_config_kwargs.pop("modifier_refine_dim", None)
    model_config_kwargs.update(materialized)
    added_keys = sorted(set(model_config_kwargs.keys()) - before_keys)
    if added_keys:
        log0(f"Patching missing architecture config keys in model config: {added_keys}")

def patch_missing_model_keys(model_data, model_config, prefix=""):
    """Add default values for new parameters that may be missing in old checkpoints."""
    n_layer = model_config.n_layer
    patched_keys = []

    def add_missing(key, value, message):
        full_key = f"{prefix}{key}"
        if full_key not in model_data:
            model_data[full_key] = value
            patched_keys.append(full_key)
            log0(message)

    # resid_lambdas defaults to 1.0 (identity scaling)
    if model_config.use_residual_scalars:
        add_missing("resid_lambdas", torch.ones(n_layer), "Patching missing resid_lambdas in model data to 1.0")
        # x0_lambdas defaults to 0.0 (disabled)
        add_missing("x0_lambdas", torch.zeros(n_layer), "Patching missing x0_lambdas in model data to 0.0")
    if model_config.use_smear:
        add_missing("smear_gate.weight", torch.zeros(1, 24), "Patching missing smear_gate.weight in model data to 0.0")
        add_missing("smear_lambda", torch.zeros(1), "Patching missing smear_lambda in model data to 0.0")
    if model_config.use_backout:
        add_missing("backout_lambda", 0.2 * torch.ones(1), "Patching missing backout_lambda in model data to 0.2")
    return patched_keys


def validate_modifier_checkpoint_compatibility(model_data, model_config, prefix=""):
    """Fail clearly when a checkpoint's modifier gate architecture differs."""
    if model_config.modifier_conditioning_mode not in {"concat_gated", "concat_gated_refine"}:
        return
    gate_key = f"{prefix}cobpe.gate.weight"
    gate_weight = model_data.get(gate_key)
    if gate_weight is None:
        return  # strict state loading will report a missing parameter
    expected_rows = 1 if model_config.modifier_gate_mode == "scalar" else len(model_config.modifier_group_sizes)
    expected_shape = (expected_rows, model_config.n_embd)
    if tuple(gate_weight.shape) != expected_shape:
        raise ValueError(
            "Unsupported modifier checkpoint architecture: "
            f"{gate_key} has shape {tuple(gate_weight.shape)}, but config requires {expected_shape} "
            f"for modifier_gate_mode={model_config.modifier_gate_mode!r}. "
            "Use the matching checkpoint config; gate weights are not reinterpreted."
        )


def validate_checkpoint_tokenizer_compatibility(tokenizer, model_config):
    """Require tokenizer token structure to match the model's modifier inputs."""
    model_group_sizes = tuple(int(size) for size in (model_config.modifier_group_sizes or ()))
    tokenizer_is_compositional = bool(
        hasattr(tokenizer, "has_compositional_mode") and tokenizer.has_compositional_mode()
    )
    model_is_compositional = bool(model_group_sizes)
    if tokenizer_is_compositional != model_is_compositional:
        expected = "a CoBPE tokenizer" if model_is_compositional else "a plain BPE tokenizer"
        actual = "a CoBPE tokenizer" if tokenizer_is_compositional else "a plain BPE tokenizer"
        raise ValueError(
            f"Checkpoint/tokenizer architecture mismatch: model requires {expected}, but loaded {actual}."
        )
    if model_is_compositional:
        if not hasattr(tokenizer, "get_modifier_group_sizes"):
            raise ValueError("Loaded CoBPE tokenizer does not expose modifier group sizes.")
        tokenizer_group_sizes = tuple(int(size) for size in tokenizer.get_modifier_group_sizes())
        if tokenizer_group_sizes != model_group_sizes:
            raise ValueError(
                "Checkpoint/tokenizer modifier group sizes differ: "
                f"model={model_group_sizes}, tokenizer={tokenizer_group_sizes}."
            )

def _patch_missing_config_keys(model_config_kwargs):
    return patch_missing_model_config_keys(model_config_kwargs)

def _patch_missing_keys(model_data, model_config):
    return patch_missing_model_keys(model_data, model_config)


def _read_step_marker(marker_path: str) -> Optional[int]:
    if not os.path.exists(marker_path):
        return None
    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
    except OSError:
        return None
    if raw == "" or not raw.isdigit():
        return None
    return int(raw)


def _meta_path_for_step(checkpoint_dir: str, step: int) -> str:
    return os.path.join(checkpoint_dir, f"meta_{step:06d}.json")


def _extract_saved_world_size_from_meta(meta_data: Any) -> Optional[int]:
    if not isinstance(meta_data, dict):
        return None
    training_env = meta_data.get("training_env")
    if not isinstance(training_env, dict):
        return None
    try:
        saved_world_size = int(training_env.get("ddp_world_size", -1))
    except (TypeError, ValueError):
        return None
    return saved_world_size if saved_world_size > 0 else None


def _try_load_meta_data_for_step(checkpoint_dir: str, step: int) -> Optional[dict]:
    meta_path = _meta_path_for_step(checkpoint_dir, step)
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return meta_data if isinstance(meta_data, dict) else None


def load_checkpoint_meta(checkpoint_dir: str, step: int) -> dict:
    meta_path = _meta_path_for_step(checkpoint_dir, step)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    if not isinstance(meta_data, dict):
        raise ValueError(f"Checkpoint metadata at {meta_path} is not a JSON object")
    return meta_data


def resolve_optimizer_shard_rank(
    meta_data: dict,
    *,
    requested_rank: int,
    current_world_size: int,
    allow_world_size_change: bool,
    canonical_rank: int = 0,
) -> int:
    if requested_rank < 0:
        raise ValueError(f"requested_rank must be >= 0, got {requested_rank}")
    if current_world_size <= 0:
        raise ValueError(f"current_world_size must be > 0, got {current_world_size}")
    if canonical_rank < 0:
        raise ValueError(f"canonical_rank must be >= 0, got {canonical_rank}")
    saved_world_size = _extract_saved_world_size_from_meta(meta_data)
    if saved_world_size is None:
        return requested_rank
    if saved_world_size != int(current_world_size):
        if not allow_world_size_change:
            raise ValueError(
                "Checkpoint world size differs from current world size and world-size resume is disabled: "
                f"checkpoint={saved_world_size} current={current_world_size}"
            )
        return canonical_rank
    if requested_rank >= saved_world_size:
        return canonical_rank
    return requested_rank


def ensure_sharded_optimizer_world_size_compatibility(
    *,
    saved_world_size: Optional[int],
    current_world_size: int,
) -> None:
    """
    Fail fast when attempting to restore world-size-sharded optimizer state across
    different DDP world sizes.

    DistMuonAdamW shards optimizer state tensors by world size (ZeRO-2 style), so
    loading a checkpoint shard from world size A into world size B is not shape-safe
    and can crash during optimizer.step().
    """
    if current_world_size <= 0:
        raise ValueError(f"current_world_size must be > 0, got {current_world_size}")
    if saved_world_size is None:
        return
    saved_world_size = int(saved_world_size)
    if saved_world_size <= 0:
        return
    if saved_world_size == int(current_world_size):
        return
    if saved_world_size > 1 or int(current_world_size) > 1:
        raise ValueError(
            "Checkpoint optimizer state is world-size sharded and cannot be resumed across a DDP topology "
            f"change: checkpoint_world_size={saved_world_size}, current_world_size={current_world_size}. "
            "This would produce incompatible optimizer-state shapes and unreliable optimization dynamics. "
            "Use the same world size as the checkpoint, or start a fresh run for the new topology."
        )


def is_checkpoint_step_complete(
    checkpoint_dir: str,
    step: int,
    *,
    expected_world_size: Optional[int] = None,
    require_optimizer: bool = True,
) -> bool:
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        return False
    if not require_optimizer:
        return True
    effective_world_size = expected_world_size
    if effective_world_size is None:
        meta_data = _try_load_meta_data_for_step(checkpoint_dir, step)
        effective_world_size = _extract_saved_world_size_from_meta(meta_data)
    if effective_world_size is not None:
        if effective_world_size <= 0:
            raise ValueError(f"expected_world_size must be > 0, got {effective_world_size}")
        for rank in range(effective_world_size):
            optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
            if not os.path.exists(optimizer_path):
                return False
        return True
    any_optimizer = glob.glob(os.path.join(checkpoint_dir, f"optim_{step:06d}_rank*.pt"))
    return len(any_optimizer) > 0


def list_checkpoint_steps(checkpoint_dir: str) -> list[int]:
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    steps = []
    for path in checkpoint_files:
        stem = os.path.basename(path)
        step_str = stem.removeprefix("model_").removesuffix(".pt")
        if step_str.isdigit():
            steps.append(int(step_str))
    return sorted(set(steps), reverse=True)


def find_last_complete_step(
    checkpoint_dir: str,
    *,
    expected_world_size: Optional[int] = None,
    require_optimizer: bool = True,
    prefer_markers: bool = True,
) -> Optional[int]:
    if not os.path.isdir(checkpoint_dir):
        return None
    candidates: list[tuple[int, bool]] = []
    if prefer_markers:
        marker_paths = [
            os.path.join(checkpoint_dir, "latest_step.txt"),
            os.path.join(checkpoint_dir, "final_step.txt"),
        ]
        for marker_path in marker_paths:
            step = _read_step_marker(marker_path)
            if step is not None:
                candidates.append((int(step), True))
    candidates.extend((int(step), False) for step in list_checkpoint_steps(checkpoint_dir))
    # Guarantee "latest complete checkpoint" semantics by sorting by step descending.
    # Markers are only used as a tie-breaker for equal steps.
    candidates.sort(key=lambda x: (x[0], 1 if x[1] else 0), reverse=True)
    seen: set[int] = set()
    for step, _is_marker in candidates:
        if step in seen:
            continue
        seen.add(step)
        if is_checkpoint_step_complete(
            checkpoint_dir,
            step,
            expected_world_size=expected_world_size,
            require_optimizer=require_optimizer,
        ):
            return step
    return None


def get_exact_resume_compatibility_mismatches(
    *,
    meta_data: dict,
    expected_num_iterations: int,
    expected_lr_schedule: dict,
    expected_ddp_world_size: int,
    expected_total_batch_size: int,
    expected_device_batch_size: int,
    expected_max_seq_len: int,
    expected_train_data_signature_hash: Optional[str],
    allow_world_size_change: bool = False,
    expected_global_microbatch_tokens: Optional[int] = None,
) -> list[str]:
    def _normalize_lr_schedule_meta(raw):
        if not isinstance(raw, dict):
            return raw
        normalized = dict(raw)
        normalized.setdefault("muon_final_momentum", 0.90)
        return normalized

    mismatches = []
    saved_num_iterations_raw = meta_data.get("num_iterations")
    try:
        saved_num_iterations = int(saved_num_iterations_raw)
    except (TypeError, ValueError):
        saved_num_iterations = -1
    if saved_num_iterations != int(expected_num_iterations):
        mismatches.append(
            f"num_iterations mismatch: checkpoint={saved_num_iterations} current={expected_num_iterations}"
        )
    saved_lr = _normalize_lr_schedule_meta(meta_data.get("lr_schedule"))
    expected_lr = _normalize_lr_schedule_meta(expected_lr_schedule)
    if saved_lr != expected_lr:
        mismatches.append("lr_schedule mismatch between checkpoint metadata and current command-derived schedule")
    training_env_raw = meta_data.get("training_env", {})
    training_env = training_env_raw if isinstance(training_env_raw, dict) else {}
    try:
        saved_world = int(training_env.get("ddp_world_size", -1))
    except (TypeError, ValueError):
        saved_world = -1
    world_size_changed = saved_world > 0 and saved_world != int(expected_ddp_world_size)
    if world_size_changed and not allow_world_size_change:
        mismatches.append(f"ddp_world_size mismatch: checkpoint={saved_world} current={expected_ddp_world_size}")
    elif saved_world <= 0:
        mismatches.append(
            f"checkpoint is missing valid training_env.ddp_world_size (got {saved_world}); cannot verify resume invariants"
        )
    try:
        saved_total_batch = int(meta_data.get("total_batch_size", -1))
    except (TypeError, ValueError):
        saved_total_batch = -1
    if saved_total_batch != int(expected_total_batch_size):
        mismatches.append(f"total_batch_size mismatch: checkpoint={saved_total_batch} current={expected_total_batch_size}")
    try:
        saved_device_batch = int(meta_data.get("device_batch_size", -1))
    except (TypeError, ValueError):
        saved_device_batch = -1
    try:
        saved_max_seq_len = int(meta_data.get("max_seq_len", -1))
    except (TypeError, ValueError):
        saved_max_seq_len = -1
    if world_size_changed and allow_world_size_change:
        if expected_global_microbatch_tokens is None or int(expected_global_microbatch_tokens) <= 0:
            mismatches.append(
                "expected_global_microbatch_tokens must be provided and > 0 when allow_world_size_change is enabled"
            )
        else:
            saved_global_microbatch_tokens = (
                saved_device_batch * saved_max_seq_len * saved_world
                if saved_device_batch > 0 and saved_max_seq_len > 0 and saved_world > 0
                else -1
            )
            if saved_global_microbatch_tokens != int(expected_global_microbatch_tokens):
                mismatches.append(
                    "global_microbatch_tokens mismatch under world-size change: "
                    f"checkpoint={saved_global_microbatch_tokens} current={int(expected_global_microbatch_tokens)} "
                    f"(checkpoint uses device_batch_size*max_seq_len*ddp_world_size = "
                    f"{saved_device_batch}*{saved_max_seq_len}*{saved_world})"
                )
    else:
        if saved_device_batch != int(expected_device_batch_size):
            mismatches.append(
                f"device_batch_size mismatch: checkpoint={saved_device_batch} current={expected_device_batch_size}"
            )
    if saved_max_seq_len != int(expected_max_seq_len):
        mismatches.append(f"max_seq_len mismatch: checkpoint={saved_max_seq_len} current={expected_max_seq_len}")
    saved_sig_hash = meta_data.get("train_data_signature_hash")
    if expected_train_data_signature_hash is not None:
        if saved_sig_hash is None:
            mismatches.append("checkpoint is missing train_data_signature_hash required for exact resume")
        elif str(saved_sig_hash) != str(expected_train_data_signature_hash):
            mismatches.append(
                f"train_data_signature_hash mismatch: checkpoint={saved_sig_hash} current={expected_train_data_signature_hash}"
            )
    return mismatches

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the model state parameters
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        logger.info(f"Saved model parameters to: {model_path}")
        # Save the metadata dict as json
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved metadata to: {meta_path}")
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if optimizer_data is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        torch.save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer state to: {optimizer_path}")
    if rank == 0:
        def write_step_marker(name):
            fd, tmp_path = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=checkpoint_dir)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as marker_file:
                    marker_file.write(f"{int(step)}\n")
                    marker_file.flush()
                    os.fsync(marker_file.fileno())
                os.replace(tmp_path, os.path.join(checkpoint_dir, name))
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        write_step_marker("latest_step.txt")
        final_training_step = meta_data.get("num_iterations") if isinstance(meta_data, dict) else None
        if final_training_step is not None and int(step) >= int(final_training_step):
            write_step_marker("final_step.txt")

def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    patch_missing_model_config_keys(model_config_kwargs)
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    patch_missing_model_keys(model_data, model_config)
    validate_modifier_checkpoint_compatibility(model_data, model_config)
    with torch.device("meta"):
        model = GPT(model_config)
    # Load the model state
    model.to_empty(device=device)
    model.init_weights() # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # Load the Tokenizer
    tokenizer = get_tokenizer()
    validate_checkpoint_tokenizer_fingerprint(meta_data)
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model config vocab size {model_config_kwargs['vocab_size']}"
    validate_checkpoint_tokenizer_compatibility(tokenizer, model_config)
    return model, tokenizer, meta_data


def find_largest_model(checkpoints_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    steps = list_checkpoint_steps(checkpoint_dir)
    if not steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return int(steps[0])

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data

def load_model(source, *args, **kwargs):
    if source != "base":
        raise ValueError(f"Unsupported checkpoint source {source!r}; only 'base' is available in this release.")
    model_dir = "base_checkpoints"
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)

def load_optimizer_state(source, device, rank, model_tag=None, step=None):
    """Load just the optimizer shard for a given rank, without re-loading the model."""
    if source != "base":
        raise ValueError(f"Unsupported checkpoint source {source!r}; only 'base' is available in this release.")
    model_dir = "base_checkpoints"
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    if model_tag is None:
        model_tag = find_largest_model(checkpoints_dir)
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        step = find_last_step(checkpoint_dir)
    optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
    if not os.path.exists(optimizer_path):
        log0(f"Optimizer checkpoint not found: {optimizer_path}")
        return None
    log0(f"Loading optimizer state from {optimizer_path}")
    optimizer_data = torch.load(optimizer_path, map_location=device)
    return optimizer_data
