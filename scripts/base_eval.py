"""
Unified evaluation script for base models.

Supports evaluation modes (comma-separated):
  --eval core    : CORE metric (accuracy on ICL tasks)
  --eval bpb     : Bits per byte on train/val splits
  --eval sample  : Generate samples from the model
  --eval downstream : SuperBPE-style downstream suite
  --eval downstream_lm : Downstream EM-only subset (LM/open-ended tasks)

Default is sample,bpb,downstream

Examples:

    # Evaluate a HuggingFace model (e.g. GPT-2 124M) using 8 GPUs
    torchrun --nproc_per_node=8 -m scripts.base_eval --hf-path openai-community/gpt2

    # Evaluate a nanochat model (e.g. d24) using 8 GPUs
    torchrun --nproc_per_node=8 -m scripts.base_eval --model-tag d24 --device-batch-size=16

    # Quick/approximate evaluation using a single GPU
    python -m scripts.base_eval --model-tag d24 --device-batch-size=16 --max-per-task=100 --split-tokens=524288
"""
import os
import csv
import time
import json
import yaml
import shutil
import glob
import random
import zipfile
import tempfile
import argparse
import torch
import torch.distributed as dist

from cobpe.tokenization.encoding import EncodedSequence, TokenCodec
from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type, download_file_with_lock
from nanochat.tokenizer import HuggingFaceTokenizer, get_token_bytes
from nanochat.checkpoint_manager import load_model, find_largest_model, find_last_step
from nanochat.core_eval import evaluate_task
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.downstream_eval import (
    DownstreamEvalConfig,
    evaluate_downstream,
    get_downstream_specs,
)
from nanochat.loss_eval import evaluate_bpb, build_case_space_base_equiv_map
from nanochat.engine import Engine, KVCache, sample_next_token
from nanochat.generation import decode_generated_batch, encode_prompt

# -----------------------------------------------------------------------------
# HuggingFace loading utilities

class ModelWrapper:
    """Lightweight wrapper to give HuggingFace models a nanochat-compatible interface."""
    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids, targets=None, loss_reduction='mean'):
        logits = self.model(input_ids).logits
        if targets is None:
            return logits
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction
        )
        return loss

    def get_device(self):
        return next(self.model.parameters()).device


def load_hf_model(hf_path: str, device):
    """Load a HuggingFace model and tokenizer."""
    print0(f"Loading HuggingFace model from: {hf_path}")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(hf_path)
    model.to(device)
    model.eval()
    max_seq_len = 1024 if "gpt2" in hf_path else None
    model = ModelWrapper(model, max_seq_len=max_seq_len)
    tokenizer = HuggingFaceTokenizer.from_pretrained(hf_path)
    return model, tokenizer


def get_hf_token_bytes(tokenizer, device="cpu"):
    """Compute token_bytes tensor for a HuggingFace tokenizer."""
    vocab_size = tokenizer.tokenizer.get_vocab_size()
    token_bytes = torch.zeros(vocab_size, dtype=torch.int64, device=device)
    for token_id in range(vocab_size):
        token_str = tokenizer.tokenizer.decode([token_id])
        token_bytes[token_id] = len(token_str.encode('utf-8'))
    return token_bytes


def _coerce_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _infer_training_summary(meta, checkpoint_dir, step):
    meta = meta if isinstance(meta, dict) else {}
    loop_state = meta.get("loop_state", {}) if isinstance(meta.get("loop_state", {}), dict) else {}
    training_env = meta.get("training_env", {}) if isinstance(meta.get("training_env", {}), dict) else {}

    total_training_seconds = _coerce_float(loop_state.get("total_training_time"))
    total_training_hours = (
        total_training_seconds / 3600.0
        if total_training_seconds is not None and total_training_seconds >= 0
        else None
    )
    total_training_hours_str = (
        f"{total_training_hours:.2f}h"
        if total_training_hours is not None
        else "unknown"
    )

    gpu_type = str(training_env.get("gpu_device_name", "")).strip()
    if gpu_type == "":
        gpu_type = str(training_env.get("device_type", "")).strip() or "unknown"

    num_gpus = training_env.get("ddp_world_size")
    if not isinstance(num_gpus, int) or num_gpus <= 0:
        try:
            num_gpus = int(num_gpus)
        except (TypeError, ValueError):
            num_gpus = 0
    if num_gpus <= 0 and checkpoint_dir and step is not None:
        shard_pat = os.path.join(checkpoint_dir, f"optim_{int(step):06d}_rank*.pt")
        shard_count = len(glob.glob(shard_pat))
        if shard_count > 0:
            num_gpus = shard_count
    num_gpus = num_gpus if num_gpus > 0 else None

    if num_gpus is not None:
        gpu_setting = f"{gpu_type} x{num_gpus}"
    else:
        gpu_setting = gpu_type

    return {
        "total_training_seconds": total_training_seconds,
        "total_training_hours": total_training_hours,
        "total_training_hours_str": total_training_hours_str,
        "gpu_type": gpu_type,
        "num_gpus": num_gpus,
        "gpu_setting": gpu_setting,
    }

# -----------------------------------------------------------------------------
# CORE evaluation

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


def place_eval_bundle(file_path):
    """Unzip eval_bundle.zip and place it in the base directory."""
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"Placed eval_bundle directory at {eval_bundle_dir}")


def ensure_eval_bundle_dir() -> str:
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)
    return eval_bundle_dir


def evaluate_core(model, tokenizer, device, max_per_task=-1):
    """
    Evaluate a base model on the CORE benchmark.
    Returns dict with results, centered_results, core_metric, and normalized variants.
    """
    eval_bundle_dir = ensure_eval_bundle_dir()

    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']

    # Load random baseline values
    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row['Eval Task']
            random_baseline = row['Random baseline']
            random_baselines[task_name] = float(random_baseline)

    # Evaluate each task
    results = {}
    centered_results = {}
    normalized_results = {}
    normalized_centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task['label']
        task_meta = {
            'label': label,
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }
        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end='')

        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        # Shuffle for consistent subsampling when using max_per_task
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        accuracy, normalized_accuracy = evaluate_task(
            model,
            tokenizer,
            data,
            device,
            task_meta,
            return_normalized=True,
        )
        results[label] = accuracy
        normalized_results[label] = normalized_accuracy
        random_baseline = random_baselines[label]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        normalized_centered_result = (normalized_accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        normalized_centered_results[label] = normalized_centered_result
        elapsed = time.time() - start_time
        show_normalized = abs(normalized_accuracy - accuracy) > 1e-12
        log_parts = [f"accuracy: {accuracy:.4f}"]
        if show_normalized:
            log_parts.append(f"normalized_accuracy: {normalized_accuracy:.4f}")
        log_parts.append(f"centered: {centered_result:.4f}")
        if show_normalized:
            log_parts.append(f"normalized_centered: {normalized_centered_result:.4f}")
        log_parts.append(f"time: {elapsed:.2f}s")
        print0(" | ".join(log_parts))

    core_metric = sum(centered_results.values()) / len(centered_results)
    normalized_core_metric = sum(normalized_centered_results.values()) / len(normalized_centered_results)
    out = {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric,
        "normalized_results": normalized_results,
        "normalized_centered_results": normalized_centered_results,
        "normalized_core_metric": normalized_core_metric,
    }
    return out


def _trim_generated_ids(token_ids, stop_ids):
    if not stop_ids:
        return token_ids
    stop = set(int(x) for x in stop_ids)
    out = []
    for t in token_ids:
        tid = int(t)
        if tid in stop:
            break
        out.append(tid)
    return out


def _resolve_base_stop_ids(tokenizer, stop_ids=None):
    out = set(int(x) for x in (stop_ids or []))
    try:
        out.add(int(tokenizer.get_bos_token_id()))
    except Exception:
        pass
    try:
        out.add(int(tokenizer.encode_special("<|assistant_end|>")))
    except Exception:
        pass
    return out


def _truncate_prompt_for_model(prompt_tokens, max_seq_len):
    if isinstance(prompt_tokens, EncodedSequence):
        if max_seq_len is None or max_seq_len <= 0 or len(prompt_tokens) <= int(max_seq_len):
            return prompt_tokens
        return prompt_tokens.slice(-int(max_seq_len), None)
    if max_seq_len is None or max_seq_len <= 0:
        return [int(t) for t in prompt_tokens]
    toks = [int(t) for t in prompt_tokens]
    if len(toks) <= int(max_seq_len):
        return toks
    return toks[-int(max_seq_len) :]


def _infer_kv_cache_dtype(model, device):
    # Match runtime trunk dtype when possible.
    try:
        if hasattr(model, "cos") and torch.is_tensor(getattr(model, "cos")):
            return model.cos.dtype
    except Exception:
        pass
    try:
        if hasattr(model, "backbone") and hasattr(model.backbone, "cos"):
            cos = getattr(model.backbone, "cos")
            if torch.is_tensor(cos):
                return cos.dtype
    except Exception:
        pass
    return torch.bfloat16 if device.type == "cuda" else torch.float32


@torch.inference_mode()
def _generate_same_length_base_batch(
    model,
    tokenizer,
    prompt_rows,
    *,
    do_sample: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    num_return_sequences: int,
    stop_ids=None,
    suppress_token_ids=None,
    seeds=None,
    timing_out: dict = None,
):
    if not prompt_rows:
        return []
    prompt_len = len(prompt_rows[0])
    if any(len(r) != prompt_len for r in prompt_rows):
        raise ValueError("All prompt rows must have equal token length for batched base generation")

    codec = TokenCodec(tokenizer)
    prompt_rows = [codec.normalize(row) for row in prompt_rows]
    compositional_mode = codec.has_modifiers

    mcfg = model.config
    max_decode = max(0, int(mcfg.sequence_len) - int(prompt_len))
    steps = min(int(max_new_tokens), max_decode)
    n = max(int(num_return_sequences), 1)
    out = [[] for _ in prompt_rows]
    device = model.get_device()
    dtype = _infer_kv_cache_dtype(model, device)
    stop = _resolve_base_stop_ids(tokenizer, stop_ids=stop_ids)
    suppress = set(int(t) for t in (suppress_token_ids or []))
    filler_token = int(next(iter(stop))) if stop else int(tokenizer.get_bos_token_id())
    base_seeds = [int(seeds[i]) if seeds is not None else 42 + i * 1009 for i in range(len(prompt_rows))]

    # Off by default: run all requested return sequences in one expanded block.
    # If this OOMs, adaptive backoff below will shrink sample_count automatically.
    sample_block = n

    sample_start = 0
    while sample_start < n:
        sample_count = min(sample_block, n - sample_start)
        while True:
            try:
                t_gen0 = time.perf_counter()
                owners = []
                expanded_ids = []
                expanded_modifiers = []
                seed_rows = []
                for i, sequence in enumerate(prompt_rows):
                    for j in range(sample_count):
                        owners.append(i)
                        expanded_ids.append([int(t) for t in sequence.ids])
                        if compositional_mode:
                            expanded_modifiers.append([list(row) for row in sequence.modifiers])
                        seed_rows.append(base_seeds[i] + sample_start + j)
                bsz = len(expanded_ids)
                if bsz <= 0:
                    break

                kv_cache = KVCache(
                    batch_size=bsz,
                    num_heads=mcfg.n_kv_head,
                    seq_len=max(1, int(prompt_len) + int(steps)),
                    head_dim=mcfg.n_embd // mcfg.n_head,
                    num_layers=mcfg.n_layer,
                    device=device,
                    dtype=dtype,
                )
                ids = torch.tensor(expanded_ids, dtype=torch.long, device=device)
                if compositional_mode:
                    modifier_ids = torch.tensor(expanded_modifiers, dtype=torch.long, device=device)
                    logits, hidden = model.forward(
                        ids,
                        kv_cache=kv_cache,
                        modifier_ids=modifier_ids,
                        return_hidden=True,
                    )
                    hidden = hidden[:, -1:, :]
                else:
                    logits = model.forward(ids, kv_cache=kv_cache)
                logits = logits[:, -1, :]

                use_sampling = bool(do_sample and float(temperature) > 0.0)
                rngs = None
                if use_sampling:
                    rngs = []
                    for s in seed_rows:
                        rng = torch.Generator(device=device) if device.type == "cuda" else torch.Generator()
                        rng.manual_seed(int(s))
                        rngs.append(rng)

                stop_tensor = None
                if stop:
                    stop_tensor = torch.tensor(sorted(stop), dtype=torch.long, device=device)
                completed = torch.zeros((bsz,), dtype=torch.bool, device=device)
                lengths = torch.zeros((bsz,), dtype=torch.long, device=device)
                generated_buf = torch.empty((bsz, int(steps)), dtype=torch.long, device=device)
                generated_modifiers = (
                    torch.empty(
                        (bsz, int(steps), len(codec.default_modifier())),
                        dtype=torch.long,
                        device=device,
                    )
                    if compositional_mode
                    else None
                )
                for _ in range(steps):
                    step_logits = logits
                    if suppress:
                        step_logits = logits.clone()
                        step_logits[:, sorted(suppress)] = -float("Inf")
                    if use_sampling:
                        next_list = []
                        for i in range(bsz):
                            if bool(completed[i].item()):
                                next_list.append(filler_token)
                                continue
                            nxt = int(
                                sample_next_token(
                                    step_logits[i : i + 1],
                                    rngs[i],
                                    float(temperature),
                                    None if int(top_k) <= 0 else int(top_k),
                                    float(top_p),
                                )[0, 0].item()
                            )
                            next_list.append(nxt)
                        next_ids = torch.tensor(next_list, dtype=torch.long, device=device)
                    else:
                        next_ids = step_logits.argmax(dim=-1).to(torch.long)

                    next_modifiers = None
                    if compositional_mode:
                        modifier_logits = model.get_modifier_logits(hidden, next_ids.view(-1, 1))
                        sampled_groups = []
                        for group_logits in modifier_logits:
                            group_rows = []
                            for row_idx in range(bsz):
                                sampled = sample_next_token(
                                    group_logits[row_idx : row_idx + 1, -1, :],
                                    rngs[row_idx] if rngs is not None else None,
                                    float(temperature) if use_sampling else 0.0,
                                    None,
                                    1.0,
                                )
                                group_rows.append(sampled[0, 0])
                            sampled_groups.append(torch.stack(group_rows))
                        next_modifiers = torch.stack(sampled_groups, dim=-1)

                    active_mask = ~completed
                    if stop_tensor is not None:
                        stop_now = active_mask & (next_ids.unsqueeze(1) == stop_tensor.unsqueeze(0)).any(dim=1)
                    else:
                        stop_now = torch.zeros_like(completed)

                    emit_mask = active_mask & (~stop_now)
                    if bool(emit_mask.any()):
                        emit_rows = torch.nonzero(emit_mask, as_tuple=False).squeeze(1)
                        emit_pos = lengths[emit_rows]
                        generated_buf[emit_rows, emit_pos] = next_ids[emit_rows]
                        if generated_modifiers is not None:
                            generated_modifiers[emit_rows, emit_pos] = next_modifiers[emit_rows]
                        lengths[emit_rows] = emit_pos + 1

                    completed = completed | stop_now
                    if bool(torch.all(completed)):
                        break
                    still_active = ~completed
                    feed = torch.where(
                        still_active,
                        next_ids,
                        torch.full_like(next_ids, int(filler_token)),
                    )
                    if compositional_mode:
                        default_modifier = torch.tensor(
                            codec.default_modifier(), dtype=torch.long, device=device
                        ).view(1, -1)
                        feed_modifiers = torch.where(
                            still_active.view(-1, 1),
                            next_modifiers,
                            default_modifier.expand(bsz, -1),
                        )
                        logits, hidden = model.forward(
                            feed.view(bsz, 1),
                            kv_cache=kv_cache,
                            modifier_ids=feed_modifiers.view(bsz, 1, -1),
                            return_hidden=True,
                        )
                        hidden = hidden[:, -1:, :]
                        logits = logits[:, -1, :]
                    else:
                        logits = model.forward(feed.view(bsz, 1), kv_cache=kv_cache)[:, -1, :]
                t_gen_chunk = max(0.0, time.perf_counter() - t_gen0)

                t_dec0 = time.perf_counter()
                lengths_cpu = lengths.detach().cpu().tolist()
                max_len = int(max(lengths_cpu)) if lengths_cpu else 0
                decoded_rows = []
                gen_cpu = None
                if max_len > 0:
                    gen_cpu = generated_buf[:, :max_len].detach().cpu().numpy()
                    gen_modifiers_cpu = (
                        generated_modifiers[:, :max_len].detach().cpu().numpy()
                        if generated_modifiers is not None
                        else None
                    )
                for row_i, row_len in enumerate(lengths_cpu):
                    if int(row_len) <= 0:
                        decoded_rows.append("")
                        continue
                    row = [int(x) for x in gen_cpu[row_i, : int(row_len)].tolist()]
                    if gen_modifiers_cpu is None:
                        decoded_rows.append(tokenizer.decode(row))
                    else:
                        modifiers = [
                            [int(v) for v in values]
                            for values in gen_modifiers_cpu[row_i, : int(row_len)].tolist()
                        ]
                        decoded_rows.append(tokenizer.decode_with_modifiers(row, modifiers))
                t_dec_chunk = max(0.0, time.perf_counter() - t_dec0)
                for text, owner in zip(decoded_rows, owners):
                    out[owner].append(str(text))
                if timing_out is not None:
                    timing_out["generate_s"] = float(timing_out.get("generate_s", 0.0)) + float(t_gen_chunk)
                    timing_out["decode_s"] = float(timing_out.get("decode_s", 0.0)) + float(t_dec_chunk)
                break
            except torch.cuda.OutOfMemoryError:
                if device.type != "cuda":
                    raise
                torch.cuda.empty_cache()
                if sample_count <= 1:
                    raise
                new_count = max(1, sample_count // 2)
                print0(
                    "[downstream] OOM in base generation chunk; reducing sampled-return block "
                    f"from {sample_count} to {new_count} (prompt_len={prompt_len}, prompts={len(prompt_rows)})"
                )
                sample_count = new_count
        sample_start += sample_count
    return out


def _build_downstream_generate_fns(model, tokenizer, is_hf_model: bool):
    timing_by_task = {}

    def _timing_bucket(task_label):
        if not task_label:
            return None
        key = str(task_label)
        rec = timing_by_task.get(key)
        if rec is None:
            rec = {"generate_s": 0.0, "decode_s": 0.0, "calls": 0.0, "completions": 0.0}
            timing_by_task[key] = rec
        return rec

    def _pop_task_timing(task_label: str):
        if not task_label:
            return None
        return timing_by_task.pop(str(task_label), None)

    if is_hf_model:
        hf_model = model.model
        device = model.get_device()
        bos_id = tokenizer.get_bos_token_id()

        def _hf_generate_batch(
            prompts: list[str],
            *,
            do_sample: bool,
            max_new_tokens: int,
            temperature: float,
            top_p: float,
            top_k: int,
            num_return_sequences: int,
            stop_sequences=None,
            stop_ids=None,
            seeds=None,
            task_label=None,
        ):
            del stop_sequences  # String stop handling is done by downstream scorer.
            ids_rows = [tokenizer(p, prepend=bos_id) for p in prompts]
            if not ids_rows:
                return []
            max_len = max(len(r) for r in ids_rows)
            padded = []
            mask = []
            for row in ids_rows:
                pad = max_len - len(row)
                padded.append(row + [bos_id] * pad)
                mask.append([1] * len(row) + [0] * pad)
            input_ids = torch.tensor(padded, dtype=torch.long, device=device)
            attention_mask = torch.tensor(mask, dtype=torch.long, device=device)
            use_sampling = bool(do_sample and float(temperature) > 0.0)
            n = max(int(num_return_sequences), 1)
            top_k_value = None if int(top_k) <= 0 else int(top_k)
            top_p_value = float(top_p)
            if top_p_value <= 0.0 or top_p_value > 1.0:
                raise ValueError(f"Invalid top_p={top_p_value}; expected 0 < top_p <= 1")
            if top_p_value >= 1.0:
                top_p_value = 1.0
            seed0 = int(seeds[0]) if seeds else 42
            generator = torch.Generator(device=device) if device.type == "cuda" else torch.Generator()
            generator.manual_seed(seed0)

            with torch.no_grad():
                t_gen0 = time.perf_counter()
                if use_sampling:
                    gen_out = hf_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=int(max_new_tokens),
                        do_sample=True,
                        temperature=float(temperature),
                        top_p=top_p_value,
                        top_k=top_k_value if top_k_value is not None else 0,
                        num_return_sequences=n,
                        use_cache=True,
                        pad_token_id=bos_id,
                        generator=generator,
                    )
                    seqs = gen_out
                else:
                    gen_out = hf_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=int(max_new_tokens),
                        do_sample=False,
                        num_return_sequences=n,
                        use_cache=True,
                        pad_token_id=bos_id,
                    )
                    seqs = gen_out
                t_gen = max(0.0, time.perf_counter() - t_gen0)

            out = [[] for _ in prompts]
            t_dec0 = time.perf_counter()
            for row_idx in range(len(prompts)):
                prompt_len = int(attention_mask[row_idx].sum().item())
                for sample_idx in range(n):
                    seq = seqs[row_idx * n + sample_idx]
                    gen_ids = seq[prompt_len:].detach().cpu().tolist()
                    gen_ids = _trim_generated_ids(gen_ids, stop_ids=stop_ids)
                    out[row_idx].append(tokenizer.decode(gen_ids))
            t_dec = max(0.0, time.perf_counter() - t_dec0)
            rec = _timing_bucket(task_label)
            if rec is not None:
                rec["generate_s"] += float(t_gen)
                rec["decode_s"] += float(t_dec)
                rec["calls"] += 1.0
                rec["completions"] += float(len(prompts) * n)
            return out

        def _hf_generate(prompt: str, **kwargs):
            return _hf_generate_batch([prompt], seeds=[int(kwargs.pop("seed"))], **kwargs)[0]

        return _hf_generate, _hf_generate_batch, _pop_task_timing

    def _base_generate_batch(
        prompts: list[str],
        *,
        do_sample: bool,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        num_return_sequences: int,
        stop_sequences=None,
        stop_ids=None,
        suppress_token_ids=None,
        seeds=None,
        task_label=None,
    ):
        del stop_sequences  # String stop handling is done by downstream scorer.
        if not prompts:
            return []
        if float(top_p) <= 0.0 or float(top_p) > 1.0:
            raise ValueError(f"Invalid top_p={top_p}; expected 0 < top_p <= 1")

        model_seq_len = int(getattr(model, "config").sequence_len)
        tokenized = [
            _truncate_prompt_for_model(
                encode_prompt(tokenizer, p),
                model_seq_len,
            )
            for p in prompts
        ]
        groups = {}
        for i, toks in enumerate(tokenized):
            groups.setdefault(len(toks), []).append(i)
        out = [[] for _ in prompts]
        generate_s_total = 0.0
        decode_s_total = 0.0
        n = max(int(num_return_sequences), 1)
        for _plen, idxs in groups.items():
            cursor = 0
            window = len(idxs)
            while cursor < len(idxs):
                cur = min(window, len(idxs) - cursor)
                while True:
                    chunk_idxs = idxs[cursor : cursor + cur]
                    prompt_rows = [tokenized[i] for i in chunk_idxs]
                    seed_rows = [int(seeds[i]) for i in chunk_idxs] if seeds is not None else None
                    try:
                        timing_chunk = {"generate_s": 0.0, "decode_s": 0.0}
                        chunk_out = _generate_same_length_base_batch(
                            model,
                            tokenizer,
                            prompt_rows,
                            do_sample=bool(do_sample),
                            max_new_tokens=int(max_new_tokens),
                            temperature=float(temperature),
                            top_p=float(top_p),
                            top_k=int(top_k),
                            num_return_sequences=int(num_return_sequences),
                            stop_ids=stop_ids,
                            suppress_token_ids=suppress_token_ids,
                            seeds=seed_rows,
                            timing_out=timing_chunk,
                        )
                        generate_s_total += float(timing_chunk.get("generate_s", 0.0))
                        decode_s_total += float(timing_chunk.get("decode_s", 0.0))
                        for local_i, src_i in enumerate(chunk_idxs):
                            out[src_i] = chunk_out[local_i]
                        break
                    except torch.cuda.OutOfMemoryError:
                        if model.get_device().type != "cuda":
                            raise
                        torch.cuda.empty_cache()
                        if cur <= 1:
                            raise
                        new_cur = max(1, cur // 2)
                        print0(
                            "[downstream] OOM in base prompt batch; reducing prompt chunk "
                            f"from {cur} to {new_cur} (prompt_len={_plen})"
                        )
                        cur = new_cur
                cursor += cur
        rec = _timing_bucket(task_label)
        if rec is not None:
            rec["generate_s"] += float(generate_s_total)
            rec["decode_s"] += float(decode_s_total)
            rec["calls"] += 1.0
            rec["completions"] += float(len(prompts) * n)
        return out

    def _base_generate(prompt: str, **kwargs):
        seed = int(kwargs.pop("seed"))
        return _base_generate_batch([prompt], seeds=[seed], **kwargs)[0]

    return _base_generate, _base_generate_batch, _pop_task_timing

# -----------------------------------------------------------------------------
# Main

def main():
    parser = argparse.ArgumentParser(description="Base model evaluation")
    parser.add_argument('--eval', type=str, default='sample,bpb,downstream', help='Comma-separated evaluations to run: core,bpb,sample,downstream,downstream_lm (default: sample,bpb,downstream)')
    parser.add_argument('--hf-path', type=str, default=None, help='HuggingFace model path (e.g. openai-community/gpt2-xl)')
    parser.add_argument('--model-tag', type=str, default=None, help='nanochat model tag to identify the checkpoint directory')
    parser.add_argument('--step', type=int, default=None, help='Model step to load (default = last)')
    parser.add_argument('--max-per-task', type=int, default=-1, help='Max examples per CORE task (-1 = all)')
    parser.add_argument('--device-batch-size', type=int, default=32, help='Per-device batch size for BPB evaluation')
    parser.add_argument('--split-tokens', type=int, default=40*524288, help='Number of tokens to evaluate per split for BPB')
    parser.add_argument('--device-type', type=str, default='', help='cuda|cpu|mps (empty = autodetect)')
    parser.add_argument("--output-base-dir", type=str, default="", help="root directory for checkpoints/eval artifacts (overrides NANOCHAT_BASE_DIR)")
    parser.add_argument("--local-parquet-dir", type=str, default="", help="directory containing raw parquet shards for BPB eval")
    parser.add_argument('--bpb-accuracy', action='store_true',
                        help='also compute BPB surface/base token accuracy metrics (higher eval overhead)')
    parser.add_argument(
        '--downstream-preset',
        type=str,
        default='',
        choices=[
            '',
            'sample_200_no_code_math',
            'full_no_code_math',
            'sample_5000_all_30',
            'sample_5000_skip_repeat_copy_logic_gsm8k_hotpotqa_humaneval_mbpp',
        ],
        help='Named downstream configuration; all presets default to 5000 examples/task except sample_200_no_code_math',
    )
    parser.add_argument(
        '--downstream-max-per-task',
        type=int,
        default=None,
        help='Max examples per downstream task (preset defaults: 200 for sample_200_no_code_math; otherwise 5000; -1 means uncapped)',
    )
    parser.add_argument('--downstream-task-filter', type=str, default='all', help='Comma-separated downstream task labels to run, or all')
    parser.add_argument('--downstream-task-exclude-filter', type=str, default='', help='Comma-separated downstream task labels to exclude')
    parser.add_argument('--downstream-sample-seed', type=int, default=1337,
                        help='Seed used to shuffle each task before applying the per-task example cap (Phoenix-compatible default: 1337)')
    parser.add_argument('--downstream-task-results-dir', type=str, default='',
                        help='Directory for per-task JSON results and an incrementally flushed task_summary.csv')
    parser.add_argument('--downstream-em-max-new-tokens', type=int, default=64, help='Generation budget for downstream EM tasks')
    parser.add_argument('--downstream-em-temperature', type=float, default=0.0, help='Sampling temperature for downstream EM generation')
    parser.add_argument('--downstream-em-top-p', type=float, default=1.0, help='Top-p for downstream EM generation')
    parser.add_argument('--downstream-em-top-k', type=int, default=0, help='Top-k for downstream EM generation (<=0 disables)')
    parser.add_argument('--downstream-em-batch-size', type=int, default=8, help='Prompt batch size for downstream EM generation')
    parser.add_argument('--downstream-coding-num-samples', type=int, default=20, help='Number of sampled completions for downstream coding tasks')
    parser.add_argument('--downstream-coding-pass-k', type=int, default=10, help='pass@k target for downstream coding tasks')
    parser.add_argument('--downstream-coding-temperature', type=float, default=0.8, help='Sampling temperature for downstream coding tasks')
    parser.add_argument('--downstream-coding-top-p', type=float, default=0.95, help='Top-p for downstream coding tasks')
    parser.add_argument('--downstream-coding-top-k', type=int, default=0, help='Top-k for downstream coding tasks (<=0 disables)')
    parser.add_argument('--downstream-coding-max-new-tokens', type=int, default=128, help='Generation budget for downstream coding tasks')
    parser.add_argument('--downstream-coding-batch-size', type=int, default=2, help='Prompt batch size for downstream coding generation')
    parser.add_argument('--downstream-verbose', action=argparse.BooleanOptionalAction, default=False, help='Verbose downstream task progress logging')
    parser.add_argument('--downstream-log-every', type=int, default=100, help='Rank-0 shard progress cadence for downstream EM/pass@10 tasks')
    args = parser.parse_args()
    downstream_exclusions = [
        x.strip() for x in str(args.downstream_task_exclude_filter or '').split(',') if x.strip()
    ]
    downstream_default_max = 5000
    if args.downstream_preset in {
        'sample_200_no_code_math',
        'full_no_code_math',
        'sample_5000_skip_repeat_copy_logic_gsm8k_hotpotqa_humaneval_mbpp',
    }:
        downstream_exclusions.extend([
            'Repeat-Copy-Logic', 'GSM8K', 'HotpotQA', 'HumanEval', 'MBPP',
        ])
    if args.downstream_preset == 'sample_200_no_code_math':
        downstream_default_max = 200
    args.downstream_max_per_task = (
        downstream_default_max
        if args.downstream_max_per_task is None
        else int(args.downstream_max_per_task)
    )
    args.downstream_task_exclude_filter = ','.join(dict.fromkeys(downstream_exclusions))
    if args.output_base_dir.strip():
        os.environ["NANOCHAT_BASE_DIR"] = os.path.abspath(os.path.expanduser(args.output_base_dir))

    # Parse evaluation modes
    eval_modes = set(mode.strip() for mode in args.eval.split(','))
    valid_modes = {'core', 'bpb', 'sample', 'downstream', 'downstream_lm'}
    invalid = eval_modes - valid_modes
    if invalid:
        parser.error(f"Invalid eval modes: {invalid}. Valid: {valid_modes}")

    # Distributed / precision setup
    device_type = autodetect_device_type() if args.device_type == '' else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    # Load model and tokenizer
    is_hf_model = args.hf_path is not None
    resolved_model_tag = None
    resolved_step = None
    loaded_meta = {}
    loaded_checkpoint_dir = None
    if is_hf_model:
        model, tokenizer = load_hf_model(args.hf_path, device)
        sequence_len = model.max_seq_len or 1024
        token_bytes = get_hf_token_bytes(tokenizer, device=device)
        model_name = args.hf_path
        model_slug = args.hf_path.replace("/", "-")
    else:
        base_checkpoints_dir = os.path.join(get_base_dir(), "base_checkpoints")
        resolved_model_tag = args.model_tag if args.model_tag else find_largest_model(base_checkpoints_dir)
        loaded_checkpoint_dir = os.path.join(base_checkpoints_dir, resolved_model_tag)
        resolved_step = args.step
        if resolved_step is None:
            resolved_step = find_last_step(loaded_checkpoint_dir)
        model, tokenizer, meta = load_model(
            "base",
            device,
            phase="eval",
            model_tag=resolved_model_tag,
            step=resolved_step,
        )
        loaded_meta = meta
        resolved_step = int(meta.get("step", resolved_step))
        sequence_len = meta["model_config"]["sequence_len"]
        token_bytes = get_token_bytes(device=device)
        model_name = f"base_model ({resolved_model_tag}, step {resolved_step})"
        model_slug = f"{resolved_model_tag}_{resolved_step:06d}"

    downstream_modes = [m for m in ("downstream", "downstream_lm") if m in eval_modes]
    task_result_writer = None
    if downstream_modes and ddp_rank == 0:
        task_results_dir = args.downstream_task_results_dir.strip()
        if task_results_dir:
            task_results_dir = os.path.abspath(os.path.expanduser(task_results_dir))
        else:
            task_results_dir = os.path.join(
                get_base_dir(), "base_eval", f"{model_slug}.downstream_tasks"
            )
        if os.path.exists(task_results_dir) and os.listdir(task_results_dir):
            raise FileExistsError(
                f"Refusing to mix or overwrite downstream task outputs in non-empty directory: {task_results_dir}"
            )
        os.makedirs(task_results_dir, exist_ok=True)
        task_summary_csv = os.path.join(task_results_dir, "task_summary.csv")

        def task_result_writer(result):
            suite = str(result["suite"])
            task = str(result["task"])
            slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in task).strip("_")
            while "__" in slug:
                slug = slug.replace("__", "_")
            suite_dir = os.path.join(task_results_dir, suite)
            os.makedirs(suite_dir, exist_ok=True)
            result_path = os.path.join(suite_dir, f"{slug}.json")
            if os.path.exists(result_path):
                raise FileExistsError(f"Refusing to overwrite completed task result: {result_path}")
            result_record = {
                "model_tag": args.hf_path if is_hf_model else resolved_model_tag,
                "step": resolved_step,
                **result,
            }
            fd, temp_path = tempfile.mkstemp(prefix=f".{slug}.", suffix=".tmp", dir=suite_dir)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as result_file:
                    json.dump(result_record, result_file, ensure_ascii=False, indent=2)
                    result_file.write("\n")
                    result_file.flush()
                    os.fsync(result_file.fileno())
                os.replace(temp_path, result_path)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

            csv_fields = [
                "model_tag", "step", "suite", "task", "category", "metric",
                "source_kind", "score", "num_examples", "elapsed_seconds", "max_per_task", "sample_seed", "arithmetic_splits", "preset", "task_exclude_filter",
            ]
            needs_header = not os.path.exists(task_summary_csv) or os.path.getsize(task_summary_csv) == 0
            with open(task_summary_csv, "a", encoding="utf-8", newline="") as summary_file:
                writer = csv.DictWriter(summary_file, fieldnames=csv_fields)
                if needs_header:
                    writer.writeheader()
                row = dict(result_record)
                row["arithmetic_splits"] = json.dumps(row.get("arithmetic_splits"), sort_keys=True)
                writer.writerow({field: row.get(field) for field in csv_fields})
                summary_file.flush()
                os.fsync(summary_file.fileno())
            print0(f"Per-task result written: {result_path}")

    print0(f"Evaluating model: {model_name}")
    print0(f"Eval modes: {', '.join(sorted(eval_modes))}")
    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print0(
        f"Distributed eval setup: world_size={ddp_world_size} "
        f"device={device} cuda_visible_count={cuda_count}"
    )
    # Results to log
    core_results = None
    downstream_results_by_suite = {}
    bpb_results = {}
    samples = []
    unconditioned_samples = []

    # --- Sampling ---
    if 'sample' in eval_modes and not is_hf_model:
        print0("\n" + "="*80)
        print0("Model Samples")
        print0("="*80)
        if ddp_rank == 0:
            prompts = [
                "The capital of France is",
                "The chemical symbol of gold is",
                "If yesterday was Friday, then tomorrow will be",
                "The opposite of hot is",
                "The planets of the solar system are:",
                "My favorite color is",
                "If 5*x + 3 = 13, then x is",
            ]
            engine = Engine(model, tokenizer)
            print0("\nConditioned samples:")
            for prompt in prompts:
                tokens = encode_prompt(tokenizer, prompt)
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
                sample_str = decode_generated_batch(tokenizer, sample)[0]
                print0("-" * 80)
                print0(sample_str)
                samples.append(sample_str)

            print0("\nUnconditioned samples:")
            tokens = encode_prompt(tokenizer, "")
            uncond, _ = engine.generate_batch(tokens, num_samples=8, max_tokens=128, temperature=1.0)
            for sample_str in decode_generated_batch(tokenizer, uncond):
                print0("-" * 80)
                print0(sample_str)
                unconditioned_samples.append(sample_str)
    elif 'sample' in eval_modes and is_hf_model:
        print0("\nSkipping sampling for HuggingFace models (not supported)")

    # --- BPB evaluation ---
    if 'bpb' in eval_modes:
        print0("\n" + "="*80)
        print0("BPB Evaluation")
        print0("="*80)
        tokens_per_step = args.device_batch_size * sequence_len * ddp_world_size
        if args.split_tokens % tokens_per_step != 0:
            # Adjust to nearest multiple
            args.split_tokens = (args.split_tokens // tokens_per_step) * tokens_per_step
            print0(f"Adjusted split_tokens to {args.split_tokens} (must be divisible by {tokens_per_step})")
        steps = args.split_tokens // tokens_per_step
        base_equiv_map = None
        if args.bpb_accuracy:
            base_equiv_map = build_case_space_base_equiv_map(tokenizer, device=device, token_bytes=token_bytes)

        def _as_stats(maybe_stats):
            if isinstance(maybe_stats, dict):
                return maybe_stats
            return {"bpb": float(maybe_stats)}

        local_parquet_dir = args.local_parquet_dir.strip() or None
        for split_name in ["train", "val"]:
            loader = tokenizing_distributed_data_loader_bos_bestfit(
                tokenizer,
                args.device_batch_size,
                sequence_len,
                split_name,
                device=device,
                local_parquet_dir=local_parquet_dir,
                with_modifiers=bool(hasattr(tokenizer, "has_compositional_mode") and tokenizer.has_compositional_mode()),
            )
            bpb_stats = evaluate_bpb(
                model,
                loader,
                steps,
                token_bytes,
                tokenizer=tokenizer,
                return_accuracy=args.bpb_accuracy,
                base_equiv_map=base_equiv_map,
            )
            bpb_results[split_name] = _as_stats(bpb_stats)
            if args.bpb_accuracy:
                print0(
                    f"{split_name} bpb: {bpb_results[split_name]['bpb']:.6f} | "
                    f"surface_acc {bpb_results[split_name]['surface_acc']:.6f} | "
                    f"base_acc {bpb_results[split_name]['base_acc']:.6f}"
                )
            else:
                print0(f"{split_name} bpb: {bpb_results[split_name]['bpb']:.6f}")

    # --- CORE evaluation ---
    if 'core' in eval_modes:
        print0("\n" + "="*80)
        print0("CORE Evaluation")
        print0("="*80)
        core_results = evaluate_core(model, tokenizer, device, max_per_task=args.max_per_task)
        if ddp_rank == 0:
            print0(
                f"CORE metric: {core_results['core_metric']:.4f} | "
                f"normalized_CORE metric: {core_results['normalized_core_metric']:.4f}"
            )

    # --- Downstream evaluation ---
    downstream_modes = [m for m in ("downstream", "downstream_lm") if m in eval_modes]
    if downstream_modes:
        generate_fn, generate_batch_fn, downstream_timing_fn = _build_downstream_generate_fns(
            model,
            tokenizer,
            is_hf_model=bool(is_hf_model),
        )
        eval_bundle_dir = ensure_eval_bundle_dir()

        def _mc_score_fn(task_data, task_meta):
            return float(
                evaluate_task(
                    model,
                    tokenizer,
                    task_data,
                    device,
                    task_meta,
                    return_normalized=False,
                )
            )

        for suite_mode in downstream_modes:
            print0("\n" + "=" * 80)
            print0("Downstream LM-Only Evaluation" if suite_mode == "downstream_lm" else "Downstream Evaluation")
            print0("=" * 80)
            ds_cfg = DownstreamEvalConfig(
                max_per_task=int(args.downstream_max_per_task),
                sample_seed=int(args.downstream_sample_seed),
                verbose=bool(args.downstream_verbose),
                log_every=int(args.downstream_log_every),
                em_batch_size=int(args.downstream_em_batch_size),
                em_max_new_tokens=int(args.downstream_em_max_new_tokens),
                em_temperature=float(args.downstream_em_temperature),
                em_top_p=float(args.downstream_em_top_p),
                em_top_k=int(args.downstream_em_top_k),
                coding_batch_size=int(args.downstream_coding_batch_size),
                coding_num_samples=int(args.downstream_coding_num_samples),
                coding_pass_k=int(args.downstream_coding_pass_k),
                coding_temperature=float(args.downstream_coding_temperature),
                coding_top_p=float(args.downstream_coding_top_p),
                coding_top_k=int(args.downstream_coding_top_k),
                coding_max_new_tokens=int(args.downstream_coding_max_new_tokens),
                suite=str(suite_mode),
                task_filter=str(args.downstream_task_filter),
                task_exclude_filter=str(args.downstream_task_exclude_filter),
                preset=str(args.downstream_preset or 'default'),
            )
            suite_results = evaluate_downstream(
                device=device,
                eval_bundle_dir=eval_bundle_dir,
                cfg=ds_cfg,
                generate_fn=generate_fn,
                generate_batch_fn=generate_batch_fn,
                mc_score_fn=_mc_score_fn,
                generation_timing_fn=downstream_timing_fn,
                task_result_callback=task_result_writer if ddp_rank == 0 else None,
            )
            downstream_results_by_suite[suite_mode] = suite_results
            if ddp_rank == 0:
                task_scores = suite_results.get("results", {})
                specs_by_label = {s.label: s for s in get_downstream_specs(suite_mode)}
                for label, score in task_scores.items():
                    spec = specs_by_label.get(label)
                    metric = spec.metric if spec is not None else "?"
                    print0(f"{label:<24} [{metric}] {float(score):.4f}")
                metric_name = "downstream_lm metric" if suite_mode == "downstream_lm" else "downstream metric"
                print0(f"{metric_name}: {float(suite_results['downstream_metric']):.4f}")

    training_summary = None
    if not is_hf_model:
        training_summary = _infer_training_summary(loaded_meta, loaded_checkpoint_dir, resolved_step)
        if ddp_rank == 0:
            print0("\n" + "=" * 80)
            print0("Training Run Summary")
            print0("=" * 80)
            print0(f"Total training time: {training_summary['total_training_hours_str']}")
            print0(f"Training hardware: {training_summary['gpu_setting']}")

    if ddp_rank == 0 and (core_results or downstream_results_by_suite or bpb_results):
        base_dir = get_base_dir()
        output_csv_path = os.path.join(base_dir, "base_eval", f"{model_slug}.csv")
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
            run_model_tag = args.hf_path if is_hf_model else resolved_model_tag
            run_step = "" if is_hf_model else resolved_step
            f.write(f"{'Model Tag':<35}, {run_model_tag}\n")
            f.write(f"{'Step':<35}, {run_step}\n")
            f.write("\n")
            if core_results:
                f.write(
                    f"{'Task':<35}, {'Accuracy':<10}, {'Norm Acc':<10}, "
                    f"{'Centered':<10}, {'Norm Centered':<13}\n"
                )
                for label in core_results["results"]:
                    acc = core_results["results"][label]
                    nacc = core_results["normalized_results"][label]
                    centered = core_results["centered_results"][label]
                    ncentered = core_results["normalized_centered_results"][label]
                    f.write(f"{label:<35}, {acc:<10.6f}, {nacc:<10.6f}, {centered:<10.6f}, {ncentered:<13.6f}\n")
                f.write(
                    f"{'CORE':<35}, {'':<10}, {'':<10}, {core_results['core_metric']:<10.6f}, "
                    f"{core_results['normalized_core_metric']:<13.6f}\n"
                )
            if downstream_results_by_suite:
                if core_results:
                    f.write("\n")
                for suite_mode in ("downstream", "downstream_lm"):
                    suite_results = downstream_results_by_suite.get(suite_mode)
                    if not suite_results:
                        continue
                    section_name = "Downstream LM Task" if suite_mode == "downstream_lm" else "Downstream Task"
                    metric_row_name = "downstream_lm" if suite_mode == "downstream_lm" else "downstream"
                    f.write(f"{section_name}, Category, Metric, Score\n")
                    task_scores = suite_results.get("results", {})
                    specs_by_label = {s.label: s for s in get_downstream_specs(suite_mode)}
                    for label, score in task_scores.items():
                        spec = specs_by_label.get(label)
                        category = spec.category if spec is not None else ""
                        metric = spec.metric if spec is not None else ""
                        f.write(f"{label}, {category}, {metric}, {float(score):.6f}\n")
                    if suite_results.get("arithmetic_splits"):
                        splits = suite_results["arithmetic_splits"]
                        f.write(
                            "Arithmetic Splits, , , "
                            f"2da={float(splits.get('2da', float('nan'))):.6f}; "
                            f"2dm={float(splits.get('2dm', float('nan'))):.6f}; "
                            f"2ds={float(splits.get('2ds', float('nan'))):.6f}\n"
                        )
                    f.write(f"{metric_row_name}, , raw_mean, {float(suite_results['downstream_metric']):.6f}\n")
                    f.write("\n")
            if bpb_results:
                if core_results or downstream_results_by_suite:
                    f.write("\n")
                if args.bpb_accuracy:
                    f.write("Split, BPB, Surface Acc, Base Acc\n")
                else:
                    f.write("Split, BPB\n")
                for split_name in ("train", "val"):
                    split_stats = bpb_results.get(split_name)
                    if not split_stats:
                        continue
                    if args.bpb_accuracy:
                        f.write(
                            f"{split_name}, "
                            f"{split_stats.get('bpb', float('nan')):.6f}, "
                            f"{split_stats.get('surface_acc', float('nan')):.6f}, "
                            f"{split_stats.get('base_acc', float('nan')):.6f}\n"
                        )
                    else:
                        f.write(f"{split_name}, {split_stats.get('bpb', float('nan')):.6f}\n")
            if training_summary is not None:
                f.write("\n")
                f.write(f"{'Total Training Time (hours)':<35}, {training_summary['total_training_hours_str']}\n")
                f.write(f"{'Training Hardware':<35}, {training_summary['gpu_setting']}\n")
        print0(f"\nResults written to: {output_csv_path}")

    # --- Log to report ---
    from nanochat.report import get_report
    report_data = [{"model": model_name}]

    if core_results:
        report_data[0]["CORE metric"] = core_results["core_metric"]
        report_data[0]["normalized CORE metric"] = core_results["normalized_core_metric"]
        report_data.append(core_results["centered_results"])
        report_data.append({f"norm::{k}": v for k, v in core_results["normalized_centered_results"].items()})
    if downstream_results_by_suite:
        if "downstream" in downstream_results_by_suite:
            report_data[0]["downstream metric"] = downstream_results_by_suite["downstream"]["downstream_metric"]
            report_data.append(
                {f"downstream::{k}": v for k, v in downstream_results_by_suite["downstream"].get("results", {}).items()}
            )
        if "downstream_lm" in downstream_results_by_suite:
            report_data[0]["downstream_lm metric"] = downstream_results_by_suite["downstream_lm"]["downstream_metric"]
            report_data.append(
                {f"downstream_lm::{k}": v for k, v in downstream_results_by_suite["downstream_lm"].get("results", {}).items()}
            )

    if bpb_results:
        train_stats = bpb_results.get("train", {})
        val_stats = bpb_results.get("val", {})
        report_data[0]["train bpb"] = train_stats.get("bpb")
        report_data[0]["val bpb"] = val_stats.get("bpb")
        if args.bpb_accuracy:
            report_data[0]["train surface_acc"] = train_stats.get("surface_acc")
            report_data[0]["train base_acc"] = train_stats.get("base_acc")
            report_data[0]["val surface_acc"] = val_stats.get("surface_acc")
            report_data[0]["val base_acc"] = val_stats.get("base_acc")
    if training_summary is not None:
        report_data[0]["total training time (hours)"] = training_summary["total_training_hours_str"]
        report_data[0]["training hardware"] = training_summary["gpu_setting"]
        report_data[0]["training num gpus"] = training_summary["num_gpus"]

    if samples:
        report_data.append({f"sample {i}": s for i, s in enumerate(samples)})
    if unconditioned_samples:
        report_data.append({f"unconditioned {i}": s for i, s in enumerate(unconditioned_samples)})

    get_report().log(section="Base model evaluation", data=report_data)

    compute_cleanup()


if __name__ == "__main__":
    main()
