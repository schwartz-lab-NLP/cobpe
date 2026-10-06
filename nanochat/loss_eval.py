"""
A number of functions that help with evaluating a base model.
"""
import math
import torch
import torch.distributed as dist

def _canonical_surface_for_base_acc(token_surface: str):
    """Canonicalize a token surface by removing space-prefix and normalizing capitalization."""
    if token_surface is None:
        return None
    stripped = token_surface.lstrip(" ")
    if not stripped:
        return None
    if any(ch.isspace() for ch in stripped):
        return None
    alpha_chars = [ch for ch in stripped if ch.isalpha()]
    if not alpha_chars or len(alpha_chars) != len(stripped):
        return None
    if all(ch.isupper() for ch in alpha_chars):
        return stripped
    return stripped.lower()


@torch.no_grad()
def build_case_space_base_equiv_map(tokenizer, device="cpu", token_bytes=None):
    """
    Build token-id mapping for normalized base accuracy in base runs.
    Tokens differing only by leading-space and capitalization map to one canonical id.
    """
    vocab_size = tokenizer.get_vocab_size()
    base_equiv_map = torch.arange(vocab_size, dtype=torch.long, device=device)
    groups = {}

    for token_id in range(vocab_size):
        if token_bytes is not None and int(token_bytes[token_id].item()) == 0:
            continue
        token_surface = tokenizer.decode([token_id])
        key = _canonical_surface_for_base_acc(token_surface)
        if key is None:
            continue
        groups.setdefault(key, []).append((token_id, token_surface))

    for key, entries in groups.items():
        def _rank(entry):
            token_id, token_surface = entry
            if token_surface == key:
                return (0, token_id)
            if token_surface == f" {key}":
                return (1, token_id)
            if token_surface.lstrip(" ") == key:
                return (2, token_id)
            return (3, token_id)
        canonical_id = min(entries, key=_rank)[0]
        for token_id, _ in entries:
            base_equiv_map[token_id] = canonical_id

    return base_equiv_map


@torch.no_grad()
def _compositional_target_bytes(y, y_mods, tokenizer, device):
    if tokenizer is None or not hasattr(tokenizer, "utf8_len_with_modifiers_batch"):
        raise ValueError(
            "Compositional BPB evaluation requires tokenizer.utf8_len_with_modifiers_batch()."
        )
    y_flat = y.reshape(-1)
    mods_flat = y_mods.reshape(-1, y_mods.size(-1))
    valid = y_flat >= 0
    out = torch.zeros_like(y_flat, dtype=torch.int64, device=device)
    idxs = valid.nonzero(as_tuple=False).view(-1)
    if idxs.numel() == 0:
        return out
    token_ids = [int(v) for v in y_flat[idxs].detach().cpu().tolist()]
    modifier_rows = [[int(x) for x in row] for row in mods_flat[idxs].detach().cpu().tolist()]
    lengths = tokenizer.utf8_len_with_modifiers_batch(token_ids, modifier_rows)
    out[idxs] = torch.tensor(lengths, dtype=torch.int64, device=device)
    return out


@torch.no_grad()
def evaluate_bpb(
    model,
    batches,
    steps,
    token_bytes,
    tokenizer=None,
    loss_impl: str = "ce",
    return_accuracy: bool = False,
    base_equiv_map: torch.Tensor = None,
):
    """
    Instead of the naive 'mean loss', this function returns the bits per byte (bpb),
    which is a tokenization vocab size-independent metric, meaning you are still comparing
    apples:apples if you change the vocab size. The way this works is that instead of just
    calculating the average loss as usual, you calculate the sum loss, and independently
    also the sum bytes (of all the target tokens), and divide. This normalizes the loss by
    the number of bytes that the target tokens represent.

    The added complexity is so that:
    1) All "normal" tokens are normalized by the length of the token in bytes
    2) No special tokens (e.g. <|bos|>) are included in the metric - they are masked out.
    3) No actively masked tokens (using ignore_index of e.g. -1) are included in the metric.

    In addition to evaluate_loss, we need the token_bytes tensor:
    It is a 1D tensor of shape (vocab_size,), indicating the number of bytes for
    each token id, or 0 if the token is to not be counted (e.g. special tokens).
    """
    # record the losses
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    total_surface_correct = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    total_base_correct = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    total_valid_tokens = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    total_surface_correct_gt_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    total_target_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    accuracy_supported = (loss_impl == "ce")
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        compositional_mode = isinstance(x, tuple) and isinstance(y, tuple)
        if compositional_mode:
            x_ids, x_mods = x
            y_ids, y_mods = y
            if return_accuracy:
                accuracy_supported = False
            loss2d = model(
                x_ids,
                y_ids,
                modifier_ids=x_mods,
                target_modifier_ids=y_mods,
                loss_reduction="none",
            ).view(-1)
            y = y_ids.view(-1)
            num_bytes2d = _compositional_target_bytes(y_ids, y_mods, tokenizer, model.get_device())
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
            continue
        preds = None
        if return_accuracy and accuracy_supported:
            logits = model(x)  # (B, T, V)
            flat_logits = logits.view(-1, logits.size(-1))
            preds = flat_logits.argmax(dim=-1)
            loss2d = torch.nn.functional.cross_entropy(
                flat_logits,
                y.view(-1),
                ignore_index=-1,
                reduction="none",
            )
        else:
            if loss_impl != "ce":
                raise ValueError("evaluate_bpb currently supports loss_impl='ce' on the upstream-aligned GPT forward path.")
            loss2d = model(x, y, loss_reduction='none').view(-1) # (B, T) -> flatten
        y = y.view(-1) # flatten
        if (y.int() < 0).any(): # mps does not currently have kernel for < 0 for int64, only int32
            # slightly more complex code path if some target tokens are ignore_index (e.g. -1)
            # any target token < 0 is to be ignored: do NOT index token_bytes with negatives
            valid = y >= 0
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            # map valid targets to their byte length; ignored targets contribute 0 bytes
            num_bytes2d = torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y, dtype=token_bytes.dtype)
            )
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
            metric_mask = valid & (num_bytes2d > 0)
            y_metric = y_safe
        else:
            # fast path: no ignored targets, safe to index directly
            num_bytes2d = token_bytes[y]
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
            metric_mask = num_bytes2d > 0
            y_metric = y

        if return_accuracy and accuracy_supported:
            surface_correct = (preds == y_metric) & metric_mask
            total_surface_correct += surface_correct.sum()
            total_surface_correct_gt_bytes += (
                num_bytes2d * (surface_correct & metric_mask).to(num_bytes2d.dtype)
            ).sum()
            total_target_bytes += (num_bytes2d * metric_mask.to(num_bytes2d.dtype)).sum()

            if base_equiv_map is not None:
                pred_base = base_equiv_map[preds]
                target_base = base_equiv_map[y_metric]
            else:
                pred_base = preds
                target_base = y_metric
            base_correct = (pred_base == target_base) & metric_mask
            total_base_correct += base_correct.sum()
            total_valid_tokens += metric_mask.sum()

    # sum reduce across all ranks
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
        if return_accuracy:
            dist.all_reduce(total_surface_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_base_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_valid_tokens, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_surface_correct_gt_bytes, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_target_bytes, op=dist.ReduceOp.SUM)
    # move both to cpu, calculate bpb and return
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    if total_bytes == 0:
        if not return_accuracy:
            return float('inf')
        return {
            "bpb": float('inf'),
            "surface_acc": 0.0,
            "base_acc": 0.0,
            "surface_acc_bytes_gt": 0.0,
            "surface_correct": 0,
            "base_correct": 0,
            "valid_tokens": 0,
            "surface_correct_gt_bytes": 0,
            "target_bytes": 0,
        }
    bpb = total_nats / (math.log(2) * total_bytes)
    if not return_accuracy:
        return bpb

    valid_tokens = int(total_valid_tokens.item()) if accuracy_supported else 0
    surface_correct = int(total_surface_correct.item()) if accuracy_supported else 0
    base_correct = int(total_base_correct.item()) if accuracy_supported else 0
    surface_correct_gt_bytes = int(total_surface_correct_gt_bytes.item()) if accuracy_supported else 0
    target_bytes = int(total_target_bytes.item()) if accuracy_supported else 0
    denom = max(valid_tokens, 1)
    bytes_denom = max(target_bytes, 1)
    surface_acc = (surface_correct / denom) if accuracy_supported else float("nan")
    base_acc = (base_correct / denom) if accuracy_supported else float("nan")
    surface_acc_bytes_gt = (surface_correct_gt_bytes / bytes_denom) if accuracy_supported else float("nan")
    return {
        "bpb": bpb,
        "surface_acc": surface_acc,
        "base_acc": base_acc,
        "surface_acc_bytes_gt": surface_acc_bytes_gt,
        "surface_correct": surface_correct,
        "base_correct": base_correct,
        "valid_tokens": valid_tokens,
        "surface_correct_gt_bytes": surface_correct_gt_bytes,
        "target_bytes": target_bytes,
    }
