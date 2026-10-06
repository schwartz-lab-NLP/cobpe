"""Small helpers for generation scripts that support plain BPE and CoBPE."""

from __future__ import annotations

import torch

from cobpe.tokenization.encoding import EncodedBatch, EncodedSequence, TokenCodec, stack_sequences


def encode_prompt(tokenizer, text: str, prepend="<|bos|>") -> EncodedSequence:
    """Encode a prompt as a structured sequence for Engine.generate_batch."""
    return TokenCodec(tokenizer).encode_text(text, prepend=prepend)


def generated_batch_to_sequences(tokenizer, batch) -> list[EncodedSequence]:
    """
    Normalize Engine.generate_batch output to EncodedSequence objects.

    Engine still returns legacy public shapes:
    - plain BPE: list[list[int]]
    - CoBPE: (list[list[int]], list[list[list[int]]])
    This adapter gives scripts one shape without changing Engine callers all at
    once.
    """
    if isinstance(batch, tuple):
        token_rows, modifier_rows = batch
        return [
            EncodedSequence(list(token_ids), [list(row) for row in modifiers])
            for token_ids, modifiers in zip(token_rows, modifier_rows)
        ]
    return [EncodedSequence(list(token_ids)) for token_ids in batch]


def decode_sequence(tokenizer, sequence: EncodedSequence) -> str:
    """Decode a structured generated sequence with the tokenizer's codec."""
    return TokenCodec(tokenizer).decode(sequence)


def decode_generated_batch(tokenizer, batch) -> list[str]:
    """Decode every sequence returned by Engine.generate_batch."""
    return [decode_sequence(tokenizer, seq) for seq in generated_batch_to_sequences(tokenizer, batch)]


def build_autoregressive_batch(
    tokenizer,
    sequences: list[EncodedSequence],
    masks: list[list[int]],
    *,
    pad_token_id: int | None = None,
    device=None,
    pin_memory: bool = False,
):
    """
    Pad structured sequences and build next-token inputs/targets.

    `masks` is aligned to the unshifted token sequence. After shifting, positions
    whose target mask is 0 are assigned ignore_index=-1. The return value keeps
    the historical plain-BPE shape `(input_ids, target_ids)`, and returns
    `EncodedBatch` pairs only when modifier rows are present.
    """
    if len(sequences) != len(masks):
        raise ValueError(f"sequences/masks length mismatch: {len(sequences)} != {len(masks)}")
    if not sequences:
        raise ValueError("cannot build an autoregressive batch from no sequences")
    for idx, (sequence, mask) in enumerate(zip(sequences, masks)):
        if len(sequence) != len(mask):
            raise ValueError(
                f"sequence/mask length mismatch at row {idx}: {len(sequence)} != {len(mask)}"
            )

    codec = TokenCodec(tokenizer)
    if pad_token_id is None:
        pad_token_id = tokenizer.get_bos_token_id()
    default_modifier = codec.default_modifier() if codec.has_modifiers else None
    stacked = stack_sequences(sequences, int(pad_token_id), default_modifier)
    max_len = stacked.ids.size(1)
    mask_rows = [
        [int(v) for v in mask] + [0] * (max_len - len(mask))
        for mask in masks
    ]
    mask_tensor = torch.tensor(mask_rows, dtype=torch.int8)

    if pin_memory:
        stacked_ids = stacked.ids.pin_memory()
        mask_tensor = mask_tensor.pin_memory()
        stacked_modifiers = None if stacked.modifiers is None else stacked.modifiers.pin_memory()
    else:
        stacked_ids = stacked.ids
        stacked_modifiers = stacked.modifiers

    move_kwargs = {"non_blocking": bool(pin_memory)}
    input_ids = stacked_ids[:, :-1].to(device=device, dtype=torch.int32, **move_kwargs).contiguous()
    target_ids = stacked_ids[:, 1:].to(device=device, dtype=torch.int64, **move_kwargs).contiguous()
    target_mask = mask_tensor[:, 1:].to(device=device, **move_kwargs)
    target_ids[target_mask == 0] = -1

    if stacked_modifiers is None:
        return input_ids, target_ids

    input_modifiers = stacked_modifiers[:, :-1].to(device=device, dtype=torch.long, **move_kwargs).contiguous()
    target_modifiers = stacked_modifiers[:, 1:].to(device=device, dtype=torch.long, **move_kwargs).contiguous()
    return EncodedBatch(input_ids, input_modifiers), EncodedBatch(target_ids, target_modifiers)
