"""
Engine for efficient inference of our models.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes:
- The engine knows nothing about tokenization, it's purely token id sequences.

The whole thing is made as efficient as possible.
"""

import torch
import torch.nn.functional as F
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from cobpe.tokenization.encoding import TokenCodec

class KVCache:
    """
    KV Cache designed for Flash Attention 3's flash_attn_with_kvcache API.

    Key differences from FA2-style cache:
    - Tensors are (B, T, H, D) not (B, H, T, D)
    - FA3 updates the cache in-place during flash_attn_with_kvcache
    - Position tracked per batch element via cache_seqlens tensor
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        # Pre-allocate cache tensors: (n_layers, B, T, H, D)
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        # Current sequence length per batch element (FA3 needs int32)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        # Previous token's normalized embedding for smear (set by model forward pass)
        self.prev_embedding = None

    def reset(self):
        """Reset cache to empty state."""
        self.cache_seqlens.zero_()
        self.prev_embedding = None

    def get_pos(self):
        """Get current position (assumes all batch elements at same position)."""
        return self.cache_seqlens[0].item()

    def get_layer_cache(self, layer_idx):
        """Return (k_cache, v_cache) views for a specific layer."""
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        """Advance the cache position by num_tokens."""
        self.cache_seqlens += num_tokens

    def prefill(self, other):
        """
        Copy cached KV from another cache into this one.
        Used when we do batch=1 prefill and then want to generate multiple samples in parallel.
        """
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        assert self.n_layers == other.n_layers and self.n_heads == other.n_heads and self.head_dim == other.head_dim
        assert self.max_seq_len >= other.max_seq_len
        other_pos = other.get_pos()
        self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
        self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
        self.cache_seqlens.fill_(other_pos)
        # Copy smear state: expand batch=1 prev_embedding to num_samples
        if other.prev_embedding is not None:
            self.prev_embedding = other.prev_embedding.expand(self.batch_size, -1, -1).clone()

# -----------------------------------------------------------------------------
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None, top_p=1.0):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    assert temperature >= 0.0, "temperature must be non-negative"
    if top_p is None:
        top_p = 1.0
    top_p = float(top_p)
    assert top_p > 0.0, "top_p must be positive"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_p >= 1.0:
        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            vals, idx = torch.topk(logits, k, dim=-1)
            vals = vals / temperature
            probs = F.softmax(vals, dim=-1)
            choice = torch.multinomial(probs, num_samples=1, generator=rng)
            return idx.gather(1, choice)
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)

    out = []
    vocab = logits.size(-1)
    for row in logits:
        row_logits = row.float()
        row_idx = None
        if top_k is not None and top_k > 0:
            k = min(int(top_k), int(vocab))
            row_logits, row_idx = torch.topk(row_logits, k, dim=-1)
        row_logits = row_logits / float(temperature)
        row_probs = F.softmax(row_logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(row_probs, descending=True)
        csum = torch.cumsum(sorted_probs, dim=-1)
        keep = csum <= top_p
        if keep.numel() > 0:
            keep[0] = True
        filtered_idx = sorted_idx[keep]
        filtered_probs = sorted_probs[keep]
        filtered_probs = filtered_probs / filtered_probs.sum()
        sampled_pos = torch.multinomial(filtered_probs, num_samples=1, generator=rng)
        sampled = int(filtered_idx[sampled_pos].item())
        if row_idx is not None:
            sampled = int(row_idx[sampled].item())
        out.append(sampled)
    return torch.tensor(out, dtype=torch.long, device=logits.device).unsqueeze(-1)

# -----------------------------------------------------------------------------

class RowState:
    # Per-row state tracking during generation
    def __init__(self, current_tokens):
        self.current_tokens = current_tokens
        self.completed = False # Whether this row has completed generation

class Engine:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer # needed for tool use
        self.token_codec = TokenCodec(tokenizer)

    def _sample_modifier_rows(self, hidden, token_ids, rng, temperature, top_k):
        if not hasattr(self.model, "get_modifier_logits"):
            raise ValueError("Compositional generation requires a model with get_modifier_logits().")
        modifier_logits = self.model.get_modifier_logits(hidden, token_ids)
        sampled_groups = []
        for group_logits in modifier_logits:
            sampled = sample_next_token(group_logits[:, -1, :], rng, temperature, top_k=None)
            sampled_groups.append(sampled[:, 0])
        if not sampled_groups:
            return None
        rows = torch.stack(sampled_groups, dim=-1)
        return rows.tolist()

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, top_p=1.0, seed=42):
        """Same as generate, but does single prefill and then clones the KV cache."""
        prompt = self.token_codec.normalize(tokens)
        if not self.token_codec.has_modifiers:
            assert prompt.modifiers is None and isinstance(prompt.ids, list) and all(isinstance(t, int) for t in prompt.ids), "expecting list of ints"
        compositional_mode = prompt.modifiers is not None
        device = self.model.get_device()
        # NOTE: setting the dtype here and in this way is an ugly hack.
        # Currently the repo assumes that cuda -> bfloat16 and everything else -> float32.
        # We need to know the dtype here to call __init__ on KVCache and pre-allocate its tensors.
        # As a quick hack, we're making generate() function inherit and know about this repo-wise assumption.
        # I think there has to be a bigger refactor to deal with device/dtype tracking across the codebase.
        # In particular, the KVCache should allocate its tensors lazily
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        assistant_end = self.tokenizer.encode_special("<|assistant_end|>") # if sampled, ends row
        bos = self.tokenizer.get_bos_token_id() # if sampled, ends row

        # 1) Run a batch 1 prefill of the prompt tokens
        m = self.model.config
        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(prompt),
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        ids, modifier_ids = self.token_codec.sequence_tensor(prompt, device)
        if compositional_mode:
            logits, hidden = self.model.forward(
                ids,
                kv_cache=kv_cache_prefill,
                modifier_ids=modifier_ids,
                return_hidden=True,
            )
            hidden = hidden[:, -1:, :].expand(num_samples, -1, -1).clone()
        else:
            logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :].expand(num_samples, -1)  # (num_samples, vocab_size)

        # 2) Replicate the KV cache for each sample/row
        kv_length_hint = (len(prompt) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill # no need to keep this memory around

        # 3) Initialize states for each sample
        row_states = [RowState(prompt.copy()) for _ in range(num_samples)]

        # 4) Main generation loop
        num_generated = 0
        while True:
            # Stop condition: we've reached max tokens
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # Stop condition: all rows are completed
            if all(state.completed for state in row_states):
                break

            # Sample the next token for each row
            next_ids = sample_next_token(logits, rng, temperature, top_k, top_p)  # (B, 1)
            sampled_tokens = next_ids[:, 0].tolist()
            sampled_modifier_rows = (
                self._sample_modifier_rows(hidden, next_ids, rng, temperature, top_k)
                if compositional_mode
                else None
            )

            # Append the sampled base token and its modifier tuple to each row.
            step = self.token_codec.empty_step()
            token_masks = [1] * num_samples
            for i, state in enumerate(row_states):
                next_item = self.token_codec.item(
                    sampled_tokens[i],
                    None if sampled_modifier_rows is None else sampled_modifier_rows[i],
                )
                step.append(next_item.id, next_item.modifier)
                state.current_tokens.append_item(next_item)
                if next_item.id == assistant_end or next_item.id == bos:
                    state.completed = True

            # Yield the token column
            if compositional_mode:
                yield (step.ids, step.modifiers), token_masks
            else:
                yield step.ids, token_masks
            num_generated += 1

            # Prepare logits for next iteration
            ids, modifier_ids = self.token_codec.step_tensor(step, device)
            if compositional_mode:
                logits, hidden = self.model.forward(
                    ids,
                    kv_cache=kv_cache_decode,
                    modifier_ids=modifier_ids,
                    return_hidden=True,
                )
                hidden = hidden[:, -1:, :]
                logits = logits[:, -1, :]
            else:
                logits = self.model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]  # (B, vocab_size)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that just returns the final token sequences.
        Returns a list of token sequences (list of lists of ints).
        Terminal tokens (assistant_end, bos) are not included in the results.
        """
        prompt = self.token_codec.normalize(tokens)
        compositional_mode = prompt.modifiers is not None
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [prompt.copy() for _ in range(num_samples)]
        masks = [[0] * len(prompt) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            if compositional_mode:
                token_ids, modifier_rows = token_column
            else:
                token_ids = token_column
                modifier_rows = None
            for i, (token, mask) in enumerate(zip(token_ids, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token, None if modifier_rows is None else modifier_rows[i])
                        masks[i].append(mask)
            # Stop if all rows are completed
            if all(completed):
                break
        if compositional_mode:
            return ([result.ids for result in results], [result.modifiers for result in results]), masks
        return [result.ids for result in results], masks
