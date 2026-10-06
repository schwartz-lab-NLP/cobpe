"""Input and output layers for CoBPE modifier groups."""

import torch
import torch.nn as nn
import torch.nn.functional as F


COBPE_CONDITIONING_MODES = ("base_bias", "concat_gated", "concat_gated_refine")
COBPE_GATE_MODES = ("scalar", "per_group")


def _round_up(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


def _norm(x):
    return F.rms_norm(x, (x.size(-1),))


class CoBPEModule(nn.Module):
    """
    Add modifier embeddings to base-token inputs and predict modifier values.

    Modifier logits use a context-only linear head, optionally with a base-token
    bias or a gated base-token prediction. The refine variant adds a residual
    logit transform to the gated prediction.
    """

    def __init__(
        self,
        group_sizes,
        n_embd,
        padded_vocab_size,
        linear_cls,
        conditioning_mode="concat_gated_refine",
        gate_mode="per_group",
        gate_conditioning="hidden",
    ):
        super().__init__()
        self.group_sizes = tuple(int(size) for size in group_sizes)
        self.num_groups = len(self.group_sizes)
        self.total_size = sum(self.group_sizes)
        # DistMuonAdamW reduce-scatter shards large matrices along dim 0.
        # Keep all modifier tables and heads shard-friendly on common GPU counts.
        self.padded_total_size = _round_up(max(1, self.total_size), 64)
        self.conditioning_mode = str(conditioning_mode).lower()
        if self.conditioning_mode not in COBPE_CONDITIONING_MODES:
            raise ValueError(
                f"Unsupported modifier_conditioning_mode={conditioning_mode!r}. "
                f"Expected one of: {list(COBPE_CONDITIONING_MODES)}."
            )
        self.gate_mode = str(gate_mode).lower()
        if self.gate_mode not in COBPE_GATE_MODES:
            raise ValueError(
                f"Unsupported modifier_gate_mode={gate_mode!r}. "
                f"Expected one of: {list(COBPE_GATE_MODES)}."
            )
        if str(gate_conditioning).lower() != "hidden":
            raise ValueError("modifier_gate_conditioning only supports 'hidden'.")

        offsets = []
        offset = 0
        for group_size in self.group_sizes:
            offsets.append(offset)
            offset += group_size
        self.group_offsets = tuple(offsets)

        self.embed = nn.Embedding(self.padded_total_size, n_embd)
        self.head = linear_cls(n_embd, self.padded_total_size, bias=False)
        self.logit_refine = None
        self.base_bias = None
        self.base_proj = None
        self.gate = None
        if self.conditioning_mode == "base_bias":
            self.base_bias = nn.Embedding(padded_vocab_size, self.padded_total_size)
        elif self.conditioning_mode in {"concat_gated", "concat_gated_refine"}:
            self.base_proj = linear_cls(n_embd, self.padded_total_size, bias=False)
            self.gate_size = 1 if self.gate_mode == "scalar" else self.num_groups
            self.gate = linear_cls(n_embd, self.gate_size, bias=False)
            if self.conditioning_mode == "concat_gated_refine":
                self.logit_refine = linear_cls(self.padded_total_size, self.padded_total_size, bias=False)

    @torch.no_grad()
    def init_weights(self, *, compute_dtype):
        torch.nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        if self.head is not None:
            torch.nn.init.normal_(self.head.weight, mean=0.0, std=0.001)
        if self.logit_refine is not None:
            # Preserve the concat-gated prediction at initialization and let the
            # refinement layer learn a residual correction.
            torch.nn.init.zeros_(self.logit_refine.weight)
        if self.base_bias is not None:
            torch.nn.init.zeros_(self.base_bias.weight)
        if self.base_proj is not None:
            torch.nn.init.normal_(self.base_proj.weight, mean=0.0, std=0.001)
        if self.gate is not None:
            torch.nn.init.zeros_(self.gate.weight)
        if self.padded_total_size > self.total_size:
            self.embed.weight[self.total_size:].zero_()
            for module in (self.head, self.base_proj, self.logit_refine):
                if module is not None:
                    module.weight[self.total_size:].zero_()
            if self.base_bias is not None:
                self.base_bias.weight[:, self.total_size:].zero_()
        if compute_dtype != torch.float16:
            self.embed.to(dtype=compute_dtype)

    def _offset_ids(self, modifier_ids):
        if modifier_ids.dim() != 3:
            raise ValueError(f"modifier_ids must have shape [B, T, G], got {tuple(modifier_ids.shape)}")
        if modifier_ids.size(-1) != self.num_groups:
            raise ValueError(f"modifier_ids group mismatch: expected {self.num_groups}, got {modifier_ids.size(-1)}")
        modifier_ids = modifier_ids.long()
        group_sizes = torch.tensor(self.group_sizes, dtype=torch.long, device=modifier_ids.device).view(1, 1, -1)
        bad_mask = (modifier_ids < 0) | (modifier_ids >= group_sizes)
        if bad_mask.any():
            bad_indices = bad_mask.nonzero(as_tuple=False)[0].detach().cpu().tolist()
            bad_value = int(modifier_ids[tuple(bad_indices)].detach().cpu().item())
            group_idx = int(bad_indices[-1])
            raise ValueError(f"modifier_ids out of range for group {group_idx}: group_size={self.group_sizes[group_idx]}, sample_bad_id={bad_value}")
        offsets = torch.tensor(self.group_offsets, dtype=torch.long, device=modifier_ids.device).view(1, 1, -1)
        return modifier_ids + offsets

    def embed_sum(self, modifier_ids):
        return self.embed(self._offset_ids(modifier_ids)).sum(dim=-2)

    def logits(self, hidden, token_ids, *, token_embedding_weight, base_unembedding):
        if hidden.dim() != 3:
            raise ValueError(f"hidden must have shape [B, T, C], got {tuple(hidden.shape)}")
        if token_ids.shape != hidden.shape[:2]:
            raise ValueError(
                "token_ids must match hidden shape: "
                f"{tuple(token_ids.shape)} != {tuple(hidden.shape[:2])}"
            )
        hidden_flat = hidden.view(-1, hidden.size(-1))
        token_flat = token_ids.view(-1).long()
        if self.conditioning_mode == "base_bias":
            all_logits = self.head(_norm(hidden_flat))
            all_logits = all_logits + self.base_bias(token_flat).to(all_logits.dtype)
        elif self.conditioning_mode in {"concat_gated", "concat_gated_refine"}:
            base_rep = F.embedding(token_flat, base_unembedding).to(dtype=hidden_flat.dtype)
            hidden_logits = self.head(_norm(hidden_flat))
            base_logits = self.base_proj(_norm(base_rep))
            gate = torch.sigmoid(self.gate(hidden_flat)).to(dtype=base_logits.dtype)
            if self.gate_mode == "scalar":
                all_logits = hidden_logits + gate * base_logits
            else:
                gated_base_groups = []
                for group_idx, group_size in enumerate(self.group_sizes):
                    start = self.group_offsets[group_idx]
                    end = start + group_size
                    gated_base_groups.append(
                        base_logits[:, start:end] * gate[:, group_idx:group_idx + 1]
                    )
                if self.total_size < self.padded_total_size:
                    # Padding has no corresponding modifier group. Reuse the
                    # final group's gate so it does not become an ungated path.
                    gated_base_groups.append(
                        base_logits[:, self.total_size:]
                        * gate[:, -1:]
                    )
                gated_base_logits = torch.cat(gated_base_groups, dim=-1)
                all_logits = hidden_logits + gated_base_logits
            if self.logit_refine is not None:
                all_logits = all_logits + self.logit_refine(F.silu(all_logits))
        else:
            raise ValueError(f"Unknown modifier_conditioning_mode: {self.conditioning_mode}")

        outputs = []
        for group_idx, group_size in enumerate(self.group_sizes):
            start = self.group_offsets[group_idx]
            outputs.append(all_logits[:, start:start + group_size].view(*token_ids.shape, group_size))
        return outputs

    def loss(self, hidden, targets, target_modifier_ids, *, token_embedding_weight, base_unembedding, loss_reduction):
        if target_modifier_ids is None:
            return None
        if target_modifier_ids.dim() != 3:
            raise ValueError(f"target_modifier_ids must have shape [B, T, G], got {tuple(target_modifier_ids.shape)}")
        if target_modifier_ids.shape[:2] != targets.shape:
            raise ValueError(
                "target_modifier_ids must match targets shape in the first two dimensions: "
                f"{tuple(target_modifier_ids.shape[:2])} != {tuple(targets.shape)}"
            )
        if target_modifier_ids.size(-1) != self.num_groups:
            raise ValueError(f"target_modifier_ids group mismatch: expected {self.num_groups}, got {target_modifier_ids.size(-1)}")

        valid_targets = targets >= 0
        safe_targets = torch.where(valid_targets, targets, torch.zeros_like(targets))
        modifier_loss = None
        for group_idx, group_logits in enumerate(
            self.logits(
                hidden,
                safe_targets,
                token_embedding_weight=token_embedding_weight,
                base_unembedding=base_unembedding,
            )
        ):
            group_targets = target_modifier_ids[..., group_idx].long()
            group_targets = torch.where(valid_targets, group_targets, torch.full_like(group_targets, -1))
            group_loss = F.cross_entropy(
                group_logits.float().view(-1, group_logits.size(-1)),
                group_targets.reshape(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            modifier_loss = group_loss if modifier_loss is None else modifier_loss + group_loss
        return modifier_loss

    def parameter_counts(self):
        modifier_conditioners = 0
        for module in (
            self.base_proj,
            self.gate,
            self.logit_refine,
        ):
            if module is not None:
                modifier_conditioners += sum(p.numel() for p in module.parameters())
        return {
            "modifier_embeds": sum(p.numel() for p in self.embed.parameters()),
            "modifier_heads": sum(
                p.numel()
                for module in (self.head,)
                if module is not None
                for p in module.parameters()
            ),
            "modifier_base_biases": 0 if self.base_bias is None else sum(p.numel() for p in self.base_bias.parameters()),
            "modifier_conditioners": modifier_conditioners,
        }

    def flop_excluded_numel(self):
        numel = self.embed.weight.numel()
        if self.base_bias is not None:
            numel += self.base_bias.weight.numel()
        return numel

    def optimizer_params(self):
        return list(self.parameters())
