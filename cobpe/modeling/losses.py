"""Framework-neutral CoBPE objective helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CoBPELossConfig:
    """Weights and reductions for base-token plus modifier prediction."""

    base_loss_weight: float = 1.0
    modifier_loss_weight: float = 1.0
    ignore_index: int = -1


def validate_modifier_ids(modifier_ids: torch.Tensor, group_sizes: Sequence[int]) -> None:
    """Validate group-relative modifier ids shaped ``[batch, seq, groups]``."""

    if modifier_ids.ndim != 3:
        raise ValueError(f"modifier ids must have shape [batch, seq, groups], got {tuple(modifier_ids.shape)}")
    if modifier_ids.shape[-1] != len(group_sizes):
        raise ValueError(f"modifier group dimension {modifier_ids.shape[-1]} != expected {len(group_sizes)}")
    if torch.is_floating_point(modifier_ids):
        raise TypeError("modifier ids must be integer tensors")
    if not modifier_ids.numel():
        return
    modifier_ids = modifier_ids.long()
    for group_idx, group_size in enumerate(group_sizes):
        values = modifier_ids[..., group_idx]
        min_value = int(values.min().item())
        max_value = int(values.max().item())
        if min_value < 0 or max_value >= int(group_size):
            raise ValueError(
                f"modifier ids out of range for group {group_idx}: "
                f"min={min_value} max={max_value} valid=[0,{int(group_size)})"
            )


def modifier_cross_entropy(
    group_logits: Sequence[torch.Tensor],
    modifier_labels: torch.Tensor,
    *,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute modifier loss from per-group logits.

    ``group_logits[g]`` must have shape ``[batch, seq, group_size]`` and
    ``modifier_labels`` must contain group-relative ids ``[batch, seq, groups]``.
    """

    if not group_logits:
        raise ValueError("at least one modifier logit tensor is required")
    group_sizes = [int(logits.shape[-1]) for logits in group_logits]
    validate_modifier_ids(modifier_labels, group_sizes)
    if modifier_labels.shape[:2] != group_logits[0].shape[:2]:
        raise ValueError("modifier labels must match modifier logits batch/seq dimensions")
    if loss_mask is not None and loss_mask.shape != modifier_labels.shape[:2]:
        raise ValueError("loss_mask must match modifier labels batch/seq dimensions")

    losses = []
    for group_idx, logits in enumerate(group_logits):
        if logits.shape[:2] != modifier_labels.shape[:2]:
            raise ValueError(
                f"group {group_idx} logits batch/seq shape {tuple(logits.shape[:2])} "
                f"!= labels {tuple(modifier_labels.shape[:2])}"
            )
        group_targets = modifier_labels[..., group_idx].long()
        losses.append(
            F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                group_targets.reshape(-1),
                reduction="none",
            ).view_as(group_targets)
        )
    stacked = torch.stack(losses, dim=-1)
    # Modifier groups are independent factors of the reconstructed surface token,
    # so their negative log-likelihoods must be added, never averaged.
    per_token = stacked.sum(dim=-1)
    if loss_mask is None:
        return per_token.mean()
    mask = loss_mask.float()
    return (per_token * mask).sum() / mask.sum().clamp_min(1.0)


def compositional_lm_loss(
    *,
    base_logits: torch.Tensor,
    base_labels: torch.Tensor,
    group_logits: Sequence[torch.Tensor],
    modifier_labels: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    config: CoBPELossConfig = CoBPELossConfig(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute base-token plus CoBPE modifier loss.

    This helper is deliberately independent of nanochat or Megatron schedules.
    Framework integrations only need to provide logits, labels, and an optional
    loss mask in the standard shapes.
    """

    if base_logits.ndim != 3:
        raise ValueError(f"base_logits must have shape [batch, seq, vocab], got {tuple(base_logits.shape)}")
    if base_labels.shape != base_logits.shape[:2]:
        raise ValueError("base_labels must match base_logits batch/seq dimensions")
    if loss_mask is not None and loss_mask.shape != base_labels.shape:
        raise ValueError("loss_mask must match base_labels shape")
    flat_base = F.cross_entropy(
        base_logits.reshape(-1, base_logits.size(-1)).float(),
        base_labels.reshape(-1).long(),
        ignore_index=int(config.ignore_index),
        reduction="none",
    ).view_as(base_labels)
    if loss_mask is None:
        base_loss = flat_base.mean()
    else:
        mask = loss_mask.float()
        base_loss = (flat_base * mask).sum() / mask.sum().clamp_min(1.0)
    mod_loss = modifier_cross_entropy(
        group_logits,
        modifier_labels,
        loss_mask=loss_mask,
    )
    total = float(config.base_loss_weight) * base_loss + float(config.modifier_loss_weight) * mod_loss
    return total, {"base_loss": base_loss.detach(), "modifier_loss": mod_loss.detach()}
