"""Compatibility shim for functions removed from transformers >= 5."""

from __future__ import annotations

import torch


def get_head_mask(
    head_mask: torch.Tensor | None,
    num_hidden_layers: int,
    is_attention_chunked: bool = False,
) -> torch.Tensor | None:
    """Replicate PreTrainedModel.get_head_mask removed in transformers v5."""
    if head_mask is not None:
        head_mask = _convert_head_mask_to_5d(head_mask, num_hidden_layers)
        if is_attention_chunked is True:
            head_mask = head_mask.unsqueeze(-1)
    else:
        head_mask = [None] * num_hidden_layers
    return head_mask


def _convert_head_mask_to_5d(
    head_mask: torch.Tensor, num_hidden_layers: int
) -> torch.Tensor:
    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    return head_mask


def find_pruneable_heads_and_indices(
    heads: set[int],
    n_heads: int,
    head_size: int,
    already_pruned_heads: set[int],
) -> tuple[set[int], torch.Tensor]:
    """
    Replicate transformers <= 4.x find_pruneable_heads_and_indices.

    Removed from transformers.pytorch_utils in v5.
    """
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()
    return heads, index
