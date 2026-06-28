"""
Compatibility shim for functions removed from transformers >= 5.

``find_pruneable_heads_and_indices`` was removed from
``transformers.pytorch_utils`` in transformers 5.0. This module vendors the
v4 implementation verbatim so the rest of the package works with both v4 and
v5.

Source: https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/pytorch_utils.py
"""

import torch


def get_head_mask(
    head_mask: torch.Tensor | None,
    num_hidden_layers: int,
) -> list[torch.Tensor | None]:
    """
    Expand head_mask to [num_hidden_layers] if it is None or a single tensor.

    Vendored from transformers v4 (removed in v5).
    """
    if head_mask is None:
        return [None] * num_hidden_layers
    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    return list(head_mask)


def find_pruneable_heads_and_indices(
    heads: list[int], n_heads: int, head_size: int, already_pruned_heads: set[int]
) -> tuple[set[int], torch.LongTensor]:
    """
    Find the heads and their indices taking ``already_pruned_heads`` into account.

    Args:
    ----
        heads: List of the indices of heads to prune.
        n_heads: The number of heads in the model.
        head_size: The size of each head.
        already_pruned_heads: A set of already pruned heads.

    Returns:
    -------
        A tuple with the indices of heads to prune taking ``already_pruned_heads``
        into account and the indices of rows/columns to keep in the layer weight.
    """
    mask = torch.ones(n_heads, head_size)
    heads = (
        set(heads) - already_pruned_heads
    )  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index
