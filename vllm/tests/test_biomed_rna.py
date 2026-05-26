#!/usr/bin/env python3
"""Unit tests for BiomedRNA model."""

from pathlib import Path

import numpy as np
import torch

# IMPORTANT: Import model class to trigger registration
from vllm_biomed_rna_plugin.biomed_rna import (
    BiomedRnaForSequenceEmbedding,  # noqa: F401
)

# Fixtures (config, model) are now in conftest.py and automatically available


def test_forward(model, config):
    """Test production path: RNA data in kwargs → forward → encoder output."""
    # Create a fake cell with 10 genes
    cell = create_fake_cell(num_genes=10, cell_id=42)

    # Extract RNA data from the cell
    rna_data = cell["multi_modal_data"]["rna"]
    gene_ids = rna_data["gene_ids"]
    expr_values = rna_data["expr_values"]
    attention_mask = rna_data["attention_mask"]

    # Forward expects 2D tensors [batch, seq_len], so add batch dimension
    gene_ids_2d = gene_ids.unsqueeze(0)  # [1, seq_len]
    expr_values_2d = expr_values.unsqueeze(0)  # [1, seq_len]
    attention_mask_2d = attention_mask.unsqueeze(0)  # [1, seq_len]

    with torch.no_grad():
        output = model.forward(
            gene_ids=gene_ids_2d,
            expr_values=expr_values_2d,
            attention_mask=attention_mask_2d,
        )

    assert output.shape == (
        1,
        1,
        config.hidden_size,
    ), f"Expected (1, 1, {config.hidden_size}), got {output.shape}"

    assert torch.isfinite(output).all(), "Output contains NaN or Inf values"
    assert not torch.allclose(output, torch.zeros_like(output)), "Output is all zeros"


def create_fake_cell(num_genes: int, cell_id: int):
    """
    Create a fake cell with specified number of genes.

    Uses the same approach as test_biomed_rna.py:
    - gene_ids: torch.randint(0, 19321, (seq_len,)).long()
    - expr_values: (torch.randn(seq_len) * 2 + 3).clamp(min=0.1)

    Args:
    ----
        num_genes: Number of genes in this cell
        cell_id: Unique identifier for this cell

    Returns:
    -------
        dict: vLLM input format with gene_ids, expr_values, attention_mask
    """
    # Use valid gene IDs from vocabulary range (0-19320)
    # Same approach as test_biomed_rna.py
    torch.manual_seed(cell_id)
    gene_ids = torch.randint(0, 19321, (num_genes,)).long()

    # Create expression values (random but deterministic)
    expr_values = (torch.randn(num_genes) * 2 + 3).clamp(min=0.1).float()

    # Create attention mask (all ones for real tokens)
    attention_mask = torch.ones(num_genes, dtype=torch.float32)

    return {
        "prompt_token_ids": [0] * num_genes,  # Single fake token
        "multi_modal_data": {
            "rna": {
                "gene_ids": gene_ids,
                "expr_values": expr_values,
                "attention_mask": attention_mask,
            }
        },
    }


def test_compare_sequential_vs_batched():
    """
    Test variable-length batching with fake data.

    Vllm batches requests even when arriving from different users. Here we verify
    that batched requests are handled correctly, we compare no batched (max_num_seqs=1)
    and batched (max_num_seqs=8) processing.
    """
    import gc

    from vllm_biomed_rna_plugin import get_vllm_biomed_rna_model

    cells = [
        create_fake_cell(num_genes=20, cell_id=1),
        create_fake_cell(num_genes=10, cell_id=2),
        create_fake_cell(num_genes=30, cell_id=3),
        create_fake_cell(num_genes=50, cell_id=4),
        create_fake_cell(num_genes=100, cell_id=5),
    ]

    # Sequential
    llm = get_vllm_biomed_rna_model(
        disable_log_stats=True,
        max_num_seqs=1,
    )
    seq_outputs = llm.embed(cells)
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # Batched
    llm = get_vllm_biomed_rna_model(
        disable_log_stats=True,
        max_num_seqs=8,
    )
    batch_outputs = llm.embed(cells)
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # Compare
    for i, (seq_out, batch_out) in enumerate(zip(seq_outputs, batch_outputs)):
        seq_emb = torch.tensor(seq_out.outputs.embedding)
        batch_emb = torch.tensor(batch_out.outputs.embedding)

        assert torch.allclose(
            seq_emb, batch_emb, rtol=1e-5, atol=1e-3
        ), f"Cell {i+1}: embeddings not close (allclose with rtol=1e-5, atol=1e-3)"


if __name__ == "__main__":
    test_compare_sequential_vs_batched()
