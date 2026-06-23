#!/usr/bin/env python3
"""Unit tests for BiomedRNA model."""


import torch

# IMPORTANT: Import model class to trigger registration
from biomed_rna_plugin.biomed_rna import (
    BiomedRnaForSequenceEmbedding,  # noqa: F401
)

# Fixtures are in conftest.py and automatically available


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


def test_compare_sequential_vs_batched(vllm_model):
    """
    Test variable-length batching with fake data.

    Uses shared vllm_model fixture (max_num_seqs=8) and compares batched
    vs sequential processing by controlling batch size in embed() call.
    """
    cells = [
        create_fake_cell(num_genes=20, cell_id=1),
        create_fake_cell(num_genes=10, cell_id=2),
        create_fake_cell(num_genes=30, cell_id=3),
        create_fake_cell(num_genes=50, cell_id=4),
        create_fake_cell(num_genes=100, cell_id=5),
    ]

    # Process sequentially (one at a time)
    seq_outputs = [vllm_model.embed([cell])[0] for cell in cells]

    # Process as batch
    batch_outputs = vllm_model.embed(cells)

    # Compare
    for i, (seq_out, batch_out) in enumerate(zip(seq_outputs, batch_outputs)):
        seq_emb = torch.tensor(seq_out.outputs.embedding)
        batch_emb = torch.tensor(batch_out.outputs.embedding)

        assert torch.allclose(
            seq_emb, batch_emb, rtol=1e-5, atol=1e-3
        ), f"Cell {i+1}: embeddings not close (allclose with rtol=1e-5, atol=1e-3)"


if __name__ == "__main__":
    # For standalone execution, create model inline
    import gc

    from biomed_rna_plugin import get_vllm_biomed_rna_model

    llm = get_vllm_biomed_rna_model(
        disable_log_stats=True,
        max_num_seqs=8,
    )
    test_compare_sequential_vs_batched(llm)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
