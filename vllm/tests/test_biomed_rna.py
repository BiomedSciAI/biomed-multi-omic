#!/usr/bin/env python3
"""Unit tests for BiomedRNA model."""


import torch

# IMPORTANT: Import model class to trigger registration
from vllm_biomed_rna_plugin.biomed_rna import (
    BiomedRnaForSequenceEmbedding,  # noqa: F401
)

# Fixtures are in conftest.py and automatically available


def create_fake_cell(num_genes: int, cell_id: int, pooling_method: str | None = None):
    """Create a fake cell in vLLM input format with deterministic data."""
    torch.manual_seed(cell_id)
    gene_ids = torch.randint(0, 19321, (num_genes,)).long()
    expr_values = (torch.randn(num_genes) * 2 + 3).clamp(min=0.1).float()
    attention_mask = torch.ones(num_genes, dtype=torch.float32)
    rna: dict = {
        "gene_ids": gene_ids,
        "expr_values": expr_values,
        "attention_mask": attention_mask,
    }
    if pooling_method is not None:
        rna["pooling_method"] = pooling_method
    return {
        "prompt_token_ids": [1],
        "multi_modal_data": {"rna": rna},
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


def test_pooling_method_override(vllm_model):
    """
    Different pooling methods must produce different embeddings, and an explicit
    first_token override must match the model default.
    """
    cell_default = create_fake_cell(num_genes=50, cell_id=42)
    cell_first = create_fake_cell(
        num_genes=50, cell_id=42, pooling_method="first_token"
    )
    cell_mean = create_fake_cell(
        num_genes=50, cell_id=42, pooling_method="mean_pooling"
    )

    default_emb = torch.tensor(vllm_model.embed([cell_default])[0].outputs.embedding)
    first_emb = torch.tensor(vllm_model.embed([cell_first])[0].outputs.embedding)
    mean_emb = torch.tensor(vllm_model.embed([cell_mean])[0].outputs.embedding)

    assert torch.allclose(first_emb, default_emb, atol=1e-4), (
        "explicit first_token override should match the model default "
        f"(max diff = {(first_emb - default_emb).abs().max().item():.6f})"
    )
    assert not torch.allclose(
        mean_emb, first_emb, atol=1e-4
    ), "mean_pooling and first_token produced identical embeddings — override not working"


def test_per_request_pooling_in_batch(vllm_model):
    """
    A mixed batch (different pooling method per request) must match processing
    each request individually with its own pooling method.
    """
    c1_first = create_fake_cell(num_genes=50, cell_id=1, pooling_method="first_token")
    c2_mean = create_fake_cell(num_genes=50, cell_id=2, pooling_method="mean_pooling")

    e1_seq = torch.tensor(vllm_model.embed([c1_first])[0].outputs.embedding)
    e2_seq = torch.tensor(vllm_model.embed([c2_mean])[0].outputs.embedding)

    batch_out = vllm_model.embed([c1_first, c2_mean])
    e1_batch = torch.tensor(batch_out[0].outputs.embedding)
    e2_batch = torch.tensor(batch_out[1].outputs.embedding)

    assert torch.allclose(e1_seq, e1_batch, atol=1e-4), (
        f"Cell 1 (first_token): batch != sequential "
        f"(max diff = {(e1_seq - e1_batch).abs().max().item():.6f})"
    )
    assert torch.allclose(e2_seq, e2_batch, atol=1e-4), (
        f"Cell 2 (mean_pooling): batch != sequential "
        f"(max diff = {(e2_seq - e2_batch).abs().max().item():.6f})"
    )


if __name__ == "__main__":
    import gc

    from vllm_biomed_rna_plugin import get_vllm_biomed_rna_model

    llm = get_vllm_biomed_rna_model(disable_log_stats=True, max_num_seqs=8)
    test_compare_sequential_vs_batched(llm)
    test_pooling_method_override(llm)
    test_per_request_pooling_in_batch(llm)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
