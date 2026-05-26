#!/usr/bin/env python3
"""Tests preprocessing of scRNA data."""

import torch

from vllm_biomed_rna_plugin.preprocess import create_rna_vllm_input


def test_create_rna_vllm_input():
    """Test create_rna_vllm_input with multi_modal_data format."""
    gene_ids = torch.Tensor([1000, 1001, 1002, 1003])
    expr_values = torch.Tensor([3.5, 7.2, 2.1, 5.8])
    attention = torch.Tensor([1, 1, 1, 1])

    result = create_rna_vllm_input(gene_ids, expr_values, attention)

    assert result["prompt_token_ids"] == [0] * 4
    assert "multi_modal_data" in result
    assert "rna" in result["multi_modal_data"]
    assert torch.equal(result["multi_modal_data"]["rna"]["gene_ids"], gene_ids)
    assert torch.equal(result["multi_modal_data"]["rna"]["expr_values"], expr_values)
