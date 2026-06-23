#!/usr/bin/env python3
"""
Tests for BiomedRNA IO Processor.

Validates data conversion between HTTP JSON and vLLM internal format.
"""

import torch
from biomed_rna_plugin.io_processor import BiomedRnaIOProcessor, RnaPrompt


def test_parse_data_valid():
    """Test parsing valid RNA data."""
    data = {
        "gene_ids": [1, 2, 3],
        "expr_values": [1.0, 2.0, 3.0],
        "attention_mask": [1, 1, 1],
    }

    prompt = RnaPrompt(data)
    assert "gene_ids" in prompt
    assert "expr_values" in prompt
    assert "attention_mask" in prompt


def test_pre_process_structure(mock_vllm_config):
    """Test pre_process returns correct structure."""
    from unittest.mock import Mock

    processor = BiomedRnaIOProcessor(mock_vllm_config, Mock())

    prompt = RnaPrompt(
        {
            "gene_ids": [1, 2, 3],
            "expr_values": [1.0, 2.0, 3.0],
            "attention_mask": [1, 1, 1],
        }
    )

    result = processor.pre_process(prompt)

    # Verify structure
    assert "prompt_token_ids" in result
    assert "multi_modal_data" in result
    assert "rna" in result["multi_modal_data"]

    # Verify RNA data converted to tensors
    rna_data = result["multi_modal_data"]["rna"]
    assert isinstance(rna_data["gene_ids"], torch.Tensor)
    assert isinstance(rna_data["expr_values"], torch.Tensor)
    assert isinstance(rna_data["attention_mask"], torch.Tensor)


def test_merge_pooling_params(mock_vllm_config):
    """Test pooling params override to task='embed'."""
    from unittest.mock import Mock

    processor = BiomedRnaIOProcessor(mock_vllm_config, Mock())
    params = processor.merge_pooling_params()

    assert params.task == "embed"
