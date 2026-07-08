#!/usr/bin/env python3
"""
Tests for BiomedRNA IO Processor.

Validates data conversion between HTTP JSON and vLLM internal format.
"""

import torch
from vllm_biomed_rna_plugin.io_processor import BiomedRnaIOProcessor, RnaPrompt


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


def test_pre_process_forwards_pooling_method(mock_vllm_config):
    """pooling_method in request body is forwarded into multi_modal_data["rna"]."""
    from unittest.mock import Mock

    processor = BiomedRnaIOProcessor(mock_vllm_config, Mock())
    data = {
        "gene_ids": [1, 2, 3],
        "expr_values": [1.0, 2.0, 3.0],
        "pooling_method": "mean_pooling",
    }
    prompt = processor.parse_data(data)
    result = processor.pre_process(prompt)

    assert result["multi_modal_data"]["rna"]["pooling_method"] == "mean_pooling"


def test_pre_process_no_pooling_method(mock_vllm_config):
    """When no pooling_method is present, rna dict does not contain the key."""
    from unittest.mock import Mock

    processor = BiomedRnaIOProcessor(mock_vllm_config, Mock())
    prompt = processor.parse_data(
        {"gene_ids": [1, 2, 3], "expr_values": [1.0, 2.0, 3.0]}
    )
    result = processor.pre_process(prompt)

    assert "pooling_method" not in result["multi_modal_data"]["rna"]


def test_merge_pooling_params_no_extra_kwargs(mock_vllm_config):
    """merge_pooling_params never sets extra_kwargs — pooling_method goes via mm_kwargs."""
    from unittest.mock import Mock

    processor = BiomedRnaIOProcessor(mock_vllm_config, Mock())
    processor.parse_data(
        {
            "gene_ids": [1, 2, 3],
            "expr_values": [1.0, 2.0, 3.0],
            "pooling_method": "mean_pooling",
        }
    )
    params = processor.merge_pooling_params()

    # extra_kwargs is not used in the new design
    assert params.extra_kwargs is None


# ---------------------------------------------------------------------------
# pooling_method_id encoding / decoding round-trip tests (no GPU required)
# ---------------------------------------------------------------------------

import pytest
from vllm_biomed_rna_plugin.biomed_rna import RnaProcessorItems
from vllm_biomed_rna_plugin.constants import (
    POOLING_METHOD_DEFAULT_ID,
    POOLING_METHOD_IDS,
    POOLING_METHOD_NAMES,
)


def _encode(method):
    """Helper: encode a pooling_method through RnaProcessorItems and return the int ID."""
    item = {
        "gene_ids": torch.tensor([1, 2, 3]),
        "expr_values": torch.tensor([1.0, 2.0, 3.0]),
        "attention_mask": torch.tensor([True, True, True]),
        "pooling_method": method,
    }
    items = RnaProcessorItems([item])
    return items.data["pooling_method_id"][0].item()


def test_encoding_first_token():
    assert _encode("first_token") == POOLING_METHOD_IDS["first_token"]


def test_encoding_mean_pooling():
    assert _encode("mean_pooling") == POOLING_METHOD_IDS["mean_pooling"]


def test_encoding_pooling_layer():
    assert _encode("pooling_layer") == POOLING_METHOD_IDS["pooling_layer"]


def test_encoding_int_token_position():
    """A non-negative integer is a token-position and passes through unchanged."""
    assert _encode(5) == 5
    assert _encode(0) == 0  # position 0 is valid — not confused with any named method


def test_encoding_none_uses_default():
    """Missing pooling_method → model default (first_token = POOLING_METHOD_DEFAULT_ID)."""
    item = {
        "gene_ids": torch.tensor([1, 2, 3]),
        "expr_values": torch.tensor([1.0, 2.0, 3.0]),
        "attention_mask": torch.tensor([True, True, True]),
    }
    items = RnaProcessorItems([item])
    assert items.data["pooling_method_id"][0].item() == POOLING_METHOD_DEFAULT_ID


def test_decode_round_trip():
    """POOLING_METHOD_NAMES is an exact inverse of POOLING_METHOD_IDS."""
    for name, code in POOLING_METHOD_IDS.items():
        assert POOLING_METHOD_NAMES[code] == name


def test_encoding_negative_int_raises():
    """Negative integers are reserved for named methods — should raise."""
    with pytest.raises(ValueError, match="must be >= 0"):
        _encode(-1)


def test_encoding_unknown_string_warns_and_uses_default(caplog):
    """Unknown string method falls back to model default and emits a warning."""
    import logging

    with caplog.at_level(logging.WARNING, logger="vllm_biomed_rna_plugin.biomed_rna"):
        result = _encode("not_a_real_method")
    assert result == POOLING_METHOD_DEFAULT_ID
    assert "not_a_real_method" in caplog.text
