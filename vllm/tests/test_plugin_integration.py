#!/usr/bin/env python3
"""
Integration test for BiomedRNA vLLM plugin.

Tests plugin installation, registration, and vLLM model loading.
This validates the complete plugin integration with vLLM.
"""

import pytest


def test_plugin_installation():
    """Test that the plugin is properly installed and discoverable."""
    from importlib.metadata import entry_points

    import vllm_biomed_rna_plugin

    # Check registration function
    from vllm_biomed_rna_plugin import register_biomed_rna_model

    eps = entry_points()
    vllm_plugins = list(eps.select(group="vllm.general_plugins"))

    biomedrna_found = any(ep.name == "biomedrna" for ep in vllm_plugins)
    assert (
        biomedrna_found
    ), "Plugin entry point 'biomedrna' not found in vllm.general_plugins"


def test_vllm_model_loading():
    """Test loading the model through vLLM (validates registration and weights)."""
    import torch

    from vllm_biomed_rna_plugin import get_vllm_biomed_rna_model
    from vllm_biomed_rna_plugin.utils import DEFAULT_MODEL_PATH

    # Skip test if CUDA is not available (vLLM requires GPU for full initialization)
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available - vLLM requires GPU for full model loading")

    if not DEFAULT_MODEL_PATH.exists():
        pytest.skip(f"Local model not found at {DEFAULT_MODEL_PATH}")

    llm = get_vllm_biomed_rna_model(
        gpu_memory_utilization=0.01,
    )

    assert llm is not None, "Failed to create LLM instance"

    # Test actual embedding generation with dummy RNA data
    seq_len = 100
    dummy_input = {
        "prompt_token_ids": [1],  # Single fake token
        "multi_modal_data": {
            "rna": {
                "gene_ids": torch.randint(0, 19321, (seq_len,)).long(),
                "expr_values": torch.randn(seq_len).clamp(
                    min=0.1
                ),  # float32 by default
            }
        },
    }

    outputs = llm.embed([dummy_input])
    assert len(outputs) == 1, "Expected 1 output"
    assert outputs[0].outputs.embedding is not None, "Expected embedding output"
