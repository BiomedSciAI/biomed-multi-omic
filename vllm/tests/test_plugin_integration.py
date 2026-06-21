#!/usr/bin/env python3
"""
Integration test for BiomedRNA vLLM plugin.

Tests plugin installation, registration, and vLLM model loading.
This validates the complete plugin integration with vLLM.
"""


def test_plugin_installation():
    """Test that the plugin is properly installed and discoverable."""
    from importlib.metadata import entry_points

    # Check registration function

    eps = entry_points()
    vllm_plugins = list(eps.select(group="vllm.general_plugins"))

    biomedrna_found = any(ep.name == "biomed_rna_model" for ep in vllm_plugins)
    assert (
        biomedrna_found
    ), "Plugin entry point 'biomed_rna_model' not found in vllm.general_plugins"


def test_vllm_model_loading(vllm_model):
    """Test loading the model through vLLM (validates registration and weights)."""
    import torch

    assert vllm_model is not None, "Failed to create LLM instance"

    # Test actual embedding generation with dummy RNA data
    seq_len = 100
    torch.manual_seed(42)
    dummy_input = {
        "prompt_token_ids": [0] * seq_len,
        "multi_modal_data": {
            "rna": {
                "gene_ids": torch.randint(0, 19321, (seq_len,)).long(),
                "expr_values": (torch.randn(seq_len) * 2 + 3).clamp(min=0.1).float(),
                "attention_mask": torch.ones(seq_len, dtype=torch.bool),
            }
        },
    }

    outputs = vllm_model.embed([dummy_input])
    assert len(outputs) == 1, "Expected 1 output"
    assert outputs[0].outputs.embedding is not None, "Expected embedding output"
