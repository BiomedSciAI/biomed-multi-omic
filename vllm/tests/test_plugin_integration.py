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
