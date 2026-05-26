"""Shared pytest fixtures for BiomedRNA tests."""

import os
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig

from vllm_biomed_rna_plugin.biomed_rna import (
    BiomedRnaConfig,
    BiomedRnaForSequenceEmbedding,
)


def pytest_configure(config):
    """Configure pytest and set environment variables for PyTorch."""
    # Disable TorchInductor compilation warnings
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    os.environ["TORCH_COMPILE_DEBUG"] = "0"

    # Use eager mode to avoid compilation issues
    torch._dynamo.config.suppress_errors = True

    # Set deterministic behavior
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


LOCAL_MODEL_PATH = Path(
    "/dccstor/bmfm-targets1/users/sivanra/models/biomed.rna.llama.47m.wced.multitask.v1"
)
HF_MODEL_PATH = "ibm-research/biomed.rna.llama.47m.wced.multitask.v1"

__all__ = [
    "create_dummy_vllm_config",
    "create_rna_multi_modal_object",
    "config",
    "model",
]


def create_rna_multi_modal_object(
    gene_ids: torch.Tensor, expr_values: torch.Tensor
) -> dict:
    """Create a multimodal data object for RNA input."""
    return {
        "rna": {
            "gene_ids": gene_ids,
            "expr_values": expr_values,
        }
    }


def create_dummy_vllm_config(config: BiomedRnaConfig):
    """Create minimal vLLM config for testing."""

    class DummyPoolerConfig:
        seq_pooling_type = "CLS"

    class DummyMultiModalConfig:
        """Dummy multimodal config for testing."""

        # Required by SupportsMultiModal interface
        mm_encoder_only = False

        def get_limit_per_prompt(self, modality: str) -> int | None:
            """Return None to indicate no limit for the modality."""
            return None

    class DummyModelConfig:
        def __init__(self, hf_config):
            self.hf_config = hf_config
            self.dtype = torch.float32
            self.head_dtype = torch.float32
            self.pooler_config = DummyPoolerConfig()
            self.multimodal_config = DummyMultiModalConfig()

    class DummyVllmConfig:
        def __init__(self, hf_config):
            self.model_config = DummyModelConfig(hf_config)

    return DummyVllmConfig(config)


@pytest.fixture(scope="module")
def config():
    return AutoConfig.from_pretrained(LOCAL_MODEL_PATH)


@pytest.fixture(scope="module")
def model(config):
    """Pytest fixture for BiomedRNA model with loaded weights."""
    from safetensors.torch import load_file

    # Load weights
    weights = load_file(str(LOCAL_MODEL_PATH / "model.safetensors"))

    # Create model with full config
    vllm_config = create_dummy_vllm_config(config)
    model = BiomedRnaForSequenceEmbedding(vllm_config=vllm_config)
    model.load_weights(weights.items())
    model.eval()
    return model
