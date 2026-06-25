"""Shared pytest fixtures for BiomedRNA tests."""

import os

import pytest
import torch
from transformers import AutoConfig
from vllm_biomed_rna_plugin.utils import WCED_MULTITASK_MODEL


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


# Use the centralized model path from utils
MODEL_PATH = WCED_MULTITASK_MODEL

__all__ = [
    "create_dummy_vllm_config",
    "create_rna_multi_modal_object",
    "config",
    "vllm_model",
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


def create_dummy_vllm_config(config):
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
    return AutoConfig.from_pretrained(MODEL_PATH)


@pytest.fixture(scope="session")
def vllm_model():
    """Session-scoped vLLM model - initialized once for all tests."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available - vLLM requires GPU")

    from vllm_biomed_rna_plugin import get_vllm_biomed_rna_model

    llm = get_vllm_biomed_rna_model(
        gpu_memory_utilization=0.01,  # Minimal memory for tests
        disable_log_stats=True,
        dtype="float32",
        max_num_seqs=8,  # Support batching
    )

    yield llm

    # Cleanup
    del llm
    torch.cuda.empty_cache()


@pytest.fixture()
def mock_vllm_config(config):
    """Mock vLLM config for IO processor tests (no GPU needed)."""
    return create_dummy_vllm_config(config)
