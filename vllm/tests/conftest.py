"""Shared pytest fixtures for BiomedRNA tests."""

import os
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig
from vllm_biomed_rna_plugin import register_biomed_rna_model
from vllm_biomed_rna_plugin.utils import MLM_MULTITASK_MODEL, WCED_MULTITASK_MODEL

# Repo root — needed so `from run.migrate_checkpoints_to_multitask import ...`
# inside bmfm_targets resolves. The editable install maps only the package dirs
# (bmfm_targets/, vllm_biomed_rna_plugin/) but not the repo root itself.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def pytest_configure(config):
    """Configure pytest and set environment variables for PyTorch."""
    # Register the BiomedRNA plugin (AutoConfig "scllama", vLLM ModelRegistry)
    # Must happen before any fixture calls AutoConfig.from_pretrained().
    register_biomed_rna_model()

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


def _make_vllm_fixture(model_repo: str):
    """Factory for session-scoped vLLM model fixtures."""

    @pytest.fixture(scope="session")
    def make_fixture():
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available - vLLM requires GPU")

        from vllm_biomed_rna_plugin import get_vllm_biomed_rna_model

        try:
            llm = get_vllm_biomed_rna_model(
                model_repo=model_repo,
                gpu_memory_utilization=0.01,
                disable_log_stats=True,
                dtype="float32",
                max_num_seqs=8,
            )
        except (RuntimeError, Exception) as e:
            msg = str(e)
            if (
                "Device string must not be empty" in msg
                or "Engine core initialization failed" in msg
            ):
                pytest.skip(f"No GPU device available for vLLM: {msg}")
            raise

        yield llm

        del llm
        torch.cuda.empty_cache()

    return make_fixture


# Session-scoped fixtures — one per model, loaded once for the entire test session.
vllm_model = _make_vllm_fixture(WCED_MULTITASK_MODEL)
vllm_model_mlm = _make_vllm_fixture(MLM_MULTITASK_MODEL)


@pytest.fixture()
def mock_vllm_config(config):
    """Mock vLLM config for IO processor tests (no GPU needed)."""
    return create_dummy_vllm_config(config)
