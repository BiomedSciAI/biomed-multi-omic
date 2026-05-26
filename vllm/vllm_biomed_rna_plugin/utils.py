"""Utility functions for BiomedRNA vLLM plugin."""

import os
from pathlib import Path
from typing import Any

import torch
from vllm import LLM

# Default model repository
DEFAULT_HF_MODEL_REPO = "ibm-research/biomed.rna.llama.47m.wced.multitask.v1"

# Default local model path
DEFAULT_MODEL_PATH = Path(
    "/dccstor/bmfm-targets1/users/sivanra/models/biomed.rna.llama.47m.wced.multitask.v1"
)


def load_tokenizer(model_repo: str = DEFAULT_HF_MODEL_REPO):
    """
    Load tokenizer from HuggingFace checkpoint.

    Args:
    ----
        model_repo: HuggingFace model repository ID (default: biomed.rna.llama.47m.wced.multitask.v1)

    Returns:
    -------
        MultiFieldTokenizer from bmfm-targets

    Example:
    -------
        >>> from vllm_biomed_rna_plugin.utils import load_tokenizer
        >>> tokenizer = load_tokenizer()
        >>> # Now ready to use with preprocess_h5ad()
    """
    from bmfm_targets.models.model_utils import download_ckpt_from_huggingface
    from bmfm_targets.tokenization import load_tokenizer as bmfm_load_tokenizer

    checkpoint_path = download_ckpt_from_huggingface(model_repo)
    return bmfm_load_tokenizer(os.path.dirname(checkpoint_path))


def get_fields(model_repo: str = DEFAULT_HF_MODEL_REPO):
    """
    Load model fields from HuggingFace checkpoint.

    Args:
    ----
        model_repo: HuggingFace model repository ID (default: biomed.rna.llama.47m.wced.multitask.v1)

    Returns:
    -------
        Model fields configuration from checkpoint

    Example:
    -------
        >>> from vllm_biomed_rna_plugin.utils import get_fields
        >>> fields = get_fields()
        >>> # Now ready to use with DataModule directly
    """
    from bmfm_targets.models.model_utils import download_ckpt_from_huggingface

    checkpoint_path = download_ckpt_from_huggingface(model_repo)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return ckpt["hyper_parameters"]["model_config"].fields


def get_vllm_biomed_rna_model(
    model_path: str | Path | None = None,
    **kwargs: Any,
) -> LLM:
    """
    Get a vLLM instance configured for BiomedRNA model.

    This helper function provides sensible defaults for BiomedRNA model parameters.
    All parameters can be overridden via kwargs.

    Args:
    ----
        model_path: Path to the BiomedRNA model directory. If None, uses DEFAULT_MODEL_PATH.
        **kwargs: Additional arguments to override defaults or pass to LLM

    Returns:
    -------
        Configured LLM instance ready for BiomedRNA inference

    Examples:
    --------
        # Use default model path
        >>> llm = get_vllm_biomed_rna_model()

        # Specify custom model path
        >>> llm = get_vllm_biomed_rna_model("/path/to/model")

        # Tests - minimal resources
        >>> llm = get_vllm_biomed_rna_model(
        ...     gpu_memory_utilization=0.01,
        ...     num_gpu_blocks_override=1,
        ... )

        # Production - more resources
        >>> llm = get_vllm_biomed_rna_model(
        ...     model_path,
        ...     gpu_memory_utilization=0.1,
        ...     num_gpu_blocks_override=8,
        ... )

    """
    # Use default model path if not provided
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    # Default params
    default_params = {
        "model": str(model_path),
        "runner": "pooling",
        "trust_remote_code": True,
        "dtype": "float32",
        "mm_encoder_only": True,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "max_model_len": 1024,
        "gpu_memory_utilization": 0.05,
        "num_gpu_blocks_override": 4,
        "hf_overrides": {
            "architectures": ["BiomedRnaForSequenceEmbedding"],
            "use_cache": False,
            "max_position_embeddings": 2048,
        },
    }

    # User kwargs override defaults
    final_params = {**default_params, **kwargs}

    return LLM(**final_params)
