"""Utility functions for BiomedRNA vLLM plugin."""

import json
from pathlib import Path
from typing import Any

from bmfm_targets.config.tokenization_config import FieldInfo
from vllm import LLM

# Available BiomedRNA model repositories
WCED_MULTITASK_MODEL = "ibm-research/biomed.rna.llama.47m.wced.multitask.v1.vllm"
MLM_MULTITASK_MODEL = "ibm-research/biomed.rna.llama.32m.mlm.multitask.v1.vllm"


def load_tokenizer(model_repo: str):
    """
    Load tokenizer from HuggingFace model repository.

    Args:
    ----
        model_repo: HuggingFace model repository ID

    Returns:
    -------
        MultiFieldTokenizer from bmfm-targets

    Example:
    -------
        >>> from vllm_biomed_rna_plugin.utils import load_tokenizer, WCED_MULTITASK_MODEL
        >>> tokenizer = load_tokenizer(WCED_MULTITASK_MODEL)
        >>> # Now ready to use with preprocess_h5ad()
    """
    from huggingface_hub import snapshot_download

    from bmfm_targets.tokenization import load_tokenizer as bmfm_load_tokenizer

    # Download model files from HuggingFace
    model_dir = snapshot_download(
        model_repo,
        allow_patterns=[
            "config.json",
            "*/tokenizer_config.json",
            "*/vocab.json",
            "*/tokenizer.json",
            "*/special_tokens_map.json",
        ],
    )
    return bmfm_load_tokenizer(model_dir)


def get_fields(model_repo: str) -> list[FieldInfo]:
    """
    Load model fields from HuggingFace model repository config.json.

    Args:
    ----
        model_repo: HuggingFace model repository ID

    Returns:
    -------
        Model fields configuration parsed as FieldInfo objects
    """
    from huggingface_hub import snapshot_download

    # Download config.json from HuggingFace
    model_path = Path(snapshot_download(model_repo, allow_patterns=["config.json"]))

    config_path = model_path / "config.json"
    with config_path.open() as f:
        config = json.load(f)

    return [FieldInfo(**field) for field in config["fields"]]


def get_vllm_biomed_rna_model(
    model_repo: str = WCED_MULTITASK_MODEL,
    **kwargs: Any,
) -> LLM:
    """
    Get a vLLM instance configured for BiomedRNA model from HuggingFace.

    This helper function provides sensible defaults for BiomedRNA model parameters.
    All parameters can be overridden via kwargs.

    Args:
    ----
        model_repo: HuggingFace model repository ID (default: WCED_MULTITASK_MODEL)
        **kwargs: Additional arguments to override defaults or pass to LLM

    Returns:
    -------
        Configured LLM instance ready for BiomedRNA inference

    Examples:
    --------
        # Use default WCED model
        >>> llm = get_vllm_biomed_rna_model()

        # Use MLM model
        >>> llm = get_vllm_biomed_rna_model(MLM_MULTITASK_MODEL)

        # Tests - minimal resources
        >>> llm = get_vllm_biomed_rna_model(
        ...     gpu_memory_utilization=0.01,
        ...     num_gpu_blocks_override=1,
        ... )

        # Production - more resources
        >>> llm = get_vllm_biomed_rna_model(
        ...     gpu_memory_utilization=0.1,
        ...     num_gpu_blocks_override=8,
        ... )

    """
    # Default params for offline batch embedding (.embed() API)
    # NOTE: vLLM will auto-resolve runner to "pooling" for embedding models
    default_params = {
        "model": model_repo,
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
