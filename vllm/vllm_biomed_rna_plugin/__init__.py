"""
vLLM BiomedRNA model plugin.

This plugin registers the BiomedRNA model with vLLM's ModelRegistry.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.model_executor.models.registry import ModelRegistry
else:
    ModelRegistry = Any

__version__ = "0.1.0"


def register_biomed_rna_model() -> None:
    """
    Register Biomed-RNA models with vLLM's ModelRegistry.

    This function is called automatically when the plugin is loaded.

    Registers BiomedRnaForSequenceEmbedding for scRNA pooling/embedding tasks.
    """
    logger = None

    try:
        from vllm.logger import init_logger
        from vllm.model_executor.models.registry import ModelRegistry

        logger = init_logger(__name__)

        ModelRegistry.register_model(
            "BiomedRnaForSequenceEmbedding",
            "vllm_biomed_rna_plugin.biomed_rna:BiomedRnaForSequenceEmbedding",
        )

        logger.info(
            "BiomedRNA plugin loaded successfully - registered BiomedRnaForSequenceEmbedding"
        )

    except Exception as e:
        if logger is not None:
            logger.error(f"Failed to load Biomed-RNA plugin: {e}")
        raise


__all__ = [
    "register_biomed_rna_model",
    "get_vllm_biomed_rna_model",
    "__version__",
]

# Import utility function for convenience
from vllm_biomed_rna_plugin.utils import get_vllm_biomed_rna_model
