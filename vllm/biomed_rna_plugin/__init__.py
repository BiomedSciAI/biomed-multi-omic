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


from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding


class NoOpTokenizer(PreTrainedTokenizerFast):
    """Dummy tokenizer for models that don't use text tokenization."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls._make_instance()

    @classmethod
    def _from_pretrained(cls, *args, **kwargs):
        # This is what AutoTokenizer actually calls — bypass all file loading
        return cls._make_instance()

    @classmethod
    def _make_instance(cls):
        # Build a real backend tokenizer so PreTrainedTokenizerFast is happy
        backend = Tokenizer(BPE())
        instance = object.__new__(cls)
        PreTrainedTokenizerFast.__init__(instance, tokenizer_object=backend)
        return instance

    def __call__(self, text="", **kwargs):
        return BatchEncoding({"input_ids": [[1]], "attention_mask": [[1]]})

    def encode(self, text, **kwargs):
        return [1]

    def decode(self, token_ids, **kwargs):
        return ""

    def get_vocab(self):
        return {"[PAD]": 0, "[UNK]": 1, "[DUMMY]": 2}


def register_biomed_rna_model() -> None:
    """
    Register Biomed-RNA models with vLLM's ModelRegistry and Transformers AutoConfig.

    This function is called automatically when the plugin is loaded.

    Registers:
    - AutoConfig for "scllama" model_type (from checkpoint)
    - AutoConfig for "biomedrna" model_type (for future use)
    - BiomedRnaForSequenceEmbedding with vLLM's ModelRegistry
    """
    logger = None

    try:
        from transformers import AutoConfig, AutoTokenizer
        from vllm.logger import init_logger
        from vllm.model_executor.models.registry import ModelRegistry

        from bmfm_targets.models.predictive.llama.config import LlamaForMultiTaskConfig

        logger = init_logger(__name__)

        # Register LlamaForMultiTaskConfig to handle the "scllama" model_type
        # This is critical for both offline and online (server) modes
        AutoConfig.register("scllama", LlamaForMultiTaskConfig, exist_ok=True)

        # Register dummy tokenizer to prevent AutoTokenizer crashes
        AutoTokenizer.register(
            LlamaForMultiTaskConfig, fast_tokenizer_class=NoOpTokenizer
        )

        # Register the model with vLLM's ModelRegistry
        ModelRegistry.register_model(
            "BiomedRnaForSequenceEmbedding",
            "biomed_rna_plugin.biomed_rna:BiomedRnaForSequenceEmbedding",
        )

        logger.info(
            "BiomedRNA plugin loaded successfully - registered BiomedRnaForSequenceEmbedding"
        )

    except Exception as e:
        if logger is not None:
            logger.error(f"Failed to load Biomed-RNA plugin: {e}")
        raise


# # Call registration when module is imported
# register_biomed_rna_model()


__all__ = [
    "register_biomed_rna_model",
    "get_vllm_biomed_rna_model",
    "__version__",
]

# Import utility function for convenience
from biomed_rna_plugin.utils import get_vllm_biomed_rna_model
