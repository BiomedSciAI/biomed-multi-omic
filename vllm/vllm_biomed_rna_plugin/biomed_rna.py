"""
BiomedRNA vLLM plugin - wraps IBM Biomed-RNA model for single-cell RNA expression analysis.
Model: ibm-research/biomed.rna.llama.47m.wced.multitask.v1.
"""

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from bmfm_targets.models.predictive.llama.config import LlamaForMultiTaskConfig
from transformers import BatchFeature, PretrainedConfig

logger = logging.getLogger(__name__)


from collections.abc import Set

from bmfm_targets.models.predictive.layers import (
    get_embeddings_from_outputs,
)
from bmfm_targets.models.predictive.llama.model import LlamaForMultiTaskModel
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict, MultiModalInput, mm_input
from vllm.model_executor.layers.pooler.abstract import Pooler, PoolerOutput
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.multimodal.parse import (
    DictEmbeddingItems,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    ProcessorInputs,
    PromptUpdate,
    TimingContext,
)
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata


class EmbeddingIdentityPooler(Pooler):
    """
    Pooler that formats already-pooled embeddings for vLLM's embedding API.

    The BMFM model's forward() already applies pooling via get_embeddings_from_outputs()
    and returns shape [batch, 1, hidden_size]. This pooler just needs to:
    1. Squeeze out the sequence dimension to get [batch, hidden_size]
    2. Return a list of 1-D tensors (one per batch item) as vLLM expects

    TODO: Refactor to proper vLLM pooler architecture
    - Move pooling logic from forward() to pooler
    - Return unpooled hidden_states [batch, seq_len, hidden_size] from forward()
    - Implement multiple pooling strategies (first_token, mean, max, etc.) in pooler
    - Support runtime pooling method selection via config
    """

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"embed"}

    # todo: extract handling of hidden states from the get_embeddings_from_output
    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        # hidden_states shape: [batch, 1, hidden_size] (already pooled by model)
        # Squeeze out the sequence dimension: [batch, hidden_size]
        embeddings = hidden_states.squeeze(1)

        # Return list of 1-D embedding vectors (vLLM requirement)
        return [embeddings[i] for i in range(embeddings.shape[0])]


class BiomedRnaConfig(PretrainedConfig):
    """Configuration class for BiomedRNA models."""

    model_type = "biomedrna"

    # Default local model path
    _default_local_path = "/dccstor/bmfm-targets1/users/sivanra/models/biomed.rna.llama.47m.wced.multitask.v1"

    def __init__(
        self,
        hidden_size: int = 384,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 1536,
        activation_function: str = "GELU",
        gene_vocab_size: int = 19321,
        pooling_method: str = "first_token",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.activation_function = activation_function
        self.gene_vocab_size = gene_vocab_size
        self.pooling_method = pooling_method

        # Encoder-only model configuration: disable KV cache


class RnaProcessorItems(DictEmbeddingItems):
    """
    RNA data items — wraps a list of dicts with gene_ids/expr_values.

    This class inherits from DictEmbeddingItems to indicate that RNA data
    is already in a processed form (like embeddings) and doesn't require
    placeholder tokens in the text prompt.

    By inheriting from DictEmbeddingItems, vLLM will skip prompt update
    validation for RNA modality (_hf_processor_applies_updates returns False).

    data is a dict with keys gene_ids, expr_value, attention_mask
    """

    def __init__(self, data):
        # wrap single dicts to list to match batches of multiple requests
        if not isinstance(data, list):
            data = [data]

        # DictEmbeddingItems parent class expects fields as lists
        combined_data = {}
        for field in ["gene_ids", "expr_values", "attention_mask"]:
            field_list = [item.get(field) for item in data if field in item]
            if field_list:
                combined_data[field] = field_list

        # Define fields factory
        def fields_factory(data_dict):
            fields = {
                "gene_ids": MultiModalFieldConfig.batched("rna"),
                "expr_values": MultiModalFieldConfig.batched("rna"),
            }
            if "attention_mask" in data_dict:
                fields["attention_mask"] = MultiModalFieldConfig.batched("rna")
            return fields

        super().__init__(
            data=combined_data,
            modality="rna",
            required_fields={"gene_ids", "expr_values"},
            fields_factory=fields_factory,
        )

    def get_processor_data(self):
        """Not passed to an HF processor."""
        return {}

    def get_passthrough_data(self):
        """
        Return RNA data to be accessed in _get_mm_fields_config.

        The data is already stored in self.data (from DictEmbeddingItems parent),
        which is the combined_data dict with fields as lists.
        """
        return (
            dict(self.data) if self.data else {}
        )  # todo - might be possible to just rutn self.data as is


class BiomedRnaDataParser(MultiModalDataParser):
    """Custom parser for 'rna' modality."""

    def _parse_rna_data(self, data) -> ModalityDataItems:
        # RnaProcessorItems handles both dict and list of dicts
        # data (dict): {"gene_ids" : Tensor[len_seq], "expr_values" : Tensor[len_seq], "attention_mask" : Tensor[len_seq[], optional
        return RnaProcessorItems(data)

    def _get_subparsers(self) -> dict[str, Any]:
        subparsers = super()._get_subparsers()
        subparsers["rna"] = self._parse_rna_data
        return subparsers


class BiomedRnaDummyProcessor:
    """
    Dummy HF processor for RNA model.

    RNA data doesn't need HF processor - it's handled directly via
    embed_multimodal(). This processor just tokenizes text.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _merge_kwargs(self, *args, **kwargs):
        """Stub for HF processor compatibility."""
        return {}

    def __call__(self, text=None, **kwargs):
        """Tokenize text only, ignore multi-modal data."""
        from transformers import BatchFeature

        if text is None:
            text = ""

        # Filter out RNA-specific kwargs
        tokenizer_kwargs = {
            k: v for k, v in kwargs.items() if k not in ["gene_ids", "expr_values"]
        }

        result = self.tokenizer(text, **tokenizer_kwargs)

        # Return BatchFeature as expected by vLLM
        return BatchFeature(result)


class BiomedRnaProcessingInfo(BaseProcessingInfo):
    """Processing information for BiomedRNA model."""

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"rna": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        # RNA doesn't add placeholder tokens to the text sequence
        return {"rna": 0}

    def get_hf_processor(self, **kwargs):
        """Return custom processor that only tokenizes text."""
        tokenizer = self.ctx.get_tokenizer()
        return BiomedRnaDummyProcessor(tokenizer)

    def parse_mm_data(
        self,
        mm_data: MultiModalDataDict,
        validate: bool = True,
    ) -> MultiModalDataItems:
        """Override to handle the custom 'rna' modality."""
        return BiomedRnaDataParser().parse_mm_data(mm_data)


class BiomedRnaDummyInputsBuilder(BaseDummyInputsBuilder):
    """Dummy inputs builder for BiomedRNA model (represent worst-case memory usage)."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        """Get dummy text for RNA modality."""
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        """
        Generate dummy RNA data for memory profiling.

        Args:
        ----
            seq_len: Target sequence length (used as RNA sequence length)
            mm_counts: Mapping of modality names to counts
            mm_options: Mapping of modality names to dummy options

        Returns:
        -------
            Dictionary containing dummy RNA data with gene_ids and expr_values
        """
        num_rna = mm_counts.get("rna", 0)

        if num_rna == 0:
            return {}

        gene_ids = torch.randint(0, 19321, (seq_len,), dtype=torch.long)
        expr_values = torch.randn(seq_len, dtype=torch.float32)

        return {
            "rna": {
                "gene_ids": gene_ids,
                "expr_values": expr_values,
            }
        }


class BiomedRnaMultiModalProcessor(BaseMultiModalProcessor):
    """
    Multi-modal processor for BiomedRNA model.

    Handles the processing of RNA data (gene IDs and expression values)
    for the BiomedRNA model.
    """

    def _get_data_parser(self) -> MultiModalDataParser:
        """Return custom data parser that registers RNA modality."""
        return BiomedRnaDataParser()

    def _get_mm_fields_config(
        self,
        hf_inputs: Mapping[str, object],
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """
        Define the schema of scRNA input.

        For RNA data, we have three fields: gene_ids, expr_values, and attention_mask (optional).

        Args:
        ----
            hf_inputs: HF processor outputs (not used for RNA)
            hf_processor_mm_kwargs: HF processor kwargs (not used for RNA)

        Returns:
        -------
            Mapping of field names to MultiModalFieldConfig
        """
        config = {
            "gene_ids": MultiModalFieldConfig.batched("rna"),
            "expr_values": MultiModalFieldConfig.batched("rna"),
        }
        # Add attention_mask if present in inputs
        if "attention_mask" in hf_inputs:
            config["attention_mask"] = MultiModalFieldConfig.batched("rna")
        return config

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """Return prompt updates for RNA data (empty since we dont use placeholders)."""
        return []

    def apply(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ) -> MultiModalInput:
        """
        Bypass standard validation for RNA modality.

        Creates BatchFeature with lists of variable-length tensors as passing is handled
        in forward. RNA data doesn't use placeholder tokens in the prompt,
        so we skip the standard HF processor pipeline and validation.
        """
        mm_items = inputs.mm_data_items
        hf_processor_mm_kwargs = inputs.hf_processor_mm_kwargs

        with timing_ctx.record("apply_hf_processor"):
            # Get passthrough data (gene_ids and expr_values)
            _, passthrough_data = self._get_hf_mm_data(mm_items)

            # Handle both single items and batches properly
            # Fix for ValueError: "Unable to create tensor, you should probably activate padding"
            # # This occurs during vLLM's dummy test when creating batches with variable-length sequences
            # processed_dict = {}
            # for k, v in passthrough_data.items():
            #     if isinstance(v, list):
            #         # List of tensors (batch) - need to stack them into a single tensor
            #         # During dummy test, vLLM may create multiple sequences with different lengths
            #         if len(v) > 0:
            #             # Check if all tensors have the same shape
            #             shapes = [t.shape for t in v]
            #             if len(set(shapes)) == 1:
            #                 # All same shape - can stack directly without padding
            #                 processed_dict[k] = torch.stack(v)
            #             else:
            #                 # Different shapes - need padding to make them uniform
            #                 # This happens during dummy batch creation with variable seq_len
            #                 # Find max length across all sequences in the batch
            #                 max_len = max(t.shape[0] for t in v)
            #                 # Pad all tensors to max length with zeros
            #                 padded = []
            #                 for t in v:
            #                     if t.shape[0] < max_len:
            #                         # Pad shorter sequences with zeros at the end
            #                         pad_size = max_len - t.shape[0]
            #                         padded_t = F.pad(t, (0, pad_size), value=0)
            #                         padded.append(padded_t)
            #                     else:
            #                         padded.append(t)
            #                 # Stack padded tensors into a single batch tensor
            #                 processed_dict[k] = torch.stack(padded)
            #         else:
            #             processed_dict[k] = torch.tensor([])
            #     else:
            #         # Single tensor - convert to tensor and ensure it has batch dimension
            #         tensor = torch.as_tensor(v)
            #         # Only add batch dimension if it's 1D (no batch dimension yet)
            #         # If it's already 2D [batch, seq_len], keep it as is
            #         if tensor.ndim == 1:
            #             processed_dict[k] = tensor.unsqueeze(0)
            #         else:
            #             processed_dict[k] = tensor

            # Use tensor_type=None to avoid automatic conversion that fails with variable lengths
            # mm_processed_data = BatchFeature(processed_dict, tensor_type=None)

            # Previous code (caused ValueError with variable-length batches):
            mm_processed_data = BatchFeature(
                {
                    k: torch.as_tensor(v).unsqueeze(0) if not isinstance(v, list) else v
                    for k, v in passthrough_data.items()
                },
                # tensor_type="pt",
                tensor_type=None,  # allow variable length, padding will be done in forward
            )
            # todo: consider this:
            # mm_processed_data = BatchFeature(
            # passthrough_data,
            # tensor_type=None,  # allow variable length, padding will be done in forward
        # )

        # Create MultiModalKwargsItems from processed data
        mm_kwargs = MultiModalKwargsItems.from_hf_inputs(
            mm_processed_data,
            self._get_mm_fields_config(
                mm_processed_data,
                hf_processor_mm_kwargs,
            ),
        )

        with timing_ctx.record("get_mm_hashes"):
            mm_hashes = inputs.get_mm_hashes(self.info.model_id)

        # Create empty placeholder range (no text placeholders for RNA)
        mm_placeholders = {"rna": [PlaceholderRange(offset=0, length=0)]}

        # Return mm_input with dummy prompt token (will be ignored)
        return mm_input(
            prompt_token_ids=[1],  # Dummy token ID
            mm_kwargs=mm_kwargs,
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
        )


class LlamaForMultiTaskModelNoCheckpoint(LlamaForMultiTaskModel):
    """wraps LlamaForMultiTaskModel without model loading on init."""

    def load_checkpoint(self):
        pass


@MULTIMODAL_REGISTRY.register_processor(
    BiomedRnaMultiModalProcessor,
    info=BiomedRnaProcessingInfo,
    dummy_inputs=BiomedRnaDummyInputsBuilder,
)
class BiomedRnaForSequenceEmbedding(nn.Module, SupportsMultiModal):
    """
    BiomedRNA model for sequence embedding generation.

    This model implements a custom LLaMa-based encoder for single-cell RNA analysis.
    It processes multi-field inputs (gene IDs + expression values) to generate cell embeddings.
    This is an encoder-only model with bidirectional attention.
    """

    supports_multimodal_raw_input_only = True
    is_pooling_model = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        """
        Return the placeholder string for RNA modality.

        RNA data doesn't use placeholder tokens in the text prompt,
        so we return None (similar to Terratorch).
        """
        if modality.startswith("rna"):
            return None

        raise ValueError("Only rna modality is supported")

    def get_data_key(self) -> str:
        return "rna"

    def __init__(
        self,
        vllm_config: object | None = None,
        prefix: str = "",
        **kwargs,
    ):
        """Initialize biomed-rna model."""
        super().__init__()

        logger = logging.getLogger(__name__)
        self.vllm_config = vllm_config
        hf_config = vllm_config.model_config.hf_config

        # # =====================================
        # # override vllm defaults
        # hf_config.use_cache = False
        # hf_config.architectures = ["BiomedRnaForSequenceEmbedding"]
        # hf_config.max_position_embeddings =  2048 #check if we need this

        # # hf_config.is_encoder_decoder = False
        # # hf_config.is_decoder = False
        # # hf_config.add_cross_attention = False
        # # ======================================

        with self._mark_tower_model(vllm_config, "rna"):
            # hf_config is already a LlamaForMultiTaskConfig instance loaded by vLLM
            # from the model's config.json via AutoConfig
            bmfm_config: LlamaForMultiTaskConfig = hf_config

            self.pooling_method = getattr(bmfm_config, "pooling_method", "first_token")
            self.hidden_size = bmfm_config.hidden_size

            self.model = LlamaForMultiTaskModelNoCheckpoint(bmfm_config)

            # Atempt remove unused decoder head, but it cause an exception in llama model.py line 205
            # if hasattr(self.model.cls.predictions.predictions, "decoder"):
            #     del self.model.cls.predictions.predictions.decoder

            self.model.eval()

            logger.info(
                "BMFM model loaded successfully via PyTorch Lightning (full model)"
            )

        self.pooler = EmbeddingIdentityPooler()

    @staticmethod
    def get_kv_cache_spec() -> dict:
        """No KV cache needed - encoder-only model."""
        return {}

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Return dummy embeddings with correct shape for multimodal-raw-input-only model.

        For models with supports_multimodal_raw_input_only=True, vLLM
        requires embeddings with the correct hidden dimension to copy into its buffer.
        The actual multimodal data is passed to forward() via **kwargs.

        Args:
        ----
            input_ids: Token IDs (used for shape and device)
            multimodal_embeddings: Ignored (data comes via forward kwargs)
            is_multimodal: Ignored

        Returns:
        -------
            Dummy embeddings with shape [input_ids.shape[0], hidden_size]
        """
        dummy_embeds = torch.zeros(
            input_ids.shape[0],
            self.hidden_size,
            dtype=next(self.parameters()).dtype,
            device=input_ids.device,
        )
        return dummy_embeds

    def _pad_variable_length_batch(
        self,
        gene_ids: list[torch.Tensor],
        expr_values: list[torch.Tensor],
        attn_masks: list[torch.Tensor] | None,
        pad_token_id: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pad variable-length sequences to uniform length for batching.

        Args:
        ----
            gene_ids: List of 1D gene ID tensors with varying lengths
            expr_values: List of 1D expression value tensors with varying lengths
            attn_masks: Optional list of 1D attention mask tensors
            pad_token_id: Token ID to use for padding gene_ids

        Returns:
        -------
            Tuple of (padded_gene_ids, padded_expr_values, padded_attention_mask)
            All tensors have shape [batch_size, max_len]
        """
        max_len = max(g.shape[0] for g in gene_ids)
        batch_size = len(gene_ids)
        device = gene_ids[0].device

        # Create padded tensors
        gene_ids_padded = torch.full(
            (batch_size, max_len), pad_token_id, dtype=torch.long, device=device
        )
        expr_padded = torch.zeros(
            batch_size, max_len, dtype=torch.float32, device=device
        )
        mask_padded = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)

        # Fill in actual values
        for i, (g, e) in enumerate(zip(gene_ids, expr_values)):
            length = g.shape[0]
            gene_ids_padded[i, :length] = g.long()
            expr_padded[i, :length] = e.float()
            mask_padded[i, :length] = (
                attn_masks[i].bool() if attn_masks is not None else True
            )

        return gene_ids_padded, expr_padded, mask_padded

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attn_metadata=None,
        **kwargs,
    ) -> torch.Tensor:
        """
            Generate cell embeddings from RNA expression data.

            Processes single-cell RNA-seq data through the BiomedRNA encoder and returns
            pooled embeddings. Handles both pre-padded batches (offline processing) and
            variable-length batches (online serving).
            variable-length batches are padded to uniform length before passing to encoder.

        Args:
        ----
                input_ids: Unused (RNA data provided via kwargs)
                positions: Unused
                inputs_embeds: Unused
                attn_metadata: Unused
                **kwargs: RNA data containing:
                    gene_ids: Gene identifiers
                        - Pre-padded: torch.Tensor [batch_size, seq_len]
                        - Variable-length: list[torch.Tensor] of shapes [seq_len_i]
                        Values: Integer gene IDs
                    expr_values: Expression values (same type/shape as gene_ids)
                        Values: Float expression levels (typically log-normalized)
                    attention_mask (optional): Attention mask (same type/shape as gene_ids)
                        Values: 1 for real tokens, 0 for padding

        Returns:
        -------
                torch.Tensor: Pooled cell embeddings [batch_size, 1, hidden_size]

        TODO: Refactor to return unpooled hidden states
        - Currently returns pooled embeddings [batch, 1, hidden_size]
        - Should return unpooled hidden_states [batch, seq_len, hidden_size]
        - Move pooling logic to EmbeddingIdentityPooler (rename to BiomedRnaPooler)
        - This will enable:
        * Multiple pooling strategies (first_token, mean, max, etc.)
        * Classification task support
        * Runtime pooling method selection
        """
        PAD_TOKEN_ID = 2

        if "gene_ids" not in kwargs or "expr_values" not in kwargs:
            raise ValueError("gene_ids and expr_values are required in kwargs")

        gene_ids = kwargs["gene_ids"]
        expr_values = kwargs["expr_values"]
        attention_mask = kwargs.get("attention_mask", None)

        # todo: check if this can be simplified to same number of dimensions
        if isinstance(gene_ids, torch.Tensor):
            # pre-padded batch, same length
            logger.debug(
                f"Pre-padded batch: gene_ids shape={gene_ids.shape}, "
                f"dtype={gene_ids.dtype}"
            )
            # gene_ids = gene_ids.long()
            # expr_values = expr_values.float()
            if attention_mask is None:
                attention_mask = torch.ones_like(gene_ids, dtype=torch.bool)
            # else:
            #     attention_mask = attention_mask.bool()
        else:
            # Batched requests - variable length requests batched together by vLLM
            # todo: check if we can move padding to processor
            logger.debug(
                f"Variable-length batch: {len(gene_ids)} sequences, "
                f"lengths={[t.shape[0] for t in gene_ids]}, "
                f"dtype={gene_ids[0].dtype}"
            )

            gene_ids, expr_values, attention_mask = self._pad_variable_length_batch(
                gene_ids, expr_values, attention_mask, PAD_TOKEN_ID
            )

        # now we have [batch_size, seq_len]
        input_ids_bmfm = torch.stack(
            [gene_ids.float(), expr_values], dim=1
        )  # [batch_size, 2, seq_len]

        output = self.model(
            input_ids=input_ids_bmfm,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        embeddings = get_embeddings_from_outputs(
            output,
            attention_mask,
            pooling_method=self.pooling_method,
        )

        return embeddings.unsqueeze(1)  # [batch_size, 1, hidden_size]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """
        Load weights from safetensors into the biomed-rna model.

        Args:
        ----
            weights: Iterable of (name, tensor) pairs from vLLM
        """
        param_targets = dict(self.model.named_parameters())
        buffer_targets = dict(self.model.named_buffers())
        loaded = set()
        missing = set()

        for name, tensor in weights:
            clean = name.removeprefix("model.")
            if clean in param_targets:
                param_targets[clean].data.copy_(tensor)
                loaded.add(clean)
            elif clean in buffer_targets:
                buffer_targets[clean].data.copy_(tensor)
                loaded.add(clean)
            else:
                missing.add(clean)

        # debug reporting:
        # - unloaded: model params/buffers that kept their initialized values;
        # - missing: checkpoint keys not found in model; (usually noise from vLLM)
        all_targets = param_targets.keys() | buffer_targets.keys()
        if unloaded := all_targets - loaded:
            logger.warning(
                f"Weight items not loaded from checkpoint: {list(unloaded)[:10]}"
            )
        if missing:
            logger.warning(
                f"Checkpoint has extra weught items not in model: {list(missing)[:10]}"
            )


try:
    from transformers import AutoConfig

    AutoConfig.register("biomedrna", BiomedRnaConfig)
except Exception:
    pass
