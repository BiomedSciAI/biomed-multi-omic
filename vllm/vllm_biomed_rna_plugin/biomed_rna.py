"""
BiomedRNA vLLM plugin — wraps IBM Biomed-RNA model for single-cell RNA expression analysis.
Model: ibm-research/biomed.rna.llama.47m.wced.multitask.v1.

Pipeline:
    h5ad → preprocess.py (log-norm, gene filtering, cell-level padding)
         → list[dict] per cell, each with gene_ids / expr_values / attention_mask
         → vLLM (one request per cell)
         → BiomedRnaMultiModalProcessor.apply()  [per-request, CPU]
         → vLLM batches requests via MultiModalKwargsItems
         → BiomedRnaForSequenceEmbedding.forward()  [per-batch, GPU]
         → Tensor[batch, 1, hidden_size]

Passthrough-only policy:
    RNA data bypasses the HF processor entirely. There are no placeholder tokens
    in the text prompt — vLLM's token-alignment mechanism does not apply.
    The dummy prompt token [1] satisfies vLLM's requirement for a non-empty prompt.

Batching and padding:
    preprocess.py pads all cells in a DataModule batch to the same seq_len before
    submitting them to vLLM. Within a single preprocessing run, all cells therefore
    arrive at forward() with identical seq_len, and vLLM stacks them into a single
    Tensor[batch, seq_len] — no further padding needed.

    The exception is concurrent users: if two requests from different preprocessing
    runs reach the same vLLM batch, their seq_len values may differ. In that case
    vLLM cannot stack and instead passes a list[Tensor] to forward(), which then
    pads to the maximum length in that batch. This is handled in
    BiomedRnaForSequenceEmbedding._pad_variable_length_batch() and is expected to
    be rare in practice.

    To eliminate the variable-length case entirely, set max_num_seqs=1 when
    initializing vLLM (LLM(..., max_num_seqs=1)). This prevents vLLM from
    batching requests from different users together, guaranteeing that forward()
    always receives a uniform Tensor[batch, seq_len]. The tradeoff is reduced
    throughput under concurrent load.
"""

import logging
from collections.abc import Iterable, Mapping, Sequence, Set

import torch
import torch.nn as nn
from transformers import BatchFeature

from bmfm_targets.models.predictive.layers import get_embeddings_from_outputs
from bmfm_targets.models.predictive.llama.config import LlamaForMultiTaskConfig
from bmfm_targets.models.predictive.llama.model import LlamaForMultiTaskModel
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict, MultiModalInput, mm_input
from vllm.model_executor.layers.pooler.abstract import Pooler
from vllm.model_executor.models.interfaces import IsAttentionFree, SupportsMultiModal
from vllm.model_executor.models.interfaces_base import attn_type
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
from vllm.sequence import IntermediateTensors
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata
from vllm_biomed_rna_plugin.constants import RNA_FIELDS_CONFIG, RNA_PAD_TOKEN_ID

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pooler
# ---------------------------------------------------------------------------


class EmbeddingIdentityPooler(Pooler):
    """
    Formats already-pooled BMFM embeddings for vLLM's embedding API.

    forward() receives [batch, 1, hidden_size] (pooled by the model),
    squeezes the sequence dimension, and returns a list of 1-D tensors
    as required by vLLM's pooling protocol.

    TODO: Move pooling out of BiomedRnaForSequenceEmbedding.forward() into
    here, return unpooled [batch, seq_len, hidden_size] from the model, and
    support multiple pooling strategies (first_token, mean, max).
    """

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"embed"}

    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, 1, hidden_size]
        pooling_metadata: PoolingMetadata,
    ) -> list[torch.Tensor]:
        # squeeze sequence dim → [batch, hidden_size], split into per-sample list
        embeddings = hidden_states.squeeze(1)
        return [embeddings[i] for i in range(embeddings.shape[0])]


# ---------------------------------------------------------------------------
# Config - Use LlamaForMultiTaskConfig directly
# ---------------------------------------------------------------------------
# Note: LlamaForMultiTaskConfig inherits from SCModelConfigBase which already
# has from_dict() that properly deserializes FieldInfo and LabelColumnInfo
# objects from dicts. No wrapper needed!


# ---------------------------------------------------------------------------
# Multimodal data parsing
# ---------------------------------------------------------------------------


class RnaProcessorItems(DictEmbeddingItems):
    """
    Parse-time validation and normalization of single-cell RNA data.

    Accepts a single dict or a list of dicts, each with:
        gene_ids:       Tensor[seq_len]  long   — gene token IDs
        expr_values:    Tensor[seq_len]  float  — log-normalized expression
        attention_mask: Tensor[seq_len]  bool   — True for real genes, False for padding

    All three fields are required. Shape alignment and dtypes are enforced here
    so that downstream code can assume clean tensors.

    Inherits from DictEmbeddingItems to signal passthrough-only processing:
    data goes directly to mm_kwargs without passing through the HF processor.
    """

    def __init__(self, data: dict | list[dict]):
        if not isinstance(data, list):
            data = [data]

        validated = []
        for i, item in enumerate(data):
            for field in ("gene_ids", "expr_values", "attention_mask"):
                if field not in item:
                    raise ValueError(f"Item {i}: missing required field '{field}'")

            gene_ids = torch.as_tensor(item["gene_ids"]).long()
            expr_values = torch.as_tensor(item["expr_values"]).float()
            attention_mask = torch.as_tensor(item["attention_mask"]).bool()

            if not (
                gene_ids.shape[0] == expr_values.shape[0] == attention_mask.shape[0]
            ):
                raise ValueError(
                    f"Item {i}: field lengths must match, got "
                    f"gene_ids={gene_ids.shape[0]}, "
                    f"expr_values={expr_values.shape[0]}, "
                    f"attention_mask={attention_mask.shape[0]}"
                )

            validated.append(
                {
                    "gene_ids": gene_ids,
                    "expr_values": expr_values,
                    "attention_mask": attention_mask,
                }
            )

        # DictEmbeddingItems stores fields as lists (one tensor per item)
        combined = {
            field: [item[field] for item in validated]
            for field in ("gene_ids", "expr_values", "attention_mask")
        }

        super().__init__(
            data=combined,
            modality="rna",
            required_fields={"gene_ids", "expr_values", "attention_mask"},
            fields_factory=lambda _: RNA_FIELDS_CONFIG.copy(),
        )

    def get_processor_data(self) -> dict:
        """RNA bypasses the HF processor — nothing to process."""
        return {}

    def get_passthrough_data(self) -> dict:
        """
        Return validated RNA tensors for direct passthrough to mm_kwargs.
        Each value is a list of tensors, one per item in the request.
        """
        return dict(self.data)


class BiomedRnaDataParser(MultiModalDataParser):
    """Registers the 'rna' modality with vLLM's multimodal data parsing system."""

    def _parse_rna_data(self, data: dict | list[dict]) -> ModalityDataItems:
        return RnaProcessorItems(data)

    def _get_subparsers(self) -> dict:
        return {**super()._get_subparsers(), "rna": self._parse_rna_data}


# ---------------------------------------------------------------------------
# Processing info
# ---------------------------------------------------------------------------


class BiomedRnaProcessingInfo(BaseProcessingInfo):
    """
    Metadata for BiomedRNA multimodal processing.

    RNA is an encoder-only modality with no placeholder tokens in the prompt.
    The dummy prompt token [1] is required by vLLM but carries no information.
    All RNA data flows through mm_kwargs, not through the text token sequence.
    """

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"rna": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {"rna": 0}  # RNA adds no placeholder tokens to the prompt

    def parse_mm_data(
        self,
        mm_data: MultiModalDataDict,
        validate: bool = True,
    ) -> MultiModalDataItems:
        return BiomedRnaDataParser().parse_mm_data(mm_data)


# ---------------------------------------------------------------------------
# Dummy inputs builder
# ---------------------------------------------------------------------------


class BiomedRnaDummyInputsBuilder(BaseDummyInputsBuilder):
    """
    Generates worst-case dummy RNA inputs for vLLM memory profiling.

    seq_len is used as the RNA sequence length. vLLM calls this during
    model initialization to estimate peak memory usage.

    Uses a safe vocab size range and positive expression values to avoid
    CUDA index out of bounds errors during warmup.
    """

    def __init__(self, info):
        super().__init__(info)
        # Get vocab size from config to generate valid gene IDs
        config = info.get_hf_config()
        self.gene_vocab_size = 19321  # default

        if hasattr(config, "fields") and config.fields:
            # Find the genes field to get vocab size
            for field in config.fields:
                if hasattr(field, "field_name") and field.field_name == "genes":
                    if hasattr(field, "vocab_size") and field.vocab_size:
                        self.gene_vocab_size = field.vocab_size
                    break

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        # RNA is always one cell per request — num_rna > 1 is not expected
        if mm_counts.get("rna", 0) == 0:
            return {}

        # Generate valid gene IDs within vocab range
        # Use values that avoid special tokens (0, 1, 2 are typically special)
        max_gene_id = min(self.gene_vocab_size - 1, 19320)
        gene_ids = torch.randint(3, max_gene_id + 1, (seq_len,), dtype=torch.long)

        # Generate positive expression values (log-normalized range ~0-10)
        expr_values = torch.rand(seq_len) * 8.0 + 0.1

        return {
            "rna": {
                "gene_ids": gene_ids,
                "expr_values": expr_values,
                "attention_mask": torch.ones(seq_len, dtype=torch.bool),
            }
        }


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


def _unwrap_field(passthrough_data: dict, field: str) -> torch.Tensor:
    """
    Extract the single per-request tensor from a passthrough_data list.

    passthrough_data values are lists of tensors (one per item in the request).
    For RNA, each request is always one cell, so each list has exactly one tensor.
    """
    value = passthrough_data.get(field)
    if value is None:
        raise ValueError(f"Missing required RNA field: '{field}'")
    return value[0] if isinstance(value, list) else value


class BiomedRnaMultiModalProcessor(BaseMultiModalProcessor):
    """
    Per-request processor for RNA data.

    Called once per request (one cell) before vLLM assembles the batch.
    Extracts validated 1D tensors from passthrough_data and wraps them
    in the MultiModalKwargsItems structure that vLLM uses to batch requests.

    Overrides apply() because RNA has no placeholder tokens in the prompt.
    The base apply() enforces a 1:1 match between mm items and prompt placeholder
    tokens (e.g. <image>), which does not apply to RNA.

    After apply(), vLLM batches N per-request MultiModalKwargsItems together.
    When all requests have the same seq_len (common case), tensors are stacked
    into Tensor[N, seq_len]. When lengths differ (concurrent users from different
    preprocessing runs), tensors arrive as list[Tensor] and are padded in forward().

    See: vllm/multimodal/inputs.py MultiModalKwargsItems, MultiModalBatchedField
    """

    def _get_data_parser(self) -> MultiModalDataParser:
        return BiomedRnaDataParser()

    def _get_mm_fields_config(
        self,
        hf_inputs: Mapping[str, object],
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return RNA_FIELDS_CONFIG

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        return []  # RNA has no placeholder tokens

    def apply(
        self, inputs: ProcessorInputs, timing_ctx: TimingContext
    ) -> MultiModalInput:
        with timing_ctx.record("apply_hf_processor"):
            _, passthrough_data = self._get_hf_mm_data(inputs.mm_data_items)

            # unsqueeze(0) adds the batch dimension that from_hf_inputs expects.
            # from_hf_inputs treats dim-0 as the item dimension and splits it
            # into one MultiModalKwargsItem per item — here always 1 per request.
            mm_processed_data = BatchFeature(
                {
                    k: _unwrap_field(passthrough_data, k).unsqueeze(0)
                    for k in ("gene_ids", "expr_values", "attention_mask")
                },
                tensor_type="pt",
            )

        mm_kwargs = MultiModalKwargsItems.from_hf_inputs(
            mm_processed_data, RNA_FIELDS_CONFIG
        )

        return mm_input(
            prompt_token_ids=[
                1
            ],  # dummy token — satisfies vLLM, carries no information
            mm_kwargs=mm_kwargs,
            mm_hashes=inputs.get_mm_hashes(self.info.model_id),
            mm_placeholders={"rna": [PlaceholderRange(offset=0, length=0)]},
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class LlamaForMultiTaskModelNoCheckpoint(LlamaForMultiTaskModel):
    """LlamaForMultiTaskModel variant that skips checkpoint loading on init."""

    def load_checkpoint(self):
        pass


@attn_type("attention_free")
@MULTIMODAL_REGISTRY.register_processor(
    BiomedRnaMultiModalProcessor,
    info=BiomedRnaProcessingInfo,
    dummy_inputs=BiomedRnaDummyInputsBuilder,
)
class BiomedRnaForSequenceEmbedding(nn.Module, IsAttentionFree, SupportsMultiModal):
    """
    BiomedRNA model for single-cell RNA embedding generation.

    Wraps IBM's LlamaForMultiTaskModel (biomed-rna) as a vLLM multimodal embedding model.
    Each request is one cell; the model outputs one embedding vector per cell.

    Input (per batch, via **kwargs from mm_kwargs):
        gene_ids:       Tensor[batch, seq_len] long  OR  list[Tensor[seq_len_i]]
        expr_values:    Tensor[batch, seq_len] float OR  list[Tensor[seq_len_i]]
        attention_mask: Tensor[batch, seq_len] bool  OR  list[Tensor[seq_len_i]]

    Output: Tensor[batch, 1, hidden_size]

    The stacked-tensor path is the common case (all cells from one preprocessing run
    share the same seq_len). The list path handles the rare case of concurrent users
    submitting cells with different seq_len values.
    """

    # Required for IsAttentionFree interface - indicates this is an encoder-only model
    # with no attention mechanism that vLLM needs to manage
    is_attention_free: bool = True

    supports_multimodal_raw_input_only = True
    is_pooling_model = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("rna"):
            return None  # RNA has no prompt placeholder tokens
        raise ValueError(f"Unsupported modality: {modality!r}")

    # def get_data_key(self) -> str:
    #     return "rna"
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        biomed_rna_config: LlamaForMultiTaskConfig = vllm_config.model_config.hf_config

        self.pooling_method = getattr(
            biomed_rna_config, "pooling_method", "first_token"
        )
        self.hidden_size = biomed_rna_config.hidden_size
        self.model = LlamaForMultiTaskModelNoCheckpoint(biomed_rna_config)
        self.model.eval()

        self.pooler = EmbeddingIdentityPooler()
        logger.info("BiomedRNA model initialized.")

    @staticmethod
    def get_kv_cache_spec() -> dict:
        return {}  # encoder-only model — no KV cache

    def get_language_model(self) -> "BiomedRnaForSequenceEmbedding":
        """
        vLLM v1 calls this to unwrap VLM wrappers and check for inner MoE models.
        BiomedRNA is not a VLM wrapper — return self so vLLM finds no MoE and moves on.
        Required for the online path to work, implementing IsAttentionFree is not enough.
        """
        return self

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Return zero embeddings of the correct hidden size for the dummy token.

        The actual RNA data arrives via **kwargs in forward(), not through embeddings.
        Unlike Terratorch which can use (batch, 0), BiomedRNA needs (batch, hidden_size)
        because vLLM V1 still allocates and copies the embedding buffer.
        """
        return torch.zeros(
            input_ids.shape[0],
            self.hidden_size,
            dtype=next(self.parameters()).dtype,
            device=input_ids.device,
        )

    def _pad_variable_length_batch(
        self,
        gene_ids: list[torch.Tensor],
        expr_values: list[torch.Tensor],
        attention_mask: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pad a ragged batch to uniform length for model input.

        Called only when concurrent users submit cells with different seq_len values.
        The common case (single preprocessing run, uniform seq_len) goes through
        the fast path in forward() which uses torch.stack() with no padding.

        Padding values:
            gene_ids:       RNA_PAD_TOKEN_ID (2)
            expr_values:    0.0
            attention_mask: False
        """
        max_len = max(g.shape[0] for g in gene_ids)
        batch = len(gene_ids)
        device = gene_ids[0].device

        gene_ids_out = torch.full(
            (batch, max_len), RNA_PAD_TOKEN_ID, dtype=torch.long, device=device
        )
        expr_out = torch.zeros((batch, max_len), dtype=torch.float, device=device)
        mask_out = torch.zeros((batch, max_len), dtype=torch.bool, device=device)

        for i, (g, e, m) in enumerate(zip(gene_ids, expr_values, attention_mask)):
            L = g.shape[0]
            gene_ids_out[i, :L] = g
            expr_out[i, :L] = e
            mask_out[i, :L] = m

        return gene_ids_out, expr_out, mask_out

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attn_metadata=None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate cell embeddings from RNA expression data.

        Receives gene_ids, expr_values, attention_mask from mm_kwargs (batched by vLLM).
        Handles two cases:
          - Tensor[batch, seq_len]: all cells same length → fast path, no padding
          - list[Tensor[seq_len_i]]: variable lengths → pad to max length in batch

        BMFM input format: torch.stack([gene_ids.float(), expr_values], dim=1)
        → Tensor[batch, 2, seq_len]. gene_ids.float() is required by BMFM, not a bug.
        """
        gene_ids = kwargs["gene_ids"]
        expr_values = kwargs["expr_values"]
        attention_mask = kwargs["attention_mask"]

        if isinstance(gene_ids, list):
            # Rare: concurrent users with different seq_len from different preprocessing runs
            gene_ids, expr_values, attention_mask = self._pad_variable_length_batch(
                gene_ids, expr_values, attention_mask
            )
        # else: Tensor[batch, seq_len] — common case, no work needed

        # gene_ids.float() is required by BMFM input format
        bmfm_input = torch.stack(
            [gene_ids.float(), expr_values], dim=1
        )  # [batch, 2, seq_len]

        output = self.model(
            input_ids=bmfm_input,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        embeddings = get_embeddings_from_outputs(
            output,
            attention_mask,
            pooling_method=self.pooling_method,
        )

        return embeddings.unsqueeze(1)  # [batch, 1, hidden_size]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load weights from a safetensors checkpoint into the biomed-rna model."""
        param_targets = dict(self.model.named_parameters())
        buffer_targets = dict(self.model.named_buffers())
        loaded = set()
        missing = set()  # checkpoint keys not found in model

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

        all_targets = param_targets.keys() | buffer_targets.keys()
        if unloaded := all_targets - loaded:
            logger.warning(
                "Weights not loaded from checkpoint: %s", list(unloaded)[:10]
            )
        if missing:
            logger.warning(
                "Checkpoint has extra weights not in model: %s", list(missing)[:10]
            )
