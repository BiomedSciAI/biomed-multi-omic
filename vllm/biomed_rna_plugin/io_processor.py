"""
IO Processor Plugin for BiomedRNA vLLM integration.

This plugin handles serialization/deserialization between HTTP JSON requests
and vLLM's internal multi-modal data format for single-cell RNA data.
"""

from collections.abc import Sequence
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.inputs import PromptType
from vllm.outputs import PoolingOutput, PoolingRequestOutput
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.renderers import BaseRenderer


class RnaPrompt(dict[str, Any]):
    """Type for RNA prompt data."""

    pass


class RnaOutput(dict[str, Any]):
    """Type for RNA output data."""

    pass


class BiomedRnaIOProcessor(IOProcessor[RnaPrompt, RnaOutput]):
    """
    IO processor for BiomedRNA multi-modal data.

    Converts between HTTP JSON format and vLLM's internal format:
    - Input: JSON with gene_ids, expr_values, attention_mask
    - Output: JSON with embedding vector
    """

    def __init__(self, vllm_config: VllmConfig, renderer: BaseRenderer):
        """
        Initialize the IO processor.

        Args:
        ----
            vllm_config: vLLM configuration object
            renderer: Renderer object for formatting output
        """
        super().__init__(vllm_config, renderer)

    def merge_pooling_params(self, params=None):
        """
        Override to set task to 'embed' instead of 'plugin'.

        The pooling endpoint only supports 'embed' task.
        """
        from vllm import PoolingParams

        return params or PoolingParams(task="embed")

    def parse_data(self, data: object) -> RnaPrompt:
        """
        Parse incoming request data.

        Args:
        ----
            data: Raw request data (dict with RNA data)

        Returns:
        -------
            RnaPrompt containing the parsed data
        """
        if isinstance(data, dict):
            return RnaPrompt(data)
        raise ValueError("Prompt data should be a dictionary with RNA data")

    def pre_process(
        self,
        prompt: RnaPrompt,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        """
        Convert incoming RNA data to vLLM prompt format.

        Args:
        ----
            prompt: RnaPrompt containing gene_ids, expr_values, attention_mask
            request_id: Optional request ID

        Returns:
        -------
            Dictionary with prompt_token_ids and multi_modal_data
        """
        # Extract RNA data arrays
        gene_ids = prompt.get("gene_ids", [])
        expr_values = prompt.get("expr_values", [])
        attention_mask = prompt.get("attention_mask", [])

        # Validate input
        if not gene_ids or not expr_values:
            raise ValueError(
                "Missing required RNA data. Expected 'gene_ids' and 'expr_values'"
            )

        # Convert to 1D tensors (no batch dimension)
        # The multi-modal processor will add the batch dimension
        gene_ids_tensor = torch.tensor(gene_ids, dtype=torch.long)
        expr_values_tensor = torch.tensor(expr_values, dtype=torch.float32)

        # Create attention mask if not provided
        if not attention_mask:
            attention_mask = [True] * len(gene_ids)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.bool)

        # Validate shapes - all should be 1D with same length
        if gene_ids_tensor.shape != expr_values_tensor.shape:
            raise ValueError(
                f"Shape mismatch: gene_ids {gene_ids_tensor.shape} != "
                f"expr_values {expr_values_tensor.shape}"
            )

        if gene_ids_tensor.shape != attention_mask_tensor.shape:
            raise ValueError(
                f"Shape mismatch: gene_ids {gene_ids_tensor.shape} != "
                f"attention_mask {attention_mask_tensor.shape}"
            )

        if gene_ids_tensor.ndim != 1:
            raise ValueError(f"Expected 1D tensors, got {gene_ids_tensor.ndim}D")

        # Return vLLM-compatible format
        return {
            "prompt_token_ids": [1],
            "multi_modal_data": {
                "rna": {
                    "gene_ids": gene_ids_tensor,
                    "expr_values": expr_values_tensor,
                    "attention_mask": attention_mask_tensor,
                }
            },
        }

    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: str | None = None,
        **kwargs,
    ) -> RnaOutput:
        """
        Convert vLLM embedding output to HTTP JSON response.

        Args:
        ----
            model_output: Sequence of PoolingRequestOutput objects
            request_id: Optional request ID

        Returns:
        -------
            RnaOutput with embedding data
        """
        if not model_output:
            raise ValueError("No model output available")

        output: PoolingRequestOutput[PoolingOutput] = model_output[0]

        # Extract embedding from PoolingRequestOutput
        # PoolingRequestOutput.outputs is a PoolingOutput with a 'data' attribute
        if hasattr(output, "outputs") and hasattr(output.outputs, "data"):
            embedding = output.outputs.data
        else:
            raise ValueError(f"Cannot find embedding data in output: {type(output)}")

        # Convert PoolingRequestOutput tensor to list for JSON serialization
        # By this point, the tensor is already on CPU (vLLM transfers it)
        embedding_list = embedding.tolist()

        return RnaOutput(
            {
                "embedding": embedding_list,
                "embedding_dim": len(embedding_list),
            }
        )


# Entry point for vLLM plugin system
def register_biomed_rna_plugin():
    """
    Factory function for vLLM plugin system.

    Returns the qualified name of the IO processor class as a string.
    vLLM will then use resolve_obj_by_qualname to import and instantiate it.
    """
    return "biomed_rna_plugin.io_processor.BiomedRnaIOProcessor"
