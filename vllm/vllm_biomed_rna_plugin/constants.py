"""
Shared constants for BiomedRNA vLLM plugin.

These constants are used across multiple modules to ensure consistency.
"""

from vllm.multimodal.inputs import MultiModalFieldConfig

# BMFM-compatible pad value for gene_ids (request-level padding only)
RNA_PAD_TOKEN_ID = 2

# RNA field configuration schema - used by both RnaProcessorItems and BiomedRnaMultiModalProcessor
RNA_FIELDS_CONFIG = {
    "gene_ids": MultiModalFieldConfig.batched("rna"),
    "expr_values": MultiModalFieldConfig.batched("rna"),
    "attention_mask": MultiModalFieldConfig.batched("rna"),
}
