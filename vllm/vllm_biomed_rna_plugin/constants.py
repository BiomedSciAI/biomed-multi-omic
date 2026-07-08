"""
Shared constants for BiomedRNA vLLM plugin.

These constants are used across multiple modules to ensure consistency.
"""

from vllm.multimodal.inputs import MultiModalFieldConfig

# BMFM-compatible pad value for gene_ids (request-level padding only)
RNA_PAD_TOKEN_ID = 2

# Pooling method integer encoding used in the pooling_method_id mm field.
# The per-request pooling method is carried through vLLM's multimodal kwargs
# pipeline as a scalar int tensor.
# Named methods use NEGATIVE codes so they can never collide with bare integer
# token positions (which are always >= 0).  The "not specified" sentinel is -1.
# Positive values are literal token-position ints passed straight through to
# get_embeddings_from_outputs(output, mask, pooling_method=<int>).
POOLING_METHOD_IDS: dict[str, int] = {
    "first_token": -2,
    "mean_pooling": -3,
    "pooling_layer": -4,
}
POOLING_METHOD_DEFAULT_ID = -2  # first_token

# Reverse map
POOLING_METHOD_NAMES: dict[int, str] = {v: k for k, v in POOLING_METHOD_IDS.items()}

# RNA field configuration schema - used by both RnaProcessorItems and BiomedRnaMultiModalProcessor
# pooling_method_id is a scalar [1] tensor per request; because all requests share the
# same shape it always stacks cleanly into Tensor[batch, 1] and arrives in forward(**kwargs).
RNA_FIELDS_CONFIG = {
    "gene_ids": MultiModalFieldConfig.batched("rna"),
    "expr_values": MultiModalFieldConfig.batched("rna"),
    "attention_mask": MultiModalFieldConfig.batched("rna"),
    "pooling_method_id": MultiModalFieldConfig.batched("rna"),
}
