"""
Preprocessing module for BiomedRNA vLLM plugin.

Converts gene expression data into vLLM-compatible format using multi_modal_data.
Uses bmfm-targets tokenizer and data module.
"""


import torch

from bmfm_targets.training.data_module import DataModule
from vllm_biomed_rna_plugin.utils import WCED_MULTITASK_MODEL, get_fields


def create_rna_vllm_input(
    gene_ids: torch.Tensor,
    expr_values: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict:
    """
    Create a single vLLM RNA input dict.

    Args:
    ----
        gene_ids: Gene IDs tensor [seq_len]
        expr_values: Expression values tensor [seq_len]
        attention_mask: Attention mask tensor [seq_len]

    Returns:
    -------
        dict: vLLM-compatible input dict with RNA multi-modal data
    """
    seq_len = gene_ids.shape[0]
    return {
        "prompt_token_ids": [0] * seq_len,  # Match sequence length
        "multi_modal_data": {
            "rna": {
                "gene_ids": gene_ids,
                "expr_values": expr_values,
                "attention_mask": attention_mask,
            }
        },
    }


def _convert_datamodule_batch_to_vllm_format(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> list[dict]:
    """
    Convert DataModule batch output to vLLM multi-modal format.

    DataModule produces Tensor[batch, 2, seq_len] internally.
    vLLM expects one request dict per cell.
    This function bridges the two by unpacking the batch.
    The processor in biomed_rna.py will repack into dense tensors.

    Args:
    ----
        input_ids: Tensor of shape [batch_size, 2, seq_len] where:
            - input_ids[:, 0, :] are gene IDs
            - input_ids[:, 1, :] are expression values
        attention_mask: Tensor of shape [batch_size, seq_len]

    Returns:
    -------
        list[dict]: List of vLLM-compatible input dicts
    """
    inputs = []
    for i in range(input_ids.shape[0]):
        gene_ids = input_ids[i, 0, :]  # [seq_len]
        expr_values = input_ids[i, 1, :]  # [seq_len]
        attn_mask = attention_mask[i, :]  # [seq_len]

        inputs.append(create_rna_vllm_input(gene_ids, expr_values, attn_mask))

    return inputs


def preprocess_anndata(
    adata,
    tokenizer,
    max_length: int = 1024,
    limit_genes: str = "protein_coding",
    log_normalize_transform: bool = True,
    batch_size: int | None = None,
    model_repo: str = WCED_MULTITASK_MODEL,
) -> list[dict]:
    """
    Preprocess h5ad data using bmfm-targets DataModule and create a list of RNA multi modal objects.

    Applies the same preprocessing as bmfm-targets inference:
    - Log normalization (if enabled)
    - Gene filtering (e.g., protein_coding only)
    - Sequence length limiting (max_length)
    - Attention mask generation

    This is the RECOMMENDED preprocessing method for production use.

    Args:
    ----
        adata: AnnData object with gene expression data
        tokenizer: MultiFieldTokenizer from bmfm-targets
        max_length: Maximum sequence length (default: 1024)
        limit_genes: Gene filtering strategy - "protein_coding" or None (default: "protein_coding")
        log_normalize_transform: Apply log normalization (default: True)
        batch_size: Batch size for DataModule (default: all cells)
        model_repo: HuggingFace model repository ID to get fields from (default: WCED_MULTITASK_MODEL)

    Returns:
    -------
        list[dict]: List of preprocessed cells in vLLM format, each containing:
            {
                "prompt_token_ids": [0] * seq_len,
                "multi_modal_data": {
                    "rna": {
                        "gene_ids": tensor [seq_len] int32,
                        "expr_values": tensor [seq_len] float32,
                        "attention_mask": tensor [seq_len] int32
                    }
                }
            }
    """
    if batch_size is None:
        batch_size = adata.n_obs

    fields = get_fields(model_repo)

    # Create DataModule with same settings as bmfm-targets inference
    dataset_kwargs = {"processed_data_source": adata}
    data_module = DataModule(
        tokenizer=tokenizer,
        fields=fields,
        label_columns=[],
        batch_size=batch_size,
        max_length=max_length,
        dataset_kwargs=dataset_kwargs,
        limit_genes=limit_genes,
        mlm=False,
        data_dir=None,
        transform_datasets=False,
        log_normalize_transform=log_normalize_transform,
    )
    data_module.setup("predict")

    # Get preprocessed batch
    batch = next(iter(data_module.predict_dataloader()))
    input_ids = batch["input_ids"]  # [batch_size, 2, seq_len]
    attention_mask = batch["attention_mask"]  # [batch_size, seq_len]

    # Convert to vLLM format
    return _convert_datamodule_batch_to_vllm_format(input_ids, attention_mask)
