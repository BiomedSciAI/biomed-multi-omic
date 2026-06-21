#!/usr/bin/env python3
"""
Generate cell embeddings from h5ad file using BiomedRNA vLLM plugin.

Contains two examples:
1. Single h5ad batch processing
2. Full file iteration (memory-efficient processing of entire dataset)
"""

import logging
from pathlib import Path

import anndata
import numpy as np

from vllm_biomed_rna_plugin import get_vllm_biomed_rna_model
from vllm_biomed_rna_plugin.preprocess import preprocess_anndata
from vllm_biomed_rna_plugin.utils import WCED_MULTITASK_MODEL, load_tokenizer

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
ZHENG_SMALL_H5AD_PATH: Path = (
    Path(__file__).parent / "resources" / "zheng68k.h5ad"
)  # 165 samples


def generate_embedding_for_h5ad_snippet(
    h5ad_path: Path = ZHENG_SMALL_H5AD_PATH,
    num_samples: int = 10,
    max_length: int = 1024,
) -> np.ndarray:
    """
    Generate embeddings for a snippet of cells from an h5ad file.

    This is useful for quick testing or when you only need embeddings
    for a small subset of cells.

    Args:
    ----
        h5ad_path: Path to h5ad file
        num_samples: Number of cells to process
        max_length: Maximum sequence length for preprocessing

    Returns:
    -------
        np.ndarray: Cell embeddings with shape [num_samples, hidden_size]
    """
    print(f"\n{'='*80}")
    print(f"Example 1: Single Batch Processing ({num_samples} cells)")
    print(f"{'='*80}")

    model_repo = WCED_MULTITASK_MODEL
    adata = anndata.read_h5ad(h5ad_path)[:num_samples]
    tokenizer = load_tokenizer(model_repo)
    print(f"Input: {adata.n_obs} cells × {adata.n_vars} genes")

    # Preprocess
    inputs = preprocess_anndata(
        adata,
        tokenizer,
        max_length=max_length,
    )
    print(
        f"Preprocessed: {len(inputs)} cells, "
        f"seq_len={len(inputs[0]['multi_modal_data']['rna']['gene_ids'])}"
    )

    llm = get_vllm_biomed_rna_model(model_repo)
    outputs = llm.embed(inputs)
    embeddings = np.array([output.outputs.embedding for output in outputs])
    print(f"Output embedding shape: {embeddings.shape}")
    return embeddings


def generate_embeddings_for_h5ad(
    h5ad_path: Path = ZHENG_SMALL_H5AD_PATH,
    batch_size: int = 1024,
    max_length: int = 1024,
    limit_cells: int | None = None,
) -> np.ndarray:
    """
    Generate embeddings for entire h5ad file using batch iteration.

    Processes file in chunks to support very large files.

    Args:
    ----
        h5ad_path: Path to h5ad file
        batch_size: Number of cells per batch (default: 32)
        max_length: Maximum gene sequence length for preprocessing
        limit_cells: Optional limit on total cells to process (for testing)

    Returns:
    -------
        np.ndarray: Cell embeddings with shape [n_cells, hidden_size]
    """
    print(f"\n{'='*80}")
    print(f"Example 2: Full File Iteration (batch_size={batch_size})")
    print(f"{'='*80}")

    # Initialize tokenizer and model
    model_repo = WCED_MULTITASK_MODEL

    tokenizer = load_tokenizer(model_repo)
    llm = get_vllm_biomed_rna_model(model_repo)

    # Get total cell count for progress reporting
    adata_info = anndata.read_h5ad(h5ad_path, backed="r")
    total_cells = (
        adata_info.n_obs if limit_cells is None else min(limit_cells, adata_info.n_obs)
    )
    print(f"Processing {total_cells} cells from {h5ad_path.name}")

    # Process in batches using the iteration helper
    all_embeddings = []
    for batch in iter_h5ad_batches(
        h5ad_path,
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        limit_cells=limit_cells,
    ):
        # Generate embeddings for this batch
        outputs = llm.embed(batch)
        batch_embeddings = [output.outputs.embedding for output in outputs]
        all_embeddings.extend(batch_embeddings)

        print(f"  Processed {len(all_embeddings)}/{total_cells} cells...")

    # Convert to numpy array
    embeddings = np.array(all_embeddings)
    print("\nCompleted!")
    print(f"Output embedding shape: {embeddings.shape}")

    return embeddings


def iter_h5ad_batches(
    h5ad_path: str | Path,
    tokenizer,
    batch_size: int = 32,
    max_length: int = 1024,
    limit_genes: str = "protein_coding",
    log_normalize_transform: bool = True,
    limit_cells: int | None = None,
):
    """
    Stream batches from h5ad file using DataModule preprocessing.

    Memory-efficient processing with full bmfm-targets transformations:
    - Log normalization (if enabled)
    - Gene filtering (e.g., protein_coding only)
    - Sequence length limiting (max_length)
    - Attention mask generation

    Uses backed="r" mode to avoid loading entire file into memory.

    Args:
    ----
        h5ad_path: Path to h5ad file
        tokenizer: MultiFieldTokenizer from bmfm-targets
        batch_size: Number of cells per batch (default: 32)
        max_length: Maximum sequence length (default: 1024)
        limit_genes: Gene filtering strategy - "protein_coding" or None (default: "protein_coding")
        log_normalize_transform: Apply log normalization (default: True)
        limit_cells: Optional limit on total cells to process (default: None = all cells)

    Yields:
    ------
        list[dict]: Batch of preprocessed cells in vLLM format

    Example:
    -------
        >>> tokenizer = load_tokenizer()
        >>> llm = get_vllm_biomed_rna_model()
        >>>
        >>> all_embeddings = []
        >>> for batch in iter_h5ad_batches("data.h5ad", tokenizer, batch_size=32):
        >>>     outputs = llm.embed(batch)
        >>>     embeddings = [out.outputs.embedding for out in outputs]
        >>>     all_embeddings.extend(embeddings)
        >>>
        >>> embeddings_array = np.array(all_embeddings)  # [n_cells, hidden_size]
    """
    # backed="r" = read-only mode, doesn't load full matrix into memory
    adata = anndata.read_h5ad(str(h5ad_path), backed="r")
    total_cells = adata.n_obs if limit_cells is None else min(limit_cells, adata.n_obs)

    cells_processed = 0
    for start in range(0, total_cells, batch_size):
        end = min(start + batch_size, total_cells)

        # Load chunk into memory
        chunk_adata = adata[start:end].to_memory()

        # Preprocess using DataModule (applies all transformations)
        batch = preprocess_anndata(
            chunk_adata,
            tokenizer,
            max_length=max_length,
            limit_genes=limit_genes,
            log_normalize_transform=log_normalize_transform,
            batch_size=None,  # Process entire chunk at once
        )

        yield batch

        cells_processed = end
        if limit_cells and cells_processed >= limit_cells:
            break


if __name__ == "__main__":
    # Example 1: Quick test with 10 cells
    embeddings_snippet = generate_embedding_for_h5ad_snippet(num_samples=10)

    # Example 2: Process full h5ad file using batch iteration
    embeddings_full = generate_embeddings_for_h5ad(
        batch_size=32,
    )
