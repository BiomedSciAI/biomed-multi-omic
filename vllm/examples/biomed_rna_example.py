#!/usr/bin/env python3
"""
Generate cell embeddings from h5ad file using BiomedRNA vLLM plugin.

This example demonstrates two approaches:
1. Single batch processing (quick test with few cells)
2. Full file iteration (memory-efficient processing of entire dataset)
"""

from pathlib import Path

import anndata
import numpy as np

from vllm_biomed_rna_plugin import get_vllm_biomed_rna_model
from vllm_biomed_rna_plugin.biomed_rna import (
    BiomedRnaForSequenceEmbedding,  # Register model class
)
from vllm_biomed_rna_plugin.preprocess import iter_h5ad_batches, preprocess_anndata
from vllm_biomed_rna_plugin.utils import DEFAULT_MODEL_PATH, load_tokenizer

# Configuration
H5AD_PATH: Path = Path("examples/resources/zheng68k.h5ad")


def generate_embedding_for_h5ad_snippet(
    h5ad_path: Path = H5AD_PATH,
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

    # Load data and tokenizer
    adata = anndata.read_h5ad(h5ad_path)[:num_samples]
    tokenizer = load_tokenizer()
    print(f"Input: {adata.n_obs} cells × {adata.n_vars} genes")

    # Preprocess (fields loaded automatically from checkpoint)
    inputs = preprocess_anndata(
        adata,
        tokenizer,
        max_length=max_length,
    )
    print(
        f"Preprocessed: {len(inputs)} cells, "
        f"seq_len={len(inputs[0]['multi_modal_data']['rna']['gene_ids'])}"
    )

    # Generate embeddings
    llm = get_vllm_biomed_rna_model(
        model_path=DEFAULT_MODEL_PATH,
    )
    outputs = llm.embed(inputs)
    embeddings = np.array([output.outputs.embedding for output in outputs])
    print(f"Output embedding shape: {embeddings.shape}")
    return embeddings


def generate_embeddings_for_full_h5ad(
    h5ad_path: Path = H5AD_PATH,
    batch_size: int = 32,
    max_length: int = 1024,
    limit_cells: int | None = None,
) -> np.ndarray:
    """
    Generate embeddings for entire h5ad file using batch iteration.

    Memory-efficient: Processes file in chunks without loading everything
    into memory at once. Uses DataModule preprocessing for each batch.

    Args:
    ----
        h5ad_path: Path to h5ad file
        batch_size: Number of cells per batch (default: 32)
        max_length: Maximum sequence length for preprocessing
        limit_cells: Optional limit on total cells to process (for testing)

    Returns:
    -------
        np.ndarray: Cell embeddings with shape [n_cells, hidden_size]
    """
    print(f"\n{'='*80}")
    print(f"Example 2: Full File Iteration (batch_size={batch_size})")
    print(f"{'='*80}")

    tokenizer = load_tokenizer()
    llm = get_vllm_biomed_rna_model(
        model_path=DEFAULT_MODEL_PATH,
    )

    # Get total cell count
    adata_info = anndata.read_h5ad(h5ad_path, backed="r")
    total_cells = (
        adata_info.n_obs if limit_cells is None else min(limit_cells, adata_info.n_obs)
    )
    print(f"Processing {total_cells} cells from {h5ad_path.name}")

    # Process in batches
    all_embeddings = []
    cells_processed = 0

    for batch in iter_h5ad_batches(
        str(h5ad_path),
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
    ):
        # Generate embeddings for this batch
        outputs = llm.embed(batch)
        batch_embeddings: list[list[float]] = [
            output.outputs.embedding for output in outputs
        ]
        all_embeddings.extend(batch_embeddings)

        cells_processed += len(batch)
        print(f"  Processed {cells_processed}/{total_cells} cells...")

        # Stop if we've reached the limit
        if limit_cells and cells_processed >= limit_cells:
            break

    # Convert to numpy array
    embeddings = np.array(all_embeddings)
    print("\nCompleted!")
    print(f"Output embedding shape: {embeddings.shape}")

    return embeddings


if __name__ == "__main__":
    # Example 1: Quick test with 10 cells
    embeddings_snippet = generate_embedding_for_h5ad_snippet(num_samples=10)

    # Example 2: Process more cells using batch iteration
    embeddings_full = generate_embeddings_for_full_h5ad(
        batch_size=32,
    )
