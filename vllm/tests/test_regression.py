#!/usr/bin/env python3
"""Regression test: vLLM vs direct bmfm-targets inference from h5ad."""

import gc
from pathlib import Path

import anndata
import numpy as np
import torch
from bmfm_targets.inference import inference

H5AD_PATH = (
    Path(__file__).resolve().parent.parent / "examples" / "resources" / "zheng68k.h5ad"
)
HF_MODEL_REPO = "ibm-research/biomed.rna.llama.47m.wced.multitask.v1"
LIMIT_SAMPLES = 20


def get_embeddings_direct(
    h5ad_path: Path,
    limit_samples: int = LIMIT_SAMPLES,
    max_length: int = 1024,
    limit_genes: str = "protein_coding",
):
    """Run direct bmfm-targets inference."""
    if not h5ad_path.exists():
        raise FileNotFoundError(f"H5AD file not found: {h5ad_path}")

    adata = anndata.read_h5ad(h5ad_path)
    adata = adata[:limit_samples]

    adata = inference(
        adata,
        checkpoint=HF_MODEL_REPO,
        embedding_key="X_bmfm",
        batch_size=limit_samples,
        max_length=max_length,
        limit_genes=limit_genes,
        device="cpu",
        copy=False,
        log_normalize_transform=True,
        pooling_method="first_token",
    )

    embeddings = adata.obsm["X_bmfm"].copy()
    cell_names = adata.obs_names.astype(str).to_numpy()

    del adata
    gc.collect()

    return embeddings, cell_names


def get_embeddings_vllm(
    h5ad_path: Path,
    limit_samples: int = LIMIT_SAMPLES,
    max_length: int = 1024,
    limit_genes: str = "protein_coding",
):
    """Run vLLM inference."""
    if not h5ad_path.exists():
        raise FileNotFoundError(f"H5AD file not found: {h5ad_path}")

    adata = anndata.read_h5ad(h5ad_path)
    adata = adata[:limit_samples]

    # Use utility functions for simplified preprocessing
    from vllm_biomed_rna_plugin import get_vllm_biomed_rna_model
    from vllm_biomed_rna_plugin.preprocess import preprocess_anndata
    from vllm_biomed_rna_plugin.utils import load_tokenizer as plugin_load_tokenizer

    tokenizer = plugin_load_tokenizer()

    # Preprocess using consolidated function
    inputs = preprocess_anndata(
        adata,
        tokenizer,
        max_length=max_length,
        limit_genes=limit_genes,
        log_normalize_transform=True,
        batch_size=limit_samples,
    )

    llm = get_vllm_biomed_rna_model(
        gpu_memory_utilization=0.01,
        num_gpu_blocks_override=1,
    )

    outputs = llm.embed(inputs)
    embeddings = np.array([output.outputs.embedding for output in outputs])
    cell_names = adata.obs_names.astype(str).to_numpy()

    return embeddings, cell_names


def test_vllm_vs_direct_full_flow():
    """Test vLLM flow vs direct bmfm-targets flow from h5ad file."""
    MAX_LENGTH = 1024
    LIMIT_GENES = "protein_coding"

    # Run direct bmfm-targets inference
    direct_embeddings, direct_cell_names = get_embeddings_direct(
        H5AD_PATH, LIMIT_SAMPLES, MAX_LENGTH, LIMIT_GENES
    )

    torch.cuda.empty_cache()
    gc.collect()

    # Run vLLM inference
    vllm_embeddings, vllm_cell_names = get_embeddings_vllm(
        H5AD_PATH, LIMIT_SAMPLES, MAX_LENGTH, LIMIT_GENES
    )

    # Verify same cells
    assert len(vllm_cell_names) == len(
        direct_cell_names
    ), f"Cell count mismatch: vLLM={len(vllm_cell_names)}, direct={len(direct_cell_names)}"
    assert np.array_equal(
        vllm_cell_names, direct_cell_names
    ), "Cell names don't match between vLLM and direct"

    # Compare embeddings
    abs_diff = np.abs(vllm_embeddings - direct_embeddings)
    max_abs_diff = np.max(abs_diff)

    assert np.allclose(
        vllm_embeddings, direct_embeddings, rtol=1e-2, atol=0.1
    ), f"vLLM embeddings don't match direct bmfm-targets embeddings (max_abs_diff={max_abs_diff:.6f}"


if __name__ == "__main__":
    test_vllm_vs_direct_full_flow()
