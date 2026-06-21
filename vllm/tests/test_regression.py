#!/usr/bin/env python3
"""Regression test: vLLM vs direct bmfm-targets inference from h5ad."""

import gc
from pathlib import Path

import anndata
import numpy as np
import pytest
import torch
from bmfm_targets.inference import inference

H5AD_PATH = (
    Path(__file__).resolve().parent.parent / "examples" / "resources" / "zheng68k.h5ad"
)

LIMIT_SAMPLES = 20


# Test model configurations
TEST_MODEL_CONFIGS = [
    (
        "ibm-research/biomed.rna.llama.47m.wced.multitask.v1.vllm",
        "ibm-research/biomed.rna.llama.47m.wced.multitask.v1",
    ),
    (
        "ibm-research/biomed.rna.llama.32m.mlm.multitask.v1.vllm",
        "ibm-research/biomed.rna.llama.32m.mlm.multitask.v1",
    ),
]


def get_embeddings_direct(
    h5ad_path: Path,
    origin_model_repo: str,
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
        checkpoint=origin_model_repo,
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
    vllm_model,
    vllm_model_repo: str,
    limit_samples: int = LIMIT_SAMPLES,
    max_length: int = 1024,
    limit_genes: str = "protein_coding",
):
    """Run vLLM inference using shared model fixture."""
    if not h5ad_path.exists():
        raise FileNotFoundError(f"H5AD file not found: {h5ad_path}")

    adata = anndata.read_h5ad(h5ad_path)
    adata = adata[:limit_samples]

    # Use utility functions for simplified preprocessing
    from vllm_biomed_rna_plugin.preprocess import preprocess_anndata
    from vllm_biomed_rna_plugin.utils import load_tokenizer as plugin_load_tokenizer

    # Each model uses its own tokenizer for proper regression testing
    tokenizer = plugin_load_tokenizer(vllm_model_repo)

    # Preprocess using consolidated function with model-specific fields
    inputs = preprocess_anndata(
        adata,
        tokenizer,
        max_length=max_length,
        limit_genes=limit_genes,
        log_normalize_transform=True,
        batch_size=limit_samples,
        model_repo=vllm_model_repo,
    )

    outputs = vllm_model.embed(inputs)
    embeddings = np.array([output.outputs.embedding for output in outputs])
    cell_names = adata.obs_names.astype(str).to_numpy()

    return embeddings, cell_names


def _model_id(config_tuple):
    """Extract model name from config tuple for test IDs."""
    return (
        config_tuple[0].split("/")[-1]
        if isinstance(config_tuple, str)
        else config_tuple
    )


@pytest.mark.parametrize(
    ("vllm_model_repo", "origin_model_repo"), TEST_MODEL_CONFIGS, ids=_model_id
)
def test_vllm_vs_direct_full_flow(vllm_model_repo, origin_model_repo):
    """Test vLLM flow vs direct bmfm-targets flow from h5ad file."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from vllm_biomed_rna_plugin import get_vllm_biomed_rna_model

    MAX_LENGTH = 1024
    LIMIT_GENES = "protein_coding"

    # Run direct bmfm-targets inference first
    direct_embeddings, direct_cell_names = get_embeddings_direct(
        H5AD_PATH, origin_model_repo, LIMIT_SAMPLES, MAX_LENGTH, LIMIT_GENES
    )

    torch.cuda.empty_cache()
    gc.collect()

    # Load vLLM model for this specific test
    vllm_model = get_vllm_biomed_rna_model(
        model_repo=vllm_model_repo,
        gpu_memory_utilization=0.01,  # Minimal memory like conftest.py
        disable_log_stats=True,
        dtype="float32",
        max_num_seqs=8,
    )

    try:
        # Run vLLM inference
        vllm_embeddings, vllm_cell_names = get_embeddings_vllm(
            H5AD_PATH,
            vllm_model,
            vllm_model_repo,
            LIMIT_SAMPLES,
            MAX_LENGTH,
            LIMIT_GENES,
        )

        # GPU memory cleanup after vLLM inference
        torch.cuda.empty_cache()
        gc.collect()

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
        ), f"vLLM embeddings don't match direct bmfm-targets embeddings (max_abs_diff={max_abs_diff:.6f})"
    finally:
        # Cleanup
        del vllm_model
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    # For standalone execution, test both models

    for vllm_model_repo, origin_model_repo in TEST_MODEL_CONFIGS:
        print(f"\nTesting model: {vllm_model_repo}")
        test_vllm_vs_direct_full_flow(vllm_model_repo, origin_model_repo)
        print(f"✓ Model {vllm_model_repo} passed regression test")
