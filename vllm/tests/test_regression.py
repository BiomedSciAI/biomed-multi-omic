#!/usr/bin/env python3
"""
Regression test: vLLM vs direct bmfm-targets inference from h5ad.

Runtime budget:
  - Direct bmfm-targets inference (CPU) is by far the slowest step (~60-90s per
    model load + forward pass).  To keep the suite under ~2 min, each model is
    run through direct inference ONCE and all pooling methods are checked in that
    single test rather than once per parametrize case.
  - vLLM model fixtures are session-scoped so the two models are each loaded once.
"""

import gc
from pathlib import Path

import anndata
import numpy as np
import torch

from bmfm_targets.inference import inference

H5AD_PATH = (
    Path(__file__).resolve().parent.parent / "examples" / "resources" / "zheng68k.h5ad"
)

LIMIT_SAMPLES = 20
POOLING_METHODS = ["first_token", "mean_pooling"]

WCED_VLLM_REPO = "ibm-research/biomed.rna.llama.47m.wced.multitask.v1.vllm"
WCED_ORIGIN_REPO = "ibm-research/biomed.rna.llama.47m.wced.multitask.v1"
MLM_VLLM_REPO = "ibm-research/biomed.rna.llama.32m.mlm.multitask.v1.vllm"
MLM_ORIGIN_REPO = "ibm-research/biomed.rna.llama.32m.mlm.multitask.v1"


def get_embeddings_direct(
    h5ad_path: Path,
    origin_model_repo: str,
    pooling_method: str,
    limit_samples: int = LIMIT_SAMPLES,
    max_length: int = 1024,
    limit_genes: str = "protein_coding",
):
    """Run direct bmfm-targets inference for one pooling method."""
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
        pooling_method=pooling_method,
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
    pooling_method: str,
    limit_samples: int = LIMIT_SAMPLES,
    max_length: int = 1024,
    limit_genes: str = "protein_coding",
):
    """Run vLLM inference using shared model fixture."""
    if not h5ad_path.exists():
        raise FileNotFoundError(f"H5AD file not found: {h5ad_path}")

    adata = anndata.read_h5ad(h5ad_path)
    adata = adata[:limit_samples]

    from vllm_biomed_rna_plugin.preprocess import preprocess_anndata
    from vllm_biomed_rna_plugin.utils import load_tokenizer as plugin_load_tokenizer

    tokenizer = plugin_load_tokenizer(vllm_model_repo)

    inputs = preprocess_anndata(
        adata,
        tokenizer,
        max_length=max_length,
        limit_genes=limit_genes,
        log_normalize_transform=True,
        batch_size=limit_samples,
        model_repo=vllm_model_repo,
        pooling_method=pooling_method,
    )

    outputs = vllm_model.embed(inputs)
    embeddings = np.array([output.outputs.embedding for output in outputs])
    cell_names = adata.obs_names.astype(str).to_numpy()

    return embeddings, cell_names


def _assert_embeddings_match(
    vllm_embeddings,
    direct_embeddings,
    vllm_cell_names,
    direct_cell_names,
    vllm_model_repo,
    pooling_method,
):
    assert np.array_equal(
        vllm_cell_names, direct_cell_names
    ), "Cell names don't match between vLLM and direct"
    abs_diff = np.abs(vllm_embeddings - direct_embeddings)
    max_abs_diff = np.max(abs_diff)
    assert np.allclose(vllm_embeddings, direct_embeddings, rtol=1e-2, atol=0.1), (
        f"model={vllm_model_repo!r} pooling_method={pooling_method!r}: "
        f"vLLM embeddings don't match direct bmfm-targets "
        f"(max_abs_diff={max_abs_diff:.6f})"
    )


def _run_regression(vllm_model, vllm_repo, origin_repo):
    """
    Run the full regression for one model across all pooling methods.

    Direct inference loads the origin model on CPU once per pooling method —
    that is the dominant cost (~60-90s each).  All methods are checked in a
    single test function so pytest counts this as one test item rather than
    one per method, keeping --co output clean while halving the number of
    CPU model loads vs. using @parametrize.
    """
    for pooling_method in POOLING_METHODS:
        direct_embeddings, direct_cell_names = get_embeddings_direct(
            H5AD_PATH,
            origin_repo,
            pooling_method=pooling_method,
        )
        gc.collect()

        vllm_embeddings, vllm_cell_names = get_embeddings_vllm(
            H5AD_PATH,
            vllm_model,
            vllm_repo,
            pooling_method=pooling_method,
        )

        _assert_embeddings_match(
            vllm_embeddings,
            direct_embeddings,
            vllm_cell_names,
            direct_cell_names,
            vllm_repo,
            pooling_method,
        )


def test_wced_vs_direct(vllm_model):
    """WCED 47M: vLLM output matches bmfm-targets direct inference for all pooling methods."""
    _run_regression(vllm_model, WCED_VLLM_REPO, WCED_ORIGIN_REPO)


def test_mlm_vs_direct(vllm_model_mlm):
    """MLM 32M: vLLM output matches bmfm-targets direct inference for all pooling methods."""
    _run_regression(vllm_model_mlm, MLM_VLLM_REPO, MLM_ORIGIN_REPO)


if __name__ == "__main__":
    from vllm_biomed_rna_plugin import get_vllm_biomed_rna_model

    for vllm_repo, origin_repo, fixture_name in [
        (WCED_VLLM_REPO, WCED_ORIGIN_REPO, "WCED"),
        (MLM_VLLM_REPO, MLM_ORIGIN_REPO, "MLM"),
    ]:
        llm = get_vllm_biomed_rna_model(model_repo=vllm_repo, disable_log_stats=True)
        for method in POOLING_METHODS:
            print(f"\nTesting {fixture_name} pooling={method}")
            direct_emb, direct_names = get_embeddings_direct(
                H5AD_PATH, origin_repo, pooling_method=method
            )
            vllm_emb, vllm_names = get_embeddings_vllm(
                H5AD_PATH, llm, vllm_repo, pooling_method=method
            )
            _assert_embeddings_match(
                vllm_emb, direct_emb, vllm_names, direct_names, vllm_repo, method
            )
            print("✓ passed")
        del llm
        torch.cuda.empty_cache()
        gc.collect()
