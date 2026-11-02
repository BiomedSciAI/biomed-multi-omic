#!/usr/bin/env python3
"""
Compute baseline integration embeddings and save to AnnData.

Usage:
    python run_baselines.py data.h5ad --batch-key batch
    python run_baselines.py data.h5ad --batch-key donor --methods Harmony Scanorama
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import scanpy as sc
from anndata import AnnData

from bmfm_targets.datasets.datasets_utils import guess_if_raw

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

AVAILABLE_METHODS = ["Unintegrated", "Harmony", "Scanorama", "LIGER"]


def get_pca(adata: AnnData, batch_key: str | None, n_comps: int = 30) -> np.ndarray:
    """Compute PCA on (optionally normalized) data with HVG selection."""
    adata = adata.copy()
    if guess_if_raw(adata.X.data):
        logger.info("Detected raw counts — applying normalization and log1p.")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        logger.info("Detected log-transformed input — skipping normalization.")

    try:
        sc.pp.highly_variable_genes(
            adata, flavor="cell_ranger", batch_key=batch_key, n_top_genes=2000
        )
    except Exception:
        logger.warning("Batch-aware HVG failed, reverting to batch-insensitive.")
        sc.pp.highly_variable_genes(adata, flavor="cell_ranger", n_top_genes=2000)

    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack", n_comps=n_comps, mask_var="highly_variable")
    return adata.obsm["X_pca"]


def compute_unintegrated(adata: AnnData, batch_key: str) -> np.ndarray | None:
    """Return PCA embeddings as the unintegrated baseline."""
    try:
        return get_pca(adata, batch_key)
    except Exception as e:
        logger.warning(f"Unintegrated (PCA) failed: {e}")
        return None


def compute_harmony(
    adata: AnnData, batch_key: str, n_pcs: int = 50
) -> np.ndarray | None:
    """Compute Harmony integration."""
    try:
        import harmonypy

        if "X_pca" not in adata.obsm:
            adata.obsm["X_pca"] = get_pca(adata, batch_key)
        X_pca = adata.obsm["X_pca"]
        X_pca = X_pca[:, : min(X_pca.shape[1], n_pcs)]
        X_pca = np.ascontiguousarray(X_pca, dtype=np.float32)
        ho = harmonypy.run_harmony(X_pca, adata.obs, batch_key)
        return np.asarray(ho.Z_corr.T, dtype=np.float32)
    except Exception as e:
        logger.warning(f"Harmony failed: {e}")
        return None


def compute_scanorama(adata: AnnData, batch_key: str) -> np.ndarray | None:
    """Compute Scanorama integration."""
    try:
        import scanorama

        batch_cats = adata.obs[batch_key].cat.categories
        adata_list = [adata[adata.obs[batch_key] == b].copy() for b in batch_cats]
        scanorama.integrate_scanpy(adata_list)

        result = np.zeros((adata.n_obs, adata_list[0].obsm["X_scanorama"].shape[1]))
        for i, b in enumerate(batch_cats):
            result[adata.obs[batch_key] == b] = adata_list[i].obsm["X_scanorama"]
        return result.astype(np.float32)
    except Exception as e:
        logger.warning(f"Scanorama failed: {e}")
        return None


def compute_liger(adata: AnnData, batch_key: str) -> np.ndarray | None:
    """Compute LIGER integration."""
    try:
        import pyliger

        k = min(adata.obs[batch_key].value_counts().min() - 1, 10)
        if k < 1:
            logger.warning("LIGER skipped: insufficient samples per batch (k < 1).")
            return None

        batch_cats = adata.obs[batch_key].cat.categories
        bdata = adata.copy()
        adata_list = [bdata[bdata.obs[batch_key] == b].copy() for b in batch_cats]
        for i, ad in enumerate(adata_list):
            ad.uns["sample_name"] = batch_cats[i]
            ad.uns["var_gene_idx"] = np.arange(bdata.n_vars)

        liger_data = pyliger.create_liger(
            adata_list, remove_missing=False, make_sparse=False
        )
        liger_data.var_genes = bdata.var_names
        pyliger.normalize(liger_data)
        pyliger.scale_not_center(liger_data)
        pyliger.optimize_ALS(liger_data, k=k)
        pyliger.quantile_norm(liger_data)

        result = np.zeros(
            (adata.n_obs, liger_data.adata_list[0].obsm["H_norm"].shape[1])
        )
        for i, b in enumerate(batch_cats):
            result[adata.obs[batch_key] == b] = liger_data.adata_list[i].obsm["H_norm"]
        return result.astype(np.float32)
    except Exception as e:
        logger.warning(f"LIGER failed: {e}")
        return None


METHOD_REGISTRY: dict[str, callable] = {
    "Unintegrated": compute_unintegrated,
    "Harmony": compute_harmony,
    "Scanorama": compute_scanorama,
    "LIGER": compute_liger,
}


def run_baselines(
    adata_path: Path,
    batch_key: str,
    methods: list[str] | None = None,
    output_path: Path | None = None,  # Reserved for future use
) -> None:
    """Load AnnData, compute baseline embeddings, and save in place."""
    output_path = output_path or adata_path  # Overwrite by default

    logger.info(f"Loading {adata_path}")
    adata = sc.read_h5ad(adata_path)

    if batch_key not in adata.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in adata.obs")
    adata.obs[batch_key] = adata.obs[batch_key].astype("category")

    methods = methods or AVAILABLE_METHODS
    invalid = set(methods) - set(METHOD_REGISTRY)
    if invalid:
        raise ValueError(
            f"Unknown methods: {invalid}. Available: {list(METHOD_REGISTRY)}"
        )

    for method in methods:
        logger.info(f"Computing {method}...")
        fn = METHOD_REGISTRY[method]
        result = fn(adata, batch_key)
        if result is not None:
            adata.obsm[method] = result
            logger.info(f"  {method}: stored {result.shape} in obsm['{method}']")
        else:
            logger.warning(f"  {method}: skipped (failed or insufficient data)")

    logger.info(f"Saving to {output_path}")
    adata.write_h5ad(output_path)
    logger.info("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute baseline integration embeddings for scRNA-seq data."
    )
    parser.add_argument("input", type=Path, help="Path to input h5ad file")
    parser.add_argument(
        "--batch-key", default="batch", help="Column name for batch in adata.obs"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=AVAILABLE_METHODS,
        default=AVAILABLE_METHODS,
        help=f"Methods to run (default: all). Choices: {AVAILABLE_METHODS}",
    )
    args = parser.parse_args()

    run_baselines(adata_path=args.input, batch_key=args.batch_key, methods=args.methods)


if __name__ == "__main__":
    main()
