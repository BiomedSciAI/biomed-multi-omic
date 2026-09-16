#!/usr/bin/env python3
"""
Compute baseline integration embeddings and save to AnnData.

Usage:
    python run_baselines.py data.h5ad --batch-key batch
    python run_baselines.py data.h5ad --batch-key donor --methods Harmony Scanorama
    python run_baselines.py data.h5ad --batch-key donor --methods scPRINT
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

AVAILABLE_METHODS = ["Unintegrated", "Harmony", "Scanorama", "LIGER", "scPRINT"]

# HuggingFace checkpoint used for the scPRINT baseline (jkobject/scPRINT repo).
SCPRINT_DEFAULT_CHECKPOINT = "medium-v1.5.ckpt"


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


def compute_scprint(
    adata: AnnData,
    batch_key: str,
    ckpt_path: str = SCPRINT_DEFAULT_CHECKPOINT,
    max_len: int = 4000,
    batch_size: int = 32,
    num_workers: int = 8,
) -> np.ndarray | None:
    """
    Embed cells with the pretrained scPRINT model (frozen weights, no finetuning).

    Embeddings are extracted as the mean of scPRINT's class tokens (`scprint_emb`),
    exactly as produced by the published checkpoint.  No classification heads or
    visualisation steps are run — only the forward pass needed for embeddings.

    Requires the optional ``scprint`` package (``pip install scprint``).
    The checkpoint ``medium-v1.5.ckpt`` can be downloaded from HuggingFace with::

        hf download jkobject/scPRINT medium-v1.5.ckpt --local-dir .

    Prerequisites for the input AnnData:
    - ``adata.X`` must contain raw integer counts (or ``layers['counts']`` if X
      is already normalised — the counts layer is used in that case).
    - ``var_names`` must be HUGO gene symbols (scPRINT uses ``is_symbol=True``).
    - ``adata.obs['organism_ontology_term_id']`` must be set
      (e.g. ``'NCBITaxon:9606'`` for human); added automatically if missing.

    Args:
    ----
        adata: Input AnnData (not modified in place).
        batch_key: Batch column — used only for the shared method signature;
            scPRINT embeddings are batch-agnostic.
        ckpt_path: Path to a scPRINT ``.ckpt`` file.
        max_len: Number of genes per cell passed to the transformer.
        batch_size: Inference batch size.
        num_workers: DataLoader workers.

    Returns:
    -------
        Float32 array of shape ``(n_cells, embedding_dim)`` or ``None`` on failure.
    """
    try:
        import torch
        from scprint import scPrint
        from scprint.tasks import Embedder
    except ImportError:
        logger.warning(
            "scPRINT not installed. Install with: pip install scprint\n"
            "  and download the checkpoint with:\n"
            "  hf download jkobject/scPRINT medium-v1.5.ckpt --local-dir ."
        )
        return None

    try:
        adata = adata.copy()

        # scPRINT Preprocessor expects raw counts in X.
        if "counts" in adata.layers and not guess_if_raw(
            adata.X.data if hasattr(adata.X, "data") else adata.X
        ):
            logger.info("scPRINT: using layers['counts'] as raw count source.")
            adata.X = adata.layers["counts"]

        if "organism_ontology_term_id" not in adata.obs.columns:
            logger.info(
                "scPRINT: 'organism_ontology_term_id' missing — defaulting to human (NCBITaxon:9606)."
            )
            adata.obs["organism_ontology_term_id"] = "NCBITaxon:9606"

        model = scPrint.load_from_checkpoint(ckpt_path, precpt_gene_emb=None)
        model.eval()

        embedder = Embedder(
            how="most var",  # deterministic HVG selection (Seurat v3)
            max_len=max_len,
            batch_size=batch_size,
            num_workers=num_workers,
            doclass=False,  # no classifier heads — pure embedding pass
            doplot=False,  # skip PCA/UMAP/Leiden
            output_expression="none",
            dtype=torch.float32,  # float32 avoids MPS/fp16 numerical issues
        )

        adata, _ = embedder(model, adata, cache=False)
        emb = adata.obsm["scprint_emb"]
        logger.info(f"scPRINT: produced embeddings of shape {emb.shape}.")
        return np.asarray(emb, dtype=np.float32)

    except Exception as e:
        logger.warning(f"scPRINT failed: {e}")
        return None


METHOD_REGISTRY: dict[str, callable] = {
    "Unintegrated": compute_unintegrated,
    "Harmony": compute_harmony,
    "Scanorama": compute_scanorama,
    "LIGER": compute_liger,
    "scPRINT": compute_scprint,
}


def run_baselines(
    adata_path: Path,
    batch_key: str,
    methods: list[str] | None = None,
    output_path: Path | None = None,  # Reserved for future use
    scprint_ckpt: str = SCPRINT_DEFAULT_CHECKPOINT,
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
        extra = {"ckpt_path": scprint_ckpt} if method == "scPRINT" else {}
        result = fn(adata, batch_key, **extra)
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
    parser.add_argument(
        "--scprint-ckpt",
        default=SCPRINT_DEFAULT_CHECKPOINT,
        help="Path to scPRINT .ckpt file (only used when scPRINT is in --methods).",
    )
    args = parser.parse_args()

    run_baselines(
        adata_path=args.input,
        batch_key=args.batch_key,
        methods=args.methods,
        scprint_ckpt=args.scprint_ckpt,
    )


if __name__ == "__main__":
    main()
