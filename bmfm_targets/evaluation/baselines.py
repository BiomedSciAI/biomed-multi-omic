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
import os
import tempfile
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
SCPRINT_HUMAN = "NCBITaxon:9606"
SCPRINT_MOUSE = "NCBITaxon:10090"


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


def _counts_source(adata: AnnData) -> str | None:
    """Return the name of a raw-count source in ``adata`` ("X", a layer, or None)."""
    candidates = [("X", adata.X), *((k, adata.layers[k]) for k in adata.layers)]
    for name, matrix in candidates:
        sample = matrix[: min(500, matrix.shape[0])]
        values = np.asarray(
            sample.toarray() if hasattr(sample, "toarray") else sample
        ).ravel()
        if values.size == 0:
            continue
        if values.min() >= 0 and np.allclose(values, np.round(values)):
            return name
    return None


def compute_scprint(
    adata: AnnData,
    batch_key: str,
    ckpt_path: str = SCPRINT_DEFAULT_CHECKPOINT,
    organism: str = SCPRINT_HUMAN,
    max_len: int = 4000,
    batch_size: int = 32,
    num_workers: int = 8,
    how: str = "random expr",
    pred_embedding: tuple[str, ...] = ("cell_type_ontology_term_id",),
    device: str | None = None,
) -> np.ndarray | None:
    """
    Embed cells with the pretrained scPRINT model (frozen weights, no finetuning).

    Mirrors the reference configuration in scPRINT's own integration benchmark
    (``scprint.tasks.cell_emb.default_benchmark``): the AnnData is run through
    ``scdataloader.Preprocessor`` to map gene identifiers onto the checkpoint's
    Ensembl-keyed vocabulary, then ``Embedder`` produces the cell embedding from
    the ``cell_type_ontology_term_id`` class token. No classification heads or
    plotting are run.

    Requires the optional ``scprint`` package, which pins ``torch==2.2.0`` and
    ``lamindb==0.76.12`` and therefore needs its own environment::

        pip install scprint 'lamindb[bionty]'
        lamin init --storage <dir> --schema bionty
        hf download jkobject/scPRINT medium-v1.5.ckpt --local-dir .

    Input requirements, all of which are checked rather than assumed:
    - raw integer counts in ``X`` or in a layer (log-normalized input is rejected,
      because scPRINT's depth conditioning is meaningless on it);
    - ``var_names`` holding gene symbols or Ensembl gene IDs (auto-detected);
    - ``organism`` present in the checkpoint's supported organisms — pass
      ``NCBITaxon:10090`` for mouse, which is *not* inferable from the data.

    Args:
    ----
        adata: Input AnnData (not modified in place).
        batch_key: Unused; kept for the shared method signature. scPRINT
            embeddings are batch-agnostic.
        ckpt_path: Path to a scPRINT ``.ckpt`` file.
        organism: NCBITaxon ontology term for the species of every cell.
        max_len: Number of genes per cell passed to the transformer.
        batch_size: Inference batch size.
        num_workers: DataLoader workers.
        how: Gene-selection strategy passed to ``Embedder``.
        pred_embedding: Class token(s) whose embedding is returned.
        device: Torch device for inference; defaults to cuda when available.

    Returns:
    -------
        Float32 array of shape ``(n_cells, embedding_dim)``, or ``None`` if the
        optional dependency is missing. Misuse (non-count input, unsupported
        organism, unmapped genes) raises rather than returning ``None``, so a
        broken baseline cannot be mistaken for a skipped one.

    """
    try:
        import torch
        from scdataloader import Preprocessor
        from scprint import scPrint
        from scprint.tasks import Embedder
    except ImportError:
        logger.warning(
            "scPRINT not installed. Install with: pip install scprint 'lamindb[bionty]'\n"
            "  initialise lamindb:  lamin init --storage <dir> --schema bionty\n"
            "  and download the checkpoint with:\n"
            "  hf download jkobject/scPRINT medium-v1.5.ckpt --local-dir ."
        )
        return None

    adata = adata.copy()

    counts_layer = _counts_source(adata)
    if counts_layer is None:
        raise ValueError(
            "scPRINT requires raw counts, but neither X nor any layer of this "
            "AnnData is non-negative and integer-valued. Refusing to embed "
            "normalized data — scPRINT conditions on sequencing depth."
        )
    logger.info(f"scPRINT: using raw counts from {counts_layer}.")

    n_ensembl = sum(
        1 for g in adata.var_names[:200] if str(g).upper().startswith("ENS")
    )
    is_symbol = n_ensembl < 100
    logger.info(
        f"scPRINT: treating var_names as {'gene symbols' if is_symbol else 'Ensembl IDs'}."
    )

    adata.obs["organism_ontology_term_id"] = organism

    model = scPrint.load_from_checkpoint(ckpt_path, precpt_gene_emb=None)
    if organism not in model.organisms:
        raise ValueError(
            f"Checkpoint {ckpt_path} supports organisms {model.organisms}, "
            f"but {organism} was requested."
        )
    model.eval()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Embedder reads model.device to place batches, so the model must be moved
    # first or the whole forward pass silently runs on CPU.
    model.to(device)

    n_genes_before = adata.n_vars
    preprocessor = Preprocessor(
        use_layer=None if counts_layer == "X" else counts_layer,
        is_symbol=is_symbol,
        force_preprocess=True,
        skip_validate=True,
        do_postp=False,
    )
    adata = preprocessor(adata)
    logger.info(
        f"scPRINT: preprocessor mapped {adata.n_vars}/{n_genes_before} genes "
        f"({adata.n_vars / n_genes_before:.1%}) onto the model vocabulary."
    )

    embedder = Embedder(
        how=how,
        max_len=max_len,
        batch_size=batch_size,
        num_workers=num_workers,
        doclass=False,
        doplot=False,
        output_expression="none",
        pred_embedding=list(pred_embedding),
    )

    # Embedder writes an intermediate h5ad to "<logger.save_dir or 'data'>/" and
    # reads it back, so run it inside a scratch cwd to keep the call side-effect free.
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="scprint_embed_") as scratch:
        (Path(scratch) / "data").mkdir()
        try:
            os.chdir(scratch)
            embedded, _ = embedder(model, adata, cache=False)
        finally:
            os.chdir(original_cwd)

    if "scprint" not in embedded.obsm:
        raise RuntimeError(
            f"scPRINT produced no 'scprint' embedding; obsm has "
            f"{sorted(embedded.obsm)}."
        )
    emb = np.asarray(embedded.obsm["scprint"], dtype=np.float32)
    if emb.shape[0] != adata.n_obs:
        raise RuntimeError(
            f"scPRINT returned {emb.shape[0]} embeddings for {adata.n_obs} cells."
        )
    if not np.isfinite(emb).all() or np.allclose(emb, 0):
        raise RuntimeError(
            "scPRINT embeddings are degenerate (all-zero or non-finite)."
        )
    logger.info(f"scPRINT: produced embeddings of shape {emb.shape}.")
    return emb


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
    scprint_organism: str = SCPRINT_HUMAN,
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
        extra = (
            {"ckpt_path": scprint_ckpt, "organism": scprint_organism}
            if method == "scPRINT"
            else {}
        )
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
    parser.add_argument(
        "--scprint-organism",
        default=SCPRINT_HUMAN,
        help=(
            "NCBITaxon term for the species of every cell, e.g. "
            f"{SCPRINT_HUMAN} (human) or {SCPRINT_MOUSE} (mouse). Not inferable "
            "from the data, so set it explicitly for non-human datasets."
        ),
    )
    args = parser.parse_args()

    run_baselines(
        adata_path=args.input,
        batch_key=args.batch_key,
        methods=args.methods,
        scprint_ckpt=args.scprint_ckpt,
        scprint_organism=args.scprint_organism,
    )


if __name__ == "__main__":
    main()
