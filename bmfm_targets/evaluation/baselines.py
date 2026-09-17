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
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import scanpy as sc
from anndata import AnnData

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

AVAILABLE_METHODS = ["Unintegrated", "Harmony", "Scanorama", "LIGER", "scPRINT"]

# HuggingFace checkpoint used for the scPRINT baseline (jkobject/scPRINT repo).
# large-v1 (101.1M params, d_model 512, nlayers 16) and NOT the medium-v1.5
# quickstart default (18.7M, 256, 8): scPRINT's zero-shot claim is about "our large
# cell model", and on the 9 sceval datasets medium-v1.5 scores 0.093 lower on
# avg_bio conservation (0.484 vs 0.577). See docs/scprint_baseline_run_plan.md §7.
SCPRINT_DEFAULT_CHECKPOINT = "large-v1.ckpt"
SCPRINT_HUMAN = "NCBITaxon:9606"
SCPRINT_MOUSE = "NCBITaxon:10090"


def get_pca(adata: AnnData, batch_key: str | None, n_comps: int = 30) -> np.ndarray:
    """Compute PCA on (optionally normalized) data with HVG selection."""
    # Imported lazily: importing bmfm_targets pulls in peft/transformers, which
    # cannot coexist with the torch 2.2.0 that scprint pins. Keeping this out of
    # module scope lets the scPRINT baseline run from a dedicated scprint env by
    # loading this file directly.
    from bmfm_targets.datasets.datasets_utils import guess_if_raw

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


# Ensembl ID prefix per organism. Needed only to attribute a *retired* gene ID to
# an organism: such an ID is by definition absent from the ontology, so the ontology
# cannot be asked which organism it belonged to. scPRINT's checkpoints cover human
# and mouse only, and compute_scprint checks ``organism in model.organisms``.
_ENSEMBL_PREFIX = {SCPRINT_HUMAN: "ENSG", SCPRINT_MOUSE: "ENSMUSG"}


def _scprint_index_drift(
    model_genes: Sequence[str], organisms: Sequence[str]
) -> dict[str, tuple[int, int, int | None]]:
    """
    Per organism, where the Collator's gene index starts disagreeing with the model.

    ``Collator`` masks the ontology's genes for an organism with
    ``index.isin(model.genes)`` and sends ``position_in_mask + start_idx`` as the
    embedding-table index (``scdataloader/collator.py:288-291``). Both sides are
    sorted by Ensembl ID, so that is correct only while the masked ontology genes
    are exactly ``model.genes`` for the organism, position for position.

    A checkpoint's vocabulary is fixed at its training-time Ensembl release, while
    bionty ships one current release, so IDs retired in between are missing from the
    ontology. Each missing ID shifts every later gene by one, and **nothing raises**:
    ``_setup`` prints "Some valid genes are not in the genedf!!!" and carries on, and
    the embedding that comes out is finite, non-zero and non-degenerate, so every
    guard in ``compute_scprint`` passes while the numbers describe noise. For
    medium-v1.5 on release-114 this silently misindexed 66.4% of human genes; large-v1
    ships the same gene vocabulary and so is affected identically.

    Returns ``{organism: (n_model_genes, n_found_in_ontology, first_bad_position)}``,
    where ``first_bad_position`` is None when the two agree.
    """
    from scdataloader.utils import load_genes

    mg = [str(g) for g in model_genes]
    in_model = set(mg)
    gdf = load_genes(list(organisms))
    drift = {}
    for org in organisms:
        # The model's genes for this organism, in the model's own order ...
        want = [g for g in mg if g.startswith(_ENSEMBL_PREFIX[org])]
        # ... versus what the Collator's mask actually selects, in ontology order.
        got = [str(g) for g in gdf[gdf.organism == org].index if str(g) in in_model]
        first = next(
            (i for i in range(min(len(got), len(want))) if got[i] != want[i]), None
        )
        if first is None and len(got) != len(want):
            first = min(len(got), len(want))
        drift[org] = (len(want), len(got), first)
    return drift


def _register_retired_genes(
    model_genes: Sequence[str], organisms: Sequence[str]
) -> int:
    """
    Add the checkpoint's retired gene IDs to the ontology so the index lines up.

    Idempotent, and benign in every respect that touches the embedding:
    ``Preprocessor`` zero-pads the restored genes into the AnnData (no real dataset
    contains a retired ID), and ``how="random expr"`` samples only from
    ``np.where(expr > 0)``, so a zero column can never be selected. The restored
    genes change gene *indexing* only, never model input.
    """
    import bionty as bt
    from scdataloader.utils import load_genes

    gdf = load_genes(list(organisms))
    added = 0
    for org in organisms:
        record = bt.Organism.filter(ontology_id=org).first()
        if record is None:
            raise RuntimeError(
                f"Organism {org} is not in the lamindb ontology; run "
                f"scdataloader.utils.populate_my_ontology first."
            )
        known = {str(g) for g in gdf[gdf.organism == org].index}
        missing = [
            g
            for g in map(str, model_genes)
            if g.startswith(_ENSEMBL_PREFIX[org]) and g not in known
        ]
        for gid in missing:
            # A retired ID has no current symbol, so reuse the ID. It cannot match
            # "MT-", "RPS"/"RPL" or "^HB[^(P)]", which is what we want: load_genes
            # uses those string tests to flag QC markers, and these genes carry no
            # counts and must not be treated as such.
            bt.Gene(
                ensembl_gene_id=gid,
                symbol=gid,
                organism=record,
                description="retired Ensembl ID retained for scPRINT checkpoint "
                "gene-index alignment",
            ).save()
            added += 1
    return added


def _ensure_scprint_gene_alignment(
    model_genes: Sequence[str], organisms: Sequence[str], fix: bool = True
) -> None:
    """Verify the Collator/model gene index agrees, repairing the ontology if asked."""
    drift = _scprint_index_drift(model_genes, organisms)
    bad = {o: d for o, d in drift.items() if d[2] is not None}
    if not bad:
        return

    detail = "; ".join(
        f"{o}: model={n_m} ontology={n_o} first misaligned at {first}"
        for o, (n_m, n_o, first) in bad.items()
    )
    if not fix:
        raise RuntimeError(
            f"scPRINT gene index is misaligned against the ontology ({detail}). "
            f"Embedding now would produce plausible-looking noise. Re-run with "
            f"align_ontology=True to register the checkpoint's retired gene IDs."
        )

    logger.warning(f"scPRINT: gene index misaligned ({detail}); repairing ontology.")
    added = _register_retired_genes(model_genes, organisms)
    still = {
        o: d
        for o, d in _scprint_index_drift(model_genes, organisms).items()
        if d[2] is not None
    }
    if still:
        raise RuntimeError(
            f"scPRINT gene index is still misaligned after registering {added} "
            f"retired IDs: {still}. Refusing to embed."
        )
    logger.info(
        f"scPRINT: registered {added} retired gene IDs; Collator indices now match "
        f"the model embedding table."
    )


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
    min_nnz_genes: int = 0,
    min_valid_genes_id: int = 1,
    min_dataset_size: int = 1,
    align_ontology: bool = True,
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
        hf download jkobject/scPRINT large-v1.ckpt --local-dir .

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
        min_nnz_genes: Cells with fewer expressed genes are dropped by
            ``scdataloader.Preprocessor`` (its own default is 200). Defaults to 0
            here: this is frozen inference on a fixed cell set, and dropping
            cells would change which cells the reported metrics describe, making
            them incomparable to other methods scored on the same file.
        min_valid_genes_id: Minimum genes mapping onto the model vocabulary
            before ``Preprocessor`` rejects the dataset outright (its own default
            is 10_000). Defaults to 1 here so that low-gene-count inputs are
            embedded and their mapping rate reported, rather than raising. Both
            thresholds are training-data curation gates, not inference
            requirements; the logged mapping rate is what tells you whether the
            resulting embedding is trustworthy.
        min_dataset_size: Minimum cells before ``Preprocessor`` rejects the
            dataset (its own default is 100).
        align_ontology: Register the checkpoint's retired Ensembl IDs in the
            lamindb ontology when the gene index does not line up. On by default
            because the failure it prevents is silent — see
            ``_scprint_index_drift``. Set False to raise instead of writing to the
            ontology; the embedding is then refused, never run misaligned.

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
            "  hf download jkobject/scPRINT large-v1.ckpt --local-dir ."
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
    # Before any GPU work: the gene index must line up, and it is cheap to check.
    _ensure_scprint_gene_alignment(model.genes, model.organisms, fix=align_ontology)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Embedder reads model.device to place batches, so the model must be moved
    # first or the whole forward pass silently runs on CPU.
    model.to(device)

    n_genes_before = adata.n_vars
    n_obs_before = adata.n_obs
    preprocessor = Preprocessor(
        use_layer=None if counts_layer == "X" else counts_layer,
        is_symbol=is_symbol,
        force_preprocess=True,
        skip_validate=True,
        do_postp=False,
        min_nnz_genes=min_nnz_genes,
        min_valid_genes_id=min_valid_genes_id,
        min_dataset_size=min_dataset_size,
    )
    adata = preprocessor(adata)
    # Preprocessor zero-pads the var axis out to the model's entire vocabulary
    # (scdataloader/preprocess.py:293-304), so adata.n_vars afterwards is the
    # vocabulary size -- comparing it to the input gene count reports nonsense
    # such as "2606% mapped". The padded-in genes are recorded in
    # uns["unseen_genes"], so the genes that actually matched are the difference.
    unseen = {str(g) for g in adata.uns.get("unseen_genes", [])}
    observed = {str(g) for g in adata.var_names} - unseen
    logger.info(
        f"scPRINT: {len(observed)}/{n_genes_before} input genes "
        f"({len(observed) / n_genes_before:.1%}) mapped onto the gene ontology."
    )
    # That rate is not yet the useful one. The padded var axis spans every gene the
    # ontology knows for this organism, but the checkpoint's own input space is far
    # smaller (44_756 genes, 23_149 human + 21_607 mouse; medium-v1.5 and large-v1
    # share this vocabulary) and Collator
    # intersects against model.genes. Coverage of *that* set is what decides how
    # much real signal reaches the transformer, so report it explicitly. Since the
    # padded var axis is exactly the organism's ontology genes, intersecting it with
    # model.genes yields the model's gene count for this organism specifically.
    model_genes = {str(g) for g in model.genes}
    padded = [str(g) for g in adata.var_names]
    n_model_organism = len(model_genes & set(padded))
    n_usable = len(model_genes & observed)
    # The alignment check above ran against the ontology. This asserts the same
    # invariant on the array actually being handed over, which is what the Collator
    # will mask: the padded axis must be sorted (its mask is POSITIONAL, so any
    # other order scrambles genes), and masking it must reproduce a contiguous run
    # of model.genes.
    mg_order = [str(g) for g in model.genes]
    selected = [g for g in padded if g in model_genes]
    start = mg_order.index(selected[0]) if selected else 0
    if padded != sorted(padded) or mg_order[start : start + len(selected)] != selected:
        raise RuntimeError(
            "scPRINT gene index does not line up with the model's embedding table "
            "after preprocessing; the embedding would be noise. Check that the "
            "lamindb ontology matches the checkpoint vocabulary."
        )
    if n_model_organism:
        logger.info(
            f"scPRINT: {n_usable}/{n_model_organism} "
            f"({n_usable / n_model_organism:.1%}) of the model's genes for this "
            f"organism carry data; the rest are zero-filled. This, not the "
            f"ontology rate above, is the fraction of the model's input that is "
            f"real -- a low value means the embedding rests on little signal."
        )
    # The returned array is positionally matched against the caller's cells, so a
    # Preprocessor that filters cells would silently misalign every label.
    if adata.n_obs != n_obs_before:
        raise RuntimeError(
            f"scdataloader.Preprocessor changed the cell count "
            f"({n_obs_before} -> {adata.n_obs}); the returned embedding could no "
            f"longer be aligned with the input cells."
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

    # The key Embedder writes moved between releases: scprint <=1.6.4 writes
    # "scprint", >=2.x writes "scprint_emb" (scprint/tasks/cell_emb.py:196).
    # scprint 2.3.8's own default_benchmark still reads "scprint" and is
    # therefore broken, so neither spelling can be assumed -- accept either.
    emb_key = next((k for k in ("scprint_emb", "scprint") if k in embedded.obsm), None)
    if emb_key is None:
        raise RuntimeError(
            f"scPRINT produced no 'scprint_emb'/'scprint' embedding; obsm has "
            f"{sorted(embedded.obsm)}."
        )
    emb = np.asarray(embedded.obsm[emb_key], dtype=np.float32)
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
