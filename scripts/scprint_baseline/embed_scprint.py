#!/usr/bin/env python3
"""
Embed cells with a frozen scPRINT model and store them in ``obsm["scPRINT"]``.

This is a standalone script, not part of the ``bmfm_targets`` package. scPRINT pins
``torch==2.2.0``, ``torchmetrics==1.6.0``, ``numpy<2`` and ``python<3.13``, which
cannot be installed alongside ``bmfm_targets``. Run it in its own environment (see
README.md); it does not import ``bmfm_targets``. Scoring the stored embedding is done
in the main environment like any other baseline.

Usage:
    python embed_scprint.py data.h5ad --ckpt large-v1.ckpt
    python embed_scprint.py data.h5ad --ckpt large-v1.ckpt --organism NCBITaxon:10090
"""

import argparse
import logging
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from anndata import AnnData

logger = logging.getLogger(__name__)

SCPRINT_DEFAULT_CHECKPOINT = "large-v1.ckpt"
SCPRINT_HUMAN = "NCBITaxon:9606"
SCPRINT_MOUSE = "NCBITaxon:10090"
OBSM_KEY = "scPRINT"


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


# Ensembl ID prefix per organism, used to attribute retired gene IDs.
_ENSEMBL_PREFIX = {SCPRINT_HUMAN: "ENSG", SCPRINT_MOUSE: "ENSMUSG"}

GenesByOrganism = Sequence[str] | Mapping[str, Sequence[str]]


def _genes_by_organism(
    model_genes: GenesByOrganism, organisms: Sequence[str]
) -> dict[str, list[str]]:
    """
    The checkpoint's gene list per organism, in the checkpoint's own order.

    scPRINT-1 exposes ``model.genes`` as one flat list over human and mouse, so the
    organism has to be recovered from the Ensembl prefix. scPRINT-2 checkpoints cover
    16 organisms and keep the vocabulary as a dict keyed by organism; prefixes are
    ambiguous at that width (``"ENSG"`` also prefixes chicken's ``ENSGALG``), so when
    the dict is available it must be used rather than guessed at.
    """
    if isinstance(model_genes, Mapping):
        return {o: [str(g) for g in model_genes[o]] for o in organisms}
    mg = [str(g) for g in model_genes]
    return {o: [g for g in mg if g.startswith(_ENSEMBL_PREFIX[o])] for o in organisms}


def _scprint_index_drift(
    model_genes: GenesByOrganism, organisms: Sequence[str]
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

    Checking each organism independently is sufficient even though ``start_idx`` is
    computed on the concatenated frame: no drift anywhere means every per-organism
    count matches, hence every block offset matches too. It is also necessary to pass
    *all* the checkpoint's organisms, because a gene missing from an early block
    shifts every later block — for ji9krimq, 3 missing mouse genes displace all 20,004
    human genes on their own.
    """
    from scdataloader.utils import load_genes

    want_by_org = _genes_by_organism(model_genes, organisms)
    gdf = load_genes(list(organisms))
    drift = {}
    for org in organisms:
        # Model genes in the model's order, vs. what the Collator's mask selects.
        want = want_by_org[org]
        in_model = set(want)
        got = [str(g) for g in gdf[gdf.organism == org].index if str(g) in in_model]
        first = next(
            (i for i in range(min(len(got), len(want))) if got[i] != want[i]), None
        )
        if first is None and len(got) != len(want):
            first = min(len(got), len(want))
        drift[org] = (len(want), len(got), first)
    return drift


def _register_retired_genes(
    model_genes: GenesByOrganism, organisms: Sequence[str]
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

    want_by_org = _genes_by_organism(model_genes, organisms)
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
        missing = [g for g in want_by_org[org] if g not in known]
        for gid in missing:
            # Retired IDs have no symbol; reuse the ID so QC-marker string tests don't match.
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
    model_genes: GenesByOrganism, organisms: Sequence[str], fix: bool = True
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
) -> np.ndarray:
    """
    Embed cells with the pretrained scPRINT model (frozen weights, no finetuning).

    Mirrors the reference configuration in scPRINT's own integration benchmark
    (``scprint.tasks.cell_emb.default_benchmark``): the AnnData is run through
    ``scdataloader.Preprocessor`` to map gene identifiers onto the checkpoint's
    Ensembl-keyed vocabulary, then ``Embedder`` produces the cell embedding from
    the ``cell_type_ontology_term_id`` class token. No classification heads or
    plotting are run.

    Requires ``scprint`` and ``lamindb[bionty]``; see README.md for setup.

    Input requirements, all of which are checked rather than assumed:
    - raw integer counts in ``X`` or in a layer (log-normalized input is rejected,
      because scPRINT's depth conditioning is meaningless on it);
    - ``var_names`` holding gene symbols or Ensembl gene IDs (auto-detected);
    - ``organism`` present in the checkpoint's supported organisms — pass
      ``NCBITaxon:10090`` for mouse, which is *not* inferable from the data.

    Args:
    ----
        adata: Input AnnData (not modified in place).
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
        Float32 array of shape ``(n_cells, embedding_dim)``. Misuse (non-count input,
        unsupported organism, unmapped genes) raises; nothing is silently skipped.
    """
    import torch
    from scdataloader import Preprocessor
    from scprint import scPrint
    from scprint.tasks import Embedder

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
    # Check gene-index alignment before any GPU work.
    _ensure_scprint_gene_alignment(model.genes, model.organisms, fix=align_ontology)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Embedder reads model.device, so move the model first.
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
    # Preprocessor pads var to the full vocabulary; count matched genes via uns["unseen_genes"].
    unseen = {str(g) for g in adata.uns.get("unseen_genes", [])}
    observed = {str(g) for g in adata.var_names} - unseen
    logger.info(
        f"scPRINT: {len(observed)}/{n_genes_before} input genes "
        f"({len(observed) / n_genes_before:.1%}) mapped onto the gene ontology."
    )
    # Also report coverage of model.genes, which decides how much signal reaches the model.
    model_genes = {str(g) for g in model.genes}
    padded = [str(g) for g in adata.var_names]
    n_model_organism = len(model_genes & set(padded))
    n_usable = len(model_genes & observed)
    # Re-check alignment on the array handed to the Collator: its mask is positional.
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
    # Positionally matched to the caller's cells, so the Preprocessor must not drop cells.
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

    # Embedder writes an intermediate h5ad; run in a scratch cwd.
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="scprint_embed_") as scratch:
        (Path(scratch) / "data").mkdir()
        try:
            os.chdir(scratch)
            embedded, _ = embedder(model, adata, cache=False)
        finally:
            os.chdir(original_cwd)

    # The obsm key differs across scprint releases ("scprint" vs "scprint_emb"); accept either.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="Path to input h5ad (raw counts)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the result (default: overwrite the input file).",
    )
    parser.add_argument("--ckpt", default=SCPRINT_DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--organism",
        default=SCPRINT_HUMAN,
        help=(
            f"NCBITaxon term for every cell, e.g. {SCPRINT_HUMAN} (human) or "
            f"{SCPRINT_MOUSE} (mouse). Not inferable from the data."
        ),
    )
    parser.add_argument(
        "--no-align-ontology",
        action="store_true",
        help="Raise on a misaligned gene index instead of repairing the ontology.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    import anndata

    adata = anndata.read_h5ad(args.input)
    adata.obsm[OBSM_KEY] = compute_scprint(
        adata,
        ckpt_path=args.ckpt,
        organism=args.organism,
        align_ontology=not args.no_align_ontology,
    )
    output = args.output or args.input
    adata.write_h5ad(output)
    logger.info(
        f"Stored {adata.obsm[OBSM_KEY].shape} in obsm['{OBSM_KEY}'] of {output}"
    )


if __name__ == "__main__":
    main()
