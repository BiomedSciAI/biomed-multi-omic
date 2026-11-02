"""
Tests for the scPRINT baseline in bmfm_targets.evaluation.baselines.

scPRINT is an optional dependency, so all tests mock the scprint/scdataloader
packages via sys.modules injection.  They verify the integration contract —
counts validation, gene-identifier preprocessing, device placement, canonical
Embedder settings, and obsm storage — without requiring a real checkpoint or GPU.

The mocks deliberately reproduce scPRINT's real behaviour rather than inventing
their own obsm key, since a suite that invents one cannot catch a key mismatch in
the code under test.  Which key is real depends on the installed version:
``scprint <= 1.6.4`` writes ``obsm["scprint"]``, ``>= 2.x`` writes
``obsm["scprint_emb"]`` (``scprint/tasks/cell_emb.py``).  2.3.8's own
``default_benchmark`` still reads ``"scprint"`` and is therefore broken against
its own Embedder, so neither spelling can be assumed.  The code accepts either,
and both are exercised here.
"""

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from bmfm_targets.evaluation.baselines import (
    SCPRINT_DEFAULT_CHECKPOINT,
    SCPRINT_HUMAN,
    SCPRINT_MOUSE,
    _counts_source,
    _genes_by_organism,
    _scprint_index_drift,
    compute_scprint,
    run_baselines,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_CELLS = 20
N_GENES = 200
EMB_DIM = 512

# The key scPRINT itself writes (version-dependent; see the module docstring).
SCPRINT_OBSM_KEY = "scprint"
SCPRINT_OBSM_KEY_V2 = "scprint_emb"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def raw_counts_adata():
    """AnnData with raw integer counts in X and HUGO-symbol var_names."""
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32))
    obs = pd.DataFrame(
        {"batch": pd.Categorical(["batch_A"] * 10 + ["batch_B"] * 10)},
        index=[f"cell_{i}" for i in range(N_CELLS)],
    )
    var = pd.DataFrame(index=[f"GENE{i}" for i in range(N_GENES)])
    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture()
def ensembl_counts_adata(raw_counts_adata):
    """Same counts, but var_names are Ensembl gene IDs."""
    adata = raw_counts_adata.copy()
    adata.var_names = [f"ENSG{i:011d}" for i in range(N_GENES)]
    return adata


@pytest.fixture()
def normalised_adata_with_counts_layer(raw_counts_adata):
    """AnnData where X is log-normalised but raw counts live in layers['counts']."""
    adata = raw_counts_adata.copy()
    adata.layers["counts"] = adata.X.copy()
    dense = adata.X.toarray()
    normed = dense / (dense.sum(axis=1, keepdims=True) + 1e-8) * 1e4
    adata.X = sp.csr_matrix(np.log1p(normed).astype(np.float32))
    return adata


@pytest.fixture()
def lognormed_only_adata(raw_counts_adata):
    """AnnData with no raw counts anywhere — must be rejected."""
    adata = raw_counts_adata.copy()
    dense = adata.X.toarray()
    normed = dense / (dense.sum(axis=1, keepdims=True) + 1e-8) * 1e4
    adata.X = sp.csr_matrix(np.log1p(normed).astype(np.float32))
    return adata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def mock_scprint(
    embedder_side_effect=None,
    n_cells=N_CELLS,
    emb_dim=EMB_DIM,
    organisms=(SCPRINT_HUMAN, SCPRINT_MOUSE),
    obsm_key=SCPRINT_OBSM_KEY,
    retired_genes=(),
):
    """
    Inject mock scprint/scdataloader/bionty modules into sys.modules.

    Yields a namespace with the mocked ``scPrint`` class, ``Embedder`` class,
    ``Preprocessor`` class, the model instance, and ``ontology`` — a dict standing
    in for the lamindb gene ontology, which ``load_genes`` reads and
    ``bt.Gene(...).save()`` writes, so the repair path can be exercised end to end.

    ``retired_genes`` names checkpoint genes to withhold from that ontology,
    reproducing the Ensembl-release drift that misaligns the gene index.
    """
    fake_emb = (
        np.random.default_rng(42).standard_normal((n_cells, emb_dim)).astype(np.float32)
    )

    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    mock_model.organisms = list(organisms)
    mock_model.genes = [f"ENSG{i:011d}" for i in range(N_GENES)]

    # Stands in for the lamindb ontology: {ensembl_id: organism}.
    ontology = {
        g: SCPRINT_HUMAN for g in mock_model.genes if g not in set(retired_genes)
    }

    def _load_genes(orgs):
        # Real load_genes returns a frame indexed by Ensembl ID, sorted ascending.
        ids = sorted(ontology)
        return pd.DataFrame({"organism": [ontology[g] for g in ids]}, index=ids)

    mock_scprint_cls = MagicMock()
    mock_scprint_cls.load_from_checkpoint.return_value = mock_model

    def _fake_preprocess(adata):
        """
        Reproduce the var-axis behaviour the code under test depends on.

        The real Preprocessor maps identifiers onto Ensembl IDs, zero-pads the var
        axis out to the organism's entire ontology, sorts it ascending, and records
        the padded-in genes in ``uns["unseen_genes"]``. The sorting matters: the
        Collator's gene mask is positional, so an identity mock cannot exercise the
        alignment checks at all.
        """
        ids = sorted(ontology)
        n_keep = min(adata.n_vars, len(ids))
        src = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
        padded = np.zeros((adata.n_obs, len(ids)), dtype=np.float32)
        padded[:, :n_keep] = src[:, :n_keep]
        out = ad.AnnData(X=padded, obs=adata.obs.copy(), var=pd.DataFrame(index=ids))
        out.uns["unseen_genes"] = ids[n_keep:]
        return out

    # Preprocessor is a callable object: Preprocessor(**kwargs)(adata) -> adata.
    mock_preprocessor_cls = MagicMock()
    mock_preprocessor_cls.return_value = MagicMock(side_effect=_fake_preprocess)

    if embedder_side_effect is None:

        def _default_embed(model, adata, cache=False):
            result = adata.copy()
            result.obsm[obsm_key] = fake_emb[: adata.n_obs]
            return result, {}

        embedder_side_effect = _default_embed

    mock_embedder_cls = MagicMock(
        return_value=MagicMock(side_effect=embedder_side_effect)
    )

    mock_scprint_mod = MagicMock()
    mock_scprint_mod.scPrint = mock_scprint_cls
    mock_tasks_mod = MagicMock()
    mock_tasks_mod.Embedder = mock_embedder_cls
    mock_scdataloader_mod = MagicMock()
    mock_scdataloader_mod.Preprocessor = mock_preprocessor_cls
    mock_utils_mod = MagicMock()
    mock_utils_mod.load_genes = MagicMock(side_effect=_load_genes)

    # bionty stands in for the ontology's write side: saving a Gene record inserts
    # it, so a repair is visible to the next load_genes call, exactly as in lamindb.
    def _organism_filter(ontology_id=None):
        found = MagicMock()
        found.first.return_value = ontology_id
        return found

    def _make_gene(ensembl_gene_id, symbol, organism, description=""):
        record = MagicMock()
        record.save = lambda: ontology.__setitem__(ensembl_gene_id, organism)
        return record

    mock_bionty_mod = MagicMock()
    mock_bionty_mod.Organism.filter = MagicMock(side_effect=_organism_filter)
    mock_bionty_mod.Gene = MagicMock(side_effect=_make_gene)

    injected = {
        "scprint": mock_scprint_mod,
        "scprint.tasks": mock_tasks_mod,
        "scdataloader": mock_scdataloader_mod,
        "scdataloader.utils": mock_utils_mod,
        "bionty": mock_bionty_mod,
    }
    orig = {k: sys.modules.get(k) for k in injected}
    sys.modules.update(injected)
    try:
        yield MagicMock(
            scPrint=mock_scprint_cls,
            Embedder=mock_embedder_cls,
            Preprocessor=mock_preprocessor_cls,
            model=mock_model,
            bionty=mock_bionty_mod,
            ontology=ontology,
        )
    finally:
        for k, v in orig.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _embed_writing(key=SCPRINT_OBSM_KEY, record=None):
    """Embedder stand-in that writes a non-degenerate embedding under ``key``."""

    def embed(model, adata, cache=False):
        if record is not None:
            record["organism"] = set(adata.obs["organism_ontology_term_id"])
        result = adata.copy()
        result.obsm[key] = np.ones((adata.n_obs, EMB_DIM), dtype=np.float32)
        return result, {}

    return embed


RETIRED = f"ENSG{100:011d}"  # mid-vocabulary: withholding it shifts every later gene


# --- _counts_source ---------------------------------------------------------


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        ("raw_counts_adata", "X"),
        ("normalised_adata_with_counts_layer", "counts"),
        ("lognormed_only_adata", None),
    ],
)
def test_counts_source(request, fixture, expected):
    assert _counts_source(request.getfixturevalue(fixture)) == expected


# --- compute_scprint: output and failure modes -----------------------------


@pytest.mark.parametrize("key", [SCPRINT_OBSM_KEY, SCPRINT_OBSM_KEY_V2])
def test_reads_either_obsm_key_and_leaves_input_untouched(raw_counts_adata, key):
    """Scprint <=1.6.4 writes ``scprint``, >=2.x writes ``scprint_emb``; read both."""
    with mock_scprint(embedder_side_effect=_embed_writing(key)):
        result = compute_scprint(raw_counts_adata, batch_key="batch", device="cpu")

    assert result.shape == (N_CELLS, EMB_DIM)
    assert result.dtype == np.float32
    assert "organism_ontology_term_id" not in raw_counts_adata.obs.columns
    assert SCPRINT_OBSM_KEY not in raw_counts_adata.obsm


@pytest.mark.parametrize(
    ("writes", "match"),
    [
        ({}, "no 'scprint_emb'/'scprint' embedding"),
        ({"scprint": 0.0}, "degenerate"),
    ],
)
def test_missing_or_degenerate_embedding_raises(raw_counts_adata, writes, match):
    def embed(model, adata, cache=False):
        result = adata.copy()
        for key, value in writes.items():
            result.obsm[key] = np.full((adata.n_obs, EMB_DIM), value, np.float32)
        return result, {}

    with mock_scprint(embedder_side_effect=embed):
        with pytest.raises(RuntimeError, match=match):
            compute_scprint(raw_counts_adata, batch_key="batch")


def test_rejects_lognormed_input(lognormed_only_adata):
    """ScPRINT conditions on depth, so normalised input gives a plausible wrong number."""
    with mock_scprint():
        with pytest.raises(ValueError, match="requires raw counts"):
            compute_scprint(lognormed_only_adata, batch_key="batch")


def test_returns_none_when_scprint_not_installed(raw_counts_adata):
    keys = ("scprint", "scprint.tasks", "scdataloader")
    orig = {k: sys.modules.get(k) for k in keys}
    sys.modules.update(dict.fromkeys(keys))  # None makes `import <key>` raise
    try:
        result = compute_scprint(raw_counts_adata, batch_key="batch")
    finally:
        for k, v in orig.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    assert result is None


# --- compute_scprint: how scPRINT is called --------------------------------


def test_model_and_embedder_are_configured_like_scprints_own_benchmark(
    raw_counts_adata,
):
    """Settings from ``scprint.tasks.cell_emb.default_benchmark``, plus device placement."""
    with mock_scprint() as mocks:
        compute_scprint(raw_counts_adata, batch_key="batch", device="cpu")

    mocks.scPrint.load_from_checkpoint.assert_called_once_with(
        SCPRINT_DEFAULT_CHECKPOINT, precpt_gene_emb=None
    )
    mocks.model.to.assert_called_once_with("cpu")  # Embedder reads model.device

    kwargs = mocks.Embedder.call_args.kwargs
    assert kwargs["how"] == "random expr"
    assert kwargs["pred_embedding"] == ["cell_type_ontology_term_id"]
    assert not kwargs["doplot"]
    assert not kwargs["doclass"]
    assert kwargs["output_expression"] == "none"
    assert "dtype" not in kwargs  # CUDA autocast rejects an explicit float32


@pytest.mark.parametrize(
    ("fixture", "is_symbol", "use_layer"),
    [
        ("raw_counts_adata", True, None),
        ("ensembl_counts_adata", False, None),
        ("normalised_adata_with_counts_layer", True, "counts"),
    ],
)
def test_preprocessor_gets_identifier_type_and_counts_location(
    request, fixture, is_symbol, use_layer
):
    adata = request.getfixturevalue(fixture)
    with mock_scprint() as mocks:
        compute_scprint(adata, batch_key="batch", device="cpu")

    kwargs = mocks.Preprocessor.call_args.kwargs
    assert kwargs["is_symbol"] is is_symbol
    assert kwargs["use_layer"] == use_layer
    assert kwargs["do_postp"] is False  # scPRINT wants counts, not normalised data


def test_organism_must_be_supported_and_is_tagged_on_obs(raw_counts_adata):
    seen = {}
    with mock_scprint(embedder_side_effect=_embed_writing(record=seen)):
        compute_scprint(
            raw_counts_adata, batch_key="batch", organism=SCPRINT_MOUSE, device="cpu"
        )
    assert seen["organism"] == {SCPRINT_MOUSE}

    with mock_scprint(organisms=(SCPRINT_HUMAN,)):
        with pytest.raises(ValueError, match="supports organisms"):
            compute_scprint(raw_counts_adata, batch_key="batch", organism=SCPRINT_MOUSE)


# --- gene-index alignment ---------------------------------------------------
# A misaligned index is silent: the output is finite and non-degenerate.


def test_drift_starts_at_the_retired_gene_and_flat_and_mapping_forms_agree():
    with mock_scprint() as mocks:
        clean = _scprint_index_drift(mocks.model.genes, [SCPRINT_HUMAN])
    assert clean[SCPRINT_HUMAN][2] is None

    with mock_scprint(retired_genes=(RETIRED,)) as mocks:
        flat = _scprint_index_drift(mocks.model.genes, [SCPRINT_HUMAN])
        mapped = _scprint_index_drift(
            {SCPRINT_HUMAN: mocks.model.genes}, [SCPRINT_HUMAN]
        )

    assert flat[SCPRINT_HUMAN] == (N_GENES, N_GENES - 1, 100)
    assert mapped == flat


def test_ensembl_prefix_attribution_is_ambiguous_so_a_mapping_must_win():
    """``"ENSGALG".startswith("ENSG")``: chicken genes would be filed under human."""
    chicken = "NCBITaxon:9031"
    human_genes = [f"ENSG{i:011d}" for i in range(3)]
    chicken_genes = [f"ENSGALG{i:011d}" for i in range(3)]

    flat = _genes_by_organism(human_genes + chicken_genes, [SCPRINT_HUMAN])
    assert flat[SCPRINT_HUMAN] == human_genes + chicken_genes  # over-collects

    mapped = _genes_by_organism(
        {SCPRINT_HUMAN: human_genes, chicken: chicken_genes},
        [SCPRINT_HUMAN, chicken],
    )
    assert mapped == {SCPRINT_HUMAN: human_genes, chicken: chicken_genes}


def test_retired_gene_is_registered_before_embedding(raw_counts_adata):
    with mock_scprint(retired_genes=(RETIRED,)) as mocks:
        result = compute_scprint(raw_counts_adata, batch_key="batch", device="cpu")

        assert mocks.ontology[RETIRED] == SCPRINT_HUMAN
        assert (
            _scprint_index_drift(mocks.model.genes, [SCPRINT_HUMAN])[SCPRINT_HUMAN][2]
            is None
        )
    assert result.shape == (N_CELLS, EMB_DIM)


def test_declined_repair_raises_before_any_write_or_gpu_work(raw_counts_adata):
    with mock_scprint(retired_genes=(RETIRED,)) as mocks:
        with pytest.raises(RuntimeError, match="misaligned"):
            compute_scprint(
                raw_counts_adata, batch_key="batch", device="cpu", align_ontology=False
            )
        mocks.Embedder.assert_not_called()
        mocks.model.to.assert_not_called()
        assert RETIRED not in mocks.ontology


# --- run_baselines wiring ---------------------------------------------------


@pytest.mark.parametrize("organism", [SCPRINT_HUMAN, SCPRINT_MOUSE])
def test_run_baselines_stores_embedding_and_threads_organism(
    raw_counts_adata, tmp_path, organism
):
    h5ad_path = tmp_path / "test.h5ad"
    raw_counts_adata.write_h5ad(h5ad_path)

    seen = {}
    with mock_scprint(embedder_side_effect=_embed_writing(record=seen)):
        run_baselines(
            h5ad_path,
            batch_key="batch",
            methods=["scPRINT"],
            scprint_organism=organism,
        )

    assert ad.io.read_h5ad(h5ad_path).obsm["scPRINT"].shape == (N_CELLS, EMB_DIM)
    assert seen["organism"] == {organism}
