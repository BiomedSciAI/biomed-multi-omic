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

# The key scPRINT itself writes.  Version-dependent -- see the module docstring.
# Do not change without checking cell_emb.py in the version you have installed.
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


# ---------------------------------------------------------------------------
# _counts_source
# ---------------------------------------------------------------------------


def test_counts_source_detects_x(raw_counts_adata):
    assert _counts_source(raw_counts_adata) == "X"


def test_counts_source_prefers_x_but_finds_layer(normalised_adata_with_counts_layer):
    assert _counts_source(normalised_adata_with_counts_layer) == "counts"


def test_counts_source_returns_none_for_lognormed(lognormed_only_adata):
    assert _counts_source(lognormed_only_adata) is None


# ---------------------------------------------------------------------------
# compute_scprint — happy path
# ---------------------------------------------------------------------------


def test_compute_scprint_returns_correct_shape(raw_counts_adata):
    with mock_scprint():
        result = compute_scprint(raw_counts_adata, batch_key="batch")

    assert result is not None
    assert result.shape == (N_CELLS, EMB_DIM)
    assert result.dtype == np.float32


@pytest.mark.parametrize("key", [SCPRINT_OBSM_KEY, SCPRINT_OBSM_KEY_V2])
def test_reads_the_obsm_key_scprint_actually_writes(raw_counts_adata, key):
    """
    Both real spellings must be read.

    Regression on two opposite bugs: the code originally read only
    ``obsm["scprint_emb"]`` while the pinned scprint 1.x wrote ``obsm["scprint"]``,
    and reading only ``"scprint"`` breaks just as silently against scprint 2.x,
    which writes ``"scprint_emb"``.  The 2.x spelling is the one the manuscript
    run actually exercised (scprint 2.3.8).
    """

    def embed_writing_real_key(model, adata, cache=False):
        result = adata.copy()
        result.obsm[key] = np.ones((adata.n_obs, EMB_DIM), dtype=np.float32)
        return result, {}

    with mock_scprint(embedder_side_effect=embed_writing_real_key):
        result = compute_scprint(raw_counts_adata, batch_key="batch")

    assert result is not None, f"must read obsm[{key!r}], a key scPRINT writes"
    assert result.shape == (N_CELLS, EMB_DIM)


def test_raises_when_embedding_key_absent(raw_counts_adata):
    """A missing embedding is a hard failure, not a silent skip."""

    def embed_writing_nothing(model, adata, cache=False):
        return adata.copy(), {}

    with mock_scprint(embedder_side_effect=embed_writing_nothing):
        with pytest.raises(RuntimeError, match="no 'scprint_emb'/'scprint' embedding"):
            compute_scprint(raw_counts_adata, batch_key="batch")


def test_raises_on_degenerate_embedding(raw_counts_adata):
    def embed_all_zeros(model, adata, cache=False):
        result = adata.copy()
        result.obsm["scprint"] = np.zeros((adata.n_obs, EMB_DIM), dtype=np.float32)
        return result, {}

    with mock_scprint(embedder_side_effect=embed_all_zeros):
        with pytest.raises(RuntimeError, match="degenerate"):
            compute_scprint(raw_counts_adata, batch_key="batch")


# ---------------------------------------------------------------------------
# compute_scprint — canonical settings
# ---------------------------------------------------------------------------


def test_embedder_uses_canonical_benchmark_settings(raw_counts_adata):
    """
    Must match scprint.tasks.cell_emb.default_benchmark.

    ``how="random expr"`` and a single ``cell_type_ontology_term_id`` class token
    are the settings scPRINT's own published integration benchmark uses; anything
    else is not the number the paper reports.
    """
    with mock_scprint() as mocks:
        compute_scprint(raw_counts_adata, batch_key="batch")

    kwargs = mocks.Embedder.call_args.kwargs
    assert kwargs["how"] == "random expr"
    assert kwargs["pred_embedding"] == ["cell_type_ontology_term_id"]
    assert kwargs["doplot"] is False
    assert kwargs["doclass"] is False
    assert kwargs["output_expression"] == "none"
    # dtype must not be pinned: scPRINT embeds under CUDA autocast, which
    # rejects float32.
    assert "dtype" not in kwargs


def test_model_loaded_from_checkpoint_and_moved_to_device(raw_counts_adata):
    """
    Embedder places batches via ``model.device``.

    If the model is left on CPU by ``load_from_checkpoint``, the whole forward
    pass silently runs on CPU.
    """
    with mock_scprint() as mocks:
        compute_scprint(raw_counts_adata, batch_key="batch", device="cpu")

    mocks.scPrint.load_from_checkpoint.assert_called_once_with(
        SCPRINT_DEFAULT_CHECKPOINT, precpt_gene_emb=None
    )
    mocks.model.eval.assert_called_once()
    mocks.model.to.assert_called_once_with("cpu")


def test_default_checkpoint_constant():
    assert SCPRINT_DEFAULT_CHECKPOINT == "medium-v1.5.ckpt"


# ---------------------------------------------------------------------------
# compute_scprint — preprocessing
# ---------------------------------------------------------------------------


def test_preprocessor_invoked_with_symbol_flag(raw_counts_adata):
    """
    Gene identifiers must be mapped onto the checkpoint vocabulary.

    ``model.genes`` are Ensembl IDs; feeding HUGO symbols straight to the
    Collator gives near-zero gene overlap and a meaningless embedding.
    """
    with mock_scprint() as mocks:
        compute_scprint(raw_counts_adata, batch_key="batch", device="cpu")

    mocks.Preprocessor.assert_called_once()
    kwargs = mocks.Preprocessor.call_args.kwargs
    assert kwargs["is_symbol"] is True, "GENE* var_names are symbols, not Ensembl IDs"
    assert kwargs["use_layer"] is None, "counts are in X here"
    assert kwargs["do_postp"] is False, "no normalization: scPRINT wants counts"
    mocks.Preprocessor.return_value.assert_called_once()


def test_preprocessor_detects_ensembl_ids(ensembl_counts_adata):
    with mock_scprint() as mocks:
        compute_scprint(ensembl_counts_adata, batch_key="batch", device="cpu")

    assert mocks.Preprocessor.call_args.kwargs["is_symbol"] is False


def test_preprocessor_points_at_counts_layer(normalised_adata_with_counts_layer):
    """When X is log-normalised, the counts layer must be named to Preprocessor."""
    with mock_scprint() as mocks:
        compute_scprint(
            normalised_adata_with_counts_layer, batch_key="batch", device="cpu"
        )

    assert mocks.Preprocessor.call_args.kwargs["use_layer"] == "counts"


# ---------------------------------------------------------------------------
# compute_scprint — gene-index alignment
#
# The failure these cover is silent: a misaligned index feeds every gene under the
# wrong embedding row, and the result is finite, non-zero and non-degenerate, so it
# passes every other guard in this file while describing noise.
# ---------------------------------------------------------------------------

# A gene in the middle of the checkpoint vocabulary. Withholding it from the
# ontology shifts every later gene by one, which is the real-world bug: 23 of
# medium-v1.5's IDs were retired between its training release and bionty's.
RETIRED = f"ENSG{100:011d}"


def test_drift_reports_first_misaligned_position():
    with mock_scprint(retired_genes=(RETIRED,)) as mocks:
        drift = _scprint_index_drift(mocks.model.genes, [SCPRINT_HUMAN])

    n_model, n_ontology, first_bad = drift[SCPRINT_HUMAN]
    assert (n_model, n_ontology) == (N_GENES, N_GENES - 1)
    assert first_bad == 100, "drift must start at the retired gene, not before it"


def test_drift_is_empty_when_ontology_matches_checkpoint():
    with mock_scprint() as mocks:
        drift = _scprint_index_drift(mocks.model.genes, [SCPRINT_HUMAN])

    assert drift[SCPRINT_HUMAN][2] is None


def test_repairs_ontology_and_then_embeds(raw_counts_adata):
    """The retired ID is registered, and only then is the embedding computed."""
    with mock_scprint(retired_genes=(RETIRED,)) as mocks:
        assert RETIRED not in mocks.ontology
        result = compute_scprint(raw_counts_adata, batch_key="batch", device="cpu")

        assert RETIRED in mocks.ontology, "retired ID must be registered"
        assert mocks.ontology[RETIRED] == SCPRINT_HUMAN
        assert (
            _scprint_index_drift(mocks.model.genes, [SCPRINT_HUMAN])[SCPRINT_HUMAN][2]
            is None
        )
    assert result is not None
    assert result.shape == (N_CELLS, EMB_DIM)


def test_refuses_to_embed_misaligned_when_repair_declined(raw_counts_adata):
    """align_ontology=False must raise, never embed against a shifted index."""
    with mock_scprint(retired_genes=(RETIRED,)) as mocks:
        with pytest.raises(RuntimeError, match="misaligned"):
            compute_scprint(
                raw_counts_adata,
                batch_key="batch",
                device="cpu",
                align_ontology=False,
            )
        mocks.Embedder.assert_not_called()
        assert RETIRED not in mocks.ontology, "must not write when declined"


def test_alignment_is_checked_before_gpu_work(raw_counts_adata):
    """
    The check must precede model.to(device).

    Alignment depends only on the ontology and the checkpoint's gene list, so
    failing after moving a multi-GB model to the GPU wastes an allocation and, in a
    batch of jobs, a scheduling slot per dataset.
    """
    with mock_scprint(retired_genes=(RETIRED,)) as mocks:
        with pytest.raises(RuntimeError, match="misaligned"):
            compute_scprint(
                raw_counts_adata,
                batch_key="batch",
                device="cpu",
                align_ontology=False,
            )
        mocks.model.to.assert_not_called()


def test_rejects_lognormed_input(lognormed_only_adata):
    """
    Log-normalised input must raise, never be embedded.

    scPRINT conditions on sequencing depth, so a normalised matrix produces a
    number that looks plausible and is wrong.
    """
    with mock_scprint():
        with pytest.raises(ValueError, match="requires raw counts"):
            compute_scprint(lognormed_only_adata, batch_key="batch")


# ---------------------------------------------------------------------------
# compute_scprint — organism
# ---------------------------------------------------------------------------


def test_organism_tag_written_to_obs(raw_counts_adata):
    captured = {}

    def capturing_embed(model, adata, cache=False):
        captured["organism"] = set(adata.obs["organism_ontology_term_id"])
        result = adata.copy()
        result.obsm["scprint"] = np.ones((adata.n_obs, EMB_DIM), dtype=np.float32)
        return result, {}

    with mock_scprint(embedder_side_effect=capturing_embed):
        compute_scprint(
            raw_counts_adata,
            batch_key="batch",
            organism=SCPRINT_MOUSE,
            device="cpu",
        )

    assert captured["organism"] == {SCPRINT_MOUSE}


def test_input_adata_not_mutated(raw_counts_adata):
    with mock_scprint():
        compute_scprint(raw_counts_adata, batch_key="batch", device="cpu")

    assert "organism_ontology_term_id" not in raw_counts_adata.obs.columns
    assert "scprint" not in raw_counts_adata.obsm


def test_raises_on_organism_unsupported_by_checkpoint(raw_counts_adata):
    with mock_scprint(organisms=(SCPRINT_HUMAN,)):
        with pytest.raises(ValueError, match="supports organisms"):
            compute_scprint(raw_counts_adata, batch_key="batch", organism=SCPRINT_MOUSE)


# ---------------------------------------------------------------------------
# Missing dependency
# ---------------------------------------------------------------------------


def test_returns_none_when_scprint_not_installed(raw_counts_adata):
    """Graceful None return when the optional dependency is absent."""
    keys = ("scprint", "scprint.tasks", "scdataloader")
    orig = {k: sys.modules.get(k) for k in keys}
    # Setting a key to None in sys.modules makes `import <key>` raise ImportError.
    sys.modules.update(dict.fromkeys(keys))
    try:
        result = compute_scprint(raw_counts_adata, batch_key="batch")
    finally:
        for k, v in orig.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    assert result is None


# ---------------------------------------------------------------------------
# run_baselines wiring
# ---------------------------------------------------------------------------


def test_run_baselines_stores_scprint_in_obsm(raw_counts_adata, tmp_path):
    """run_baselines dispatches to scPRINT and stores the result under obsm['scPRINT']."""
    h5ad_path = tmp_path / "test.h5ad"
    raw_counts_adata.write_h5ad(h5ad_path)

    with mock_scprint():
        run_baselines(h5ad_path, batch_key="batch", methods=["scPRINT"])

    result = ad.io.read_h5ad(h5ad_path)
    assert "scPRINT" in result.obsm
    assert result.obsm["scPRINT"].shape == (N_CELLS, EMB_DIM)


def test_run_baselines_threads_organism_through(raw_counts_adata, tmp_path):
    """--scprint-organism must reach compute_scprint, else mouse data is scored as human."""
    h5ad_path = tmp_path / "test.h5ad"
    raw_counts_adata.write_h5ad(h5ad_path)

    captured = {}

    def capturing_embed(model, adata, cache=False):
        captured["organism"] = set(adata.obs["organism_ontology_term_id"])
        result = adata.copy()
        result.obsm["scprint"] = np.ones((adata.n_obs, EMB_DIM), dtype=np.float32)
        return result, {}

    with mock_scprint(embedder_side_effect=capturing_embed):
        run_baselines(
            h5ad_path,
            batch_key="batch",
            methods=["scPRINT"],
            scprint_organism=SCPRINT_MOUSE,
        )

    assert captured["organism"] == {SCPRINT_MOUSE}
