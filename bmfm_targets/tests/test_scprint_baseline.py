"""
Tests for the scPRINT baseline in bmfm_targets.evaluation.baselines.

scPRINT is an optional dependency, so all tests mock the scprint/scdataloader
packages via sys.modules injection.  They verify the integration contract —
counts validation, gene-identifier preprocessing, device placement, canonical
Embedder settings, and obsm storage — without requiring a real checkpoint or GPU.

The mocks deliberately reproduce scPRINT's real behaviour: ``Embedder`` writes
the embedding to ``obsm["scprint"]`` (see ``scprint.tasks.cell_emb``), not to
``obsm["scprint_emb"]``.  A test suite that invents its own key cannot catch a
key mismatch in the code under test, so the key used here matters.
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
    compute_scprint,
    run_baselines,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_CELLS = 20
N_GENES = 200
EMB_DIM = 512

# The key scPRINT itself writes.  Do not change without checking cell_emb.py.
SCPRINT_OBSM_KEY = "scprint"

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
):
    """
    Inject mock scprint/scdataloader modules into sys.modules.

    Yields a namespace with the mocked ``scPrint`` class, ``Embedder`` class,
    ``Preprocessor`` class and the model instance, so tests can inspect calls.
    """
    fake_emb = (
        np.random.default_rng(42).standard_normal((n_cells, emb_dim)).astype(np.float32)
    )

    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    mock_model.organisms = list(organisms)
    mock_model.genes = [f"ENSG{i:011d}" for i in range(N_GENES)]

    mock_scprint_cls = MagicMock()
    mock_scprint_cls.load_from_checkpoint.return_value = mock_model

    # Preprocessor is a callable object: Preprocessor(**kwargs)(adata) -> adata.
    mock_preprocessor_cls = MagicMock()
    mock_preprocessor_cls.return_value = MagicMock(side_effect=lambda adata: adata)

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

    injected = {
        "scprint": mock_scprint_mod,
        "scprint.tasks": mock_tasks_mod,
        "scdataloader": mock_scdataloader_mod,
    }
    orig = {k: sys.modules.get(k) for k in injected}
    sys.modules.update(injected)
    try:
        yield MagicMock(
            scPrint=mock_scprint_cls,
            Embedder=mock_embedder_cls,
            Preprocessor=mock_preprocessor_cls,
            model=mock_model,
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


def test_reads_the_obsm_key_scprint_actually_writes(raw_counts_adata):
    """Regression: the code once read obsm['scprint_emb'], which never exists."""

    def embed_writing_real_key(model, adata, cache=False):
        result = adata.copy()
        result.obsm["scprint"] = np.ones((adata.n_obs, EMB_DIM), dtype=np.float32)
        return result, {}

    with mock_scprint(embedder_side_effect=embed_writing_real_key):
        result = compute_scprint(raw_counts_adata, batch_key="batch")

    assert result is not None, "must read obsm['scprint'], the key scPRINT writes"
    assert result.shape == (N_CELLS, EMB_DIM)


def test_raises_when_embedding_key_absent(raw_counts_adata):
    """A missing embedding is a hard failure, not a silent skip."""

    def embed_writing_nothing(model, adata, cache=False):
        return adata.copy(), {}

    with mock_scprint(embedder_side_effect=embed_writing_nothing):
        with pytest.raises(RuntimeError, match="no 'scprint' embedding"):
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
