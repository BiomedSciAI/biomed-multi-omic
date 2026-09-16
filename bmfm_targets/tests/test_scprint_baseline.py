"""
Tests for the scPRINT baseline in bmfm_targets.evaluation.baselines.

scPRINT is an optional dependency, so all tests mock the scprint package via
sys.modules injection.  They verify the integration contract — correct data
prep, Embedder call signature, and obsm storage — without requiring a real
checkpoint or GPU.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from bmfm_targets.evaluation.baselines import (
    SCPRINT_DEFAULT_CHECKPOINT,
    compute_scprint,
    run_baselines,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_CELLS = 20
N_GENES = 200
EMB_DIM = 512

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def raw_counts_adata():
    """AnnData with raw integer counts in X and CELLxGENE organism metadata."""
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32))
    obs = pd.DataFrame(
        {
            "batch": pd.Categorical(["batch_A"] * 10 + ["batch_B"] * 10),
            "organism_ontology_term_id": ["NCBITaxon:9606"] * N_CELLS,
        },
        index=[f"cell_{i}" for i in range(N_CELLS)],
    )
    var = pd.DataFrame(index=[f"GENE{i}" for i in range(N_GENES)])
    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture()
def normalised_adata_with_counts_layer(raw_counts_adata):
    """AnnData where X is log-normalised but raw counts live in layers['counts']."""
    adata = raw_counts_adata.copy()
    adata.layers["counts"] = adata.X.copy()
    normed = (
        adata.X.toarray() / (adata.X.toarray().sum(axis=1, keepdims=True) + 1e-8) * 1e4
    )
    adata.X = sp.csr_matrix((normed + 1e-3).astype(np.float32))
    return adata


@pytest.fixture()
def adata_without_organism(raw_counts_adata):
    """AnnData missing the organism_ontology_term_id column."""
    adata = raw_counts_adata.copy()
    del adata.obs["organism_ontology_term_id"]
    return adata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def mock_scprint(embedder_side_effect=None, n_cells=N_CELLS, emb_dim=EMB_DIM):
    """
    Context manager that injects a mock scprint into sys.modules.

    Yields (mock_scPrint_cls, mock_Embedder_cls) so tests can inspect calls.
    """
    import sys

    fake_emb = (
        np.random.default_rng(42).standard_normal((n_cells, emb_dim)).astype(np.float32)
    )

    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model

    mock_scPrint_cls = MagicMock()
    mock_scPrint_cls.load_from_checkpoint.return_value = mock_model

    if embedder_side_effect is None:

        def _default_embed(model, adata, cache=False):
            result = adata.copy()
            result.obsm["scprint_emb"] = fake_emb
            return result, {}

        embedder_side_effect = _default_embed

    mock_Embedder_cls = MagicMock(
        return_value=MagicMock(side_effect=embedder_side_effect)
    )

    mock_scprint_mod = MagicMock()
    mock_scprint_mod.scPrint = mock_scPrint_cls
    mock_scprint_tasks_mod = MagicMock()
    mock_scprint_tasks_mod.Embedder = mock_Embedder_cls

    orig = {k: sys.modules.get(k) for k in ("scprint", "scprint.tasks")}
    sys.modules["scprint"] = mock_scprint_mod
    sys.modules["scprint.tasks"] = mock_scprint_tasks_mod
    try:
        yield mock_scPrint_cls, mock_Embedder_cls
    finally:
        for k, v in orig.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_compute_scprint_returns_correct_shape(raw_counts_adata):
    with mock_scprint():
        result = compute_scprint(raw_counts_adata, batch_key="batch")

    assert result is not None
    assert result.shape == (N_CELLS, EMB_DIM)
    assert result.dtype == np.float32


def test_embedder_instantiated_with_deterministic_and_minimal_settings(
    raw_counts_adata,
):
    """Embedder must use 'most var' (deterministic) and disable plot/classify/expression output."""
    with mock_scprint() as (_, mock_Embedder_cls):
        compute_scprint(raw_counts_adata, batch_key="batch")

    kwargs = mock_Embedder_cls.call_args.kwargs
    assert kwargs["how"] == "most var", "gene selection must be deterministic"
    assert kwargs["doplot"] is False, "plotting must be disabled"
    assert kwargs["doclass"] is False, "classifier heads must be disabled"
    assert kwargs["output_expression"] == "none"


def test_model_loaded_from_checkpoint(raw_counts_adata):
    with mock_scprint() as (mock_scPrint_cls, _):
        compute_scprint(raw_counts_adata, batch_key="batch", ckpt_path="my.ckpt")

    mock_scPrint_cls.load_from_checkpoint.assert_called_once_with(
        "my.ckpt", precpt_gene_emb=None
    )


def test_uses_counts_layer_when_x_is_normalised(normalised_adata_with_counts_layer):
    """When X looks normalised, the function should swap in layers['counts'] before embedding."""
    captured = {}

    def capturing_embed(model, adata, cache=False):
        arr = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        captured["x_max"] = arr.max()
        result = adata.copy()
        result.obsm["scprint_emb"] = np.zeros((adata.n_obs, EMB_DIM), dtype=np.float32)
        return result, {}

    with mock_scprint(embedder_side_effect=capturing_embed):
        compute_scprint(normalised_adata_with_counts_layer, batch_key="batch")

    assert "x_max" in captured, "embedder was never called"
    assert (
        captured["x_max"] > 1.0
    ), "Model should receive raw counts (max > 1); got fractional normalised values"


def test_organism_tag_added_when_missing(adata_without_organism):
    """Missing organism_ontology_term_id must be auto-filled before the embedder is called."""
    captured = {}

    def capturing_embed(model, adata, cache=False):
        captured["cols"] = list(adata.obs.columns)
        result = adata.copy()
        result.obsm["scprint_emb"] = np.zeros((adata.n_obs, EMB_DIM), dtype=np.float32)
        return result, {}

    with mock_scprint(embedder_side_effect=capturing_embed):
        result = compute_scprint(adata_without_organism, batch_key="batch")

    assert result is not None
    assert "organism_ontology_term_id" in captured["cols"]


def test_returns_none_when_scprint_not_installed(raw_counts_adata):
    """Graceful None return when the scprint package is absent."""
    import sys

    # Setting a key to None in sys.modules causes ImportError on `import scprint`.
    orig_scprint = sys.modules.pop("scprint", None)
    orig_tasks = sys.modules.pop("scprint.tasks", None)
    sys.modules["scprint"] = None  # triggers ImportError
    sys.modules["scprint.tasks"] = None

    try:
        result = compute_scprint(raw_counts_adata, batch_key="batch")
    finally:
        if orig_scprint is None:
            sys.modules.pop("scprint", None)
        else:
            sys.modules["scprint"] = orig_scprint
        if orig_tasks is None:
            sys.modules.pop("scprint.tasks", None)
        else:
            sys.modules["scprint.tasks"] = orig_tasks

    assert result is None


def test_default_checkpoint_constant():
    assert SCPRINT_DEFAULT_CHECKPOINT == "medium-v1.5.ckpt"


def test_run_baselines_stores_scprint_in_obsm(raw_counts_adata, tmp_path):
    """run_baselines dispatches to scPRINT and stores the result under obsm['scPRINT']."""
    h5ad_path = tmp_path / "test.h5ad"
    raw_counts_adata.write_h5ad(h5ad_path)

    with mock_scprint():
        run_baselines(h5ad_path, batch_key="batch", methods=["scPRINT"])

    result = ad.io.read_h5ad(h5ad_path)
    assert "scPRINT" in result.obsm
    assert result.obsm["scPRINT"].shape == (N_CELLS, EMB_DIM)
