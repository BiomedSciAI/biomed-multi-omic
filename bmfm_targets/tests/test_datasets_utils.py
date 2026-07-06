"""Tests for bmfm_targets.datasets.datasets_utils.make_group_means."""
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as ss

from bmfm_targets.datasets.datasets_utils import make_group_means


def _make_synthetic_adata() -> sc.AnnData:
    """
    Build a tiny synthetic AnnData for make_group_means tests.

    6 cells x 4 genes.
    obs column "cond": values A (2 cells), B (2 cells), Control (2 cells).
    obs column "split": "train" for first 4 cells, "test" for last 2.
    """
    rng = np.random.default_rng(42)
    X = rng.integers(0, 10, size=(6, 4)).astype(float)
    obs = pd.DataFrame(
        {
            "cond": ["A", "A", "B", "B", "Control", "Control"],
            "split": ["train", "train", "train", "train", "test", "test"],
        }
    )
    var = pd.DataFrame(index=[f"gene{i}" for i in range(4)])
    return sc.AnnData(X=ss.csr_matrix(X), obs=obs, var=var)


# ---------------------------------------------------------------------------
# Backward-compatibility guard (must be GREEN on unmodified code)
# ---------------------------------------------------------------------------


def test_make_group_means_default_has_synthetic_row():
    """Default call includes 'Average_Perturbation_Train'; row count == n_unique + 1."""
    ad = _make_synthetic_adata()
    result = make_group_means(ad, "cond", "split")
    n_unique = ad.obs["cond"].nunique()
    assert "Average_Perturbation_Train" in result.obs_names
    assert result.n_obs == n_unique + 1


# ---------------------------------------------------------------------------
# RED tests (expected to fail on unmodified code)
# ---------------------------------------------------------------------------


def test_make_group_means_avg_row_label_none_skips_synthetic_row():
    """avg_row_label=None -> no synthetic row; row count == n_unique_conds."""
    ad = _make_synthetic_adata()
    result = make_group_means(ad, "cond", "split", avg_row_label=None)
    n_unique = ad.obs["cond"].nunique()
    assert "Average_Perturbation_Train" not in result.obs_names
    assert result.n_obs == n_unique


def test_make_group_means_none_avg_allows_missing_split():
    """avg_row_label=None, split_column_name=None -> returns without error."""
    ad = _make_synthetic_adata()
    # Should not raise even though split_column_name is absent
    result = make_group_means(ad, "cond", split_column_name=None, avg_row_label=None)
    assert result is not None
    n_unique = ad.obs["cond"].nunique()
    assert result.n_obs == n_unique


def test_make_group_means_requires_split_when_avg_set():
    """avg_row_label='X', split_column_name=None -> raises ValueError."""
    ad = _make_synthetic_adata()
    with pytest.raises(ValueError, match="split_column_name required"):
        make_group_means(ad, "cond", split_column_name=None, avg_row_label="X")


# ---------------------------------------------------------------------------
# Additional: custom avg_row_label string is used in the output
# ---------------------------------------------------------------------------


def test_make_group_means_custom_avg_row_label():
    """Custom avg_row_label string appears in result index instead of default."""
    ad = _make_synthetic_adata()
    result = make_group_means(ad, "cond", "split", avg_row_label="MyCustomAvg")
    assert "MyCustomAvg" in result.obs_names
    assert "Average_Perturbation_Train" not in result.obs_names
