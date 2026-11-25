import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression

from bmfm_targets.evaluation import expression_baselines as eb


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n_cells, n_genes = 300, 12
    types = rng.integers(0, 3, n_cells)
    size = rng.integers(2000, 20000, n_cells)
    dense = rng.gamma(4, 0.5, (n_cells, n_genes)) + types[:, None] * 0.5
    dense[:, 1] = dense[:, 0] * 0.8 + rng.normal(0, 0.02, n_cells)
    dense[rng.random(dense.shape) < 0.2] = 0
    x = eb.check_expression_matrix(dense)
    return x, rng.integers(0, 4, n_cells), types, size


def oracle(x, folds, design):
    """Plain sklearn OLS per gene and fold on an arbitrary cell-level design."""
    pred = np.empty(x.nnz)
    for g in range(x.shape[1]):
        entries = np.arange(x.indptr[g], x.indptr[g + 1])
        cells, y, d = x.indices[entries], x.data[entries], design(g)
        for f in np.unique(folds):
            te = folds[cells] == f
            fit = LinearRegression().fit(d(f)[cells[~te]], y[~te])
            pred[entries[te]] = fit.predict(d(f)[cells[te]])
    return pred


def test_gene_mean_and_median_known_case():
    x = eb.check_expression_matrix([[1.0], [2.0], [6.0], [9.0]])
    folds = np.array([0, 1, 1, 1])
    assert eb.predict_gene_mean(x, folds).tolist() == pytest.approx([17 / 3, 1, 1, 1])
    assert eb.predict_gene_median(x, folds).tolist() == [6, 1, 1, 1]


def test_held_out_values_do_not_affect_their_prediction(data):
    x, folds, types, size = data
    changed = x.copy()
    held = folds[x.indices] == 0
    changed.data[held] += 5
    for predict in (
        lambda m: eb.predict_gene_mean(m, folds),
        lambda m: eb.predict_cell_type_mean(m, folds, types),
        lambda m: eb.predict_library_size(m, folds, size),
        lambda m: eb.predict_cell_type_and_library_size(m, folds, types, size),
    ):
        np.testing.assert_array_equal(predict(x)[held], predict(changed)[held])


def test_cell_type_mean_matches_one_hot_regression(data):
    x, folds, types, _ = data
    one_hot = np.eye(3)[types]
    np.testing.assert_allclose(
        eb.predict_cell_type_mean(x, folds, types),
        oracle(x, folds, lambda g: lambda f: one_hot),
    )


def test_library_size_matches_sklearn(data):
    x, folds, types, size = data
    z = np.log(size)[:, None]
    np.testing.assert_allclose(
        eb.predict_library_size(x, folds, size),
        oracle(x, folds, lambda g: lambda f: z),
    )
    both = np.hstack([z, np.eye(3)[types]])
    np.testing.assert_allclose(
        eb.predict_cell_type_and_library_size(x, folds, types, size),
        oracle(x, folds, lambda g: lambda f: both),
    )


def test_coexpression_matches_sklearn(data):
    x, folds, types, size = data
    chosen = eb.select_coexpressed_genes(x, folds, 3)
    dense, z = x.toarray(), np.log(size)[:, None]
    np.testing.assert_allclose(
        eb.predict_coexpression(x, folds, chosen),
        oracle(x, folds, lambda g: lambda f: dense[:, chosen[g, f]]),
    )
    extra = np.hstack([z, np.eye(3)[types]])
    np.testing.assert_allclose(
        eb.predict_cell_type_library_size_coexpression(x, folds, types, size, chosen),
        oracle(
            x, folds, lambda g: lambda f: np.hstack([dense[:, chosen[g, f]], extra])
        ),
    )


def test_selection_matches_numpy_correlation(data):
    x, folds, _, _ = data
    chosen = eb.select_coexpressed_genes(x, folds, 3)
    corr = np.abs(np.corrcoef(x.toarray()[folds != 2].T))
    np.fill_diagonal(corr, -1)
    for g in range(x.shape[1]):
        assert set(chosen[g, 2]) == set(np.argsort(-corr[g])[:3])


def test_library_size_floor_is_one_molecule():
    counts = np.array([[1, 3], [1, 7]])
    size = counts.sum(axis=1)
    x = eb.check_expression_matrix(np.log1p(counts / size[:, None] * 1e4))
    pred = eb.predict_library_size_floor(x, size)
    np.testing.assert_allclose(pred[:2], x.data[:2])
    assert (pred[2:] < x.data[2:]).all()


def test_pca_recovers_low_rank_data():
    rng = np.random.default_rng(1)
    dense = 5 + rng.normal(size=(200, 2)) @ rng.normal(size=(2, 15)) * 0.3
    x, folds = eb.check_expression_matrix(dense), np.arange(200) % 4
    err = np.abs(eb.predict_pca(x, folds, rank=2) - x.data).mean()
    assert err < 1e-3 < np.abs(eb.predict_gene_mean(x, folds) - x.data).mean()


def test_nearest_cells_finds_the_duplicates():
    rng = np.random.default_rng(2)
    base = rng.gamma(3, 1, (4, 20)) + 0.1
    x = eb.check_expression_matrix(np.repeat(base, 40, axis=0))
    folds = np.arange(160) % 4
    pred = eb.predict_nearest_cells(x, folds, n_neighbors=10, n_components=3)
    np.testing.assert_allclose(pred, x.data, rtol=1e-6)


def test_fallback_is_the_training_median(data):
    x, folds, types, size = data
    median = eb.predict_gene_median(x, folds)
    for pred in (
        eb.predict_cell_type_mean(x, folds, types, min_train=10**6),
        eb.predict_library_size(x, folds, size, min_train=10**6),
        eb.predict_pca(x, folds, rank=2, min_train=10**6),
    ):
        np.testing.assert_array_equal(pred, median)
    chosen = eb.select_coexpressed_genes(x, folds, 3)
    assert np.isnan(eb.predict_coexpression(x, folds, chosen, min_train=10**6)).all()


def test_folds_keep_donor_pairs_together_and_balanced():
    rng = np.random.default_rng(3)
    dataset, donor = rng.integers(0, 4, 2000), rng.integers(0, 25, 2000)
    folds, groups = eb.donor_grouped_folds(dataset, donor)
    frame = pd.DataFrame({"d": dataset, "o": donor, "f": folds, "g": groups})
    assert len(frame.drop_duplicates(["d", "o", "f"])) == frame.g.nunique()
    assert frame.g.nunique() == frame.groupby(["d", "o"]).ngroups == 100
    assert np.ptp(np.bincount(folds)) < 30


def test_compare_known_ratio_and_bootstrap():
    rng = np.random.default_rng(4)
    x = eb.check_expression_matrix(rng.gamma(3, 1, (200, 6)) + 0.1)
    groups = np.arange(200) % 20
    noise = rng.normal(size=x.nnz)
    per_gene, summary = eb.compare_to_baseline(
        x, x.data + noise / 2, x.data + noise, groups, n_boot=50
    )
    summary = summary.set_index("loss")
    for loss, expected in (("mae", 0.5), ("mse", 0.25)):
        np.testing.assert_allclose(per_gene.ratio[per_gene.loss == loss], expected)
        row = summary.loc[loss]
        assert row.pooled_ci_low == pytest.approx(expected)
        assert row.pooled_ci_high == pytest.approx(expected)
        assert row.frac_genes_better == 1


def test_compare_skips_unscored_genes():
    x = eb.check_expression_matrix(np.ones((60, 3)))
    base = np.zeros(x.nnz)
    base[60:120] = np.nan
    mask = np.ones(x.nnz, dtype=bool)
    mask[120:] = False
    per_gene, summary = eb.compare_to_baseline(
        x, np.full(x.nnz, 0.5), base, np.arange(60) % 6, entry_mask=mask, n_boot=10
    )
    assert per_gene.gene_id.unique().tolist() == [0]
    assert (summary.n_genes == 1).all()


def test_stored_zero_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        eb.check_expression_matrix(sp.csc_matrix(([0.0], ([0], [0])), shape=(2, 2)))
