"""
Cross-fitted baselines for predicting the expression level of expressed genes.

Each ``predict_*`` function takes a cells x genes CSC matrix ``x`` of positive
log1p(CP10K) values and a fold per cell, and returns one prediction per stored entry, in
the order of ``x.data``. Every prediction for a cell is made by a fit on the other folds.
Baselines are fitted on, and scored on, expressed entries only. Where a fit has too few
training cells the prediction falls back to the gene's training median.
"""

from collections.abc import Iterator

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd

# Singular values below this fraction of the largest are dropped. Few expressing cells
# can leave co-expressed genes nearly collinear, and the fit then extrapolates wildly.
RCOND = 1e-5


def check_expression_matrix(x) -> sp.csc_matrix:
    """Return ``x`` as CSC with sorted indices, rejecting non-positive stored values."""
    x = sp.csc_matrix(x)
    x.sum_duplicates()
    if x.nnz and not x.data.min() > 0:
        raise ValueError("Stored entries must be positive (expressed) values")
    return x


def donor_grouped_folds(
    dataset_ids, donor_ids, n_folds: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign folds so that all cells of a (dataset, donor) pair share one fold.

    Donor labels are only unique within a dataset, so the pair is the group. Groups are
    placed largest first into the fold with the fewest cells.

    Returns
    -------
        The fold and the group index of each cell.
    """
    frame = pd.DataFrame({"dataset": list(dataset_ids), "donor": list(donor_ids)})
    if frame.isna().any().any():
        raise ValueError("dataset and donor must be known for every cell")
    groups = frame.groupby(["dataset", "donor"], sort=True).ngroup().to_numpy()
    sizes = np.bincount(groups)
    if len(sizes) < n_folds:
        raise ValueError(f"{len(sizes)} donor groups cannot fill {n_folds} folds")
    fold_of_group = np.empty(len(sizes), dtype=np.int64)
    load = np.zeros(n_folds, dtype=np.int64)
    for group in np.argsort(-sizes, kind="stable"):
        fold = int(np.argmin(load))
        fold_of_group[group] = fold
        load[fold] += sizes[group]
    return fold_of_group[groups], groups


def predict_gene_mean(x: sp.csc_matrix, folds: np.ndarray) -> np.ndarray:
    """Mean of the gene over the training cells that express it."""
    pred = np.empty(x.nnz)
    for entries, y, _, train in _gene_folds(x, folds):
        pred[entries[~train]] = y[train].mean() if train.any() else np.nan
    return pred


def predict_gene_median(x: sp.csc_matrix, folds: np.ndarray) -> np.ndarray:
    """Median of the gene over the training cells that express it."""
    pred = np.empty(x.nnz)
    for entries, y, _, train in _gene_folds(x, folds):
        pred[entries[~train]] = np.median(y[train]) if train.any() else np.nan
    return pred


def predict_cell_type_mean(
    x: sp.csc_matrix, folds: np.ndarray, cell_types, min_train: int = 20
) -> np.ndarray:
    """Mean of the gene within the cell's type; the median for small types."""
    types, n_types = _codes(cell_types)
    pred = np.empty(x.nnz)
    for entries, y, cells, train in _gene_folds(x, folds):
        t = types[cells]
        count = np.bincount(t[train], minlength=n_types)
        total = np.bincount(t[train], weights=y[train], minlength=n_types)
        fallback = np.median(y[train]) if train.any() else np.nan
        means = np.where(count >= min_train, total / np.maximum(count, 1), fallback)
        pred[entries[~train]] = means[t[~train]]
    return pred


def predict_library_size_floor(x: sp.csc_matrix, library_size) -> np.ndarray:
    """Value of one molecule at the cell's library size. Nothing is fitted."""
    return np.log1p(1e4 / np.asarray(library_size, dtype=float)[x.indices])


def predict_library_size(
    x: sp.csc_matrix, folds: np.ndarray, library_size, min_train: int = 20
) -> np.ndarray:
    """Per-gene least squares on log library size."""
    log_size = np.log(np.asarray(library_size, dtype=float))
    pred = np.empty(x.nnz)
    for entries, y, cells, train in _gene_folds(x, folds):
        pred[entries[~train]] = _median_or(
            y, train, min_train, lambda: _line(log_size[cells], y, train)[~train]
        )
    return pred


def predict_cell_type_and_library_size(
    x: sp.csc_matrix,
    folds: np.ndarray,
    cell_types,
    library_size,
    min_type: int = 5,
    min_train: int = 20,
) -> np.ndarray:
    """Cell-type intercepts with one shared slope; small types use the pooled line."""
    types, n_types = _codes(cell_types)
    log_size = np.log(np.asarray(library_size, dtype=float))
    pred = np.empty(x.nnz)
    for entries, y, cells, train in _gene_folds(x, folds):
        z, t = log_size[cells], types[cells]

        def fit(z=z, t=t, y=y, train=train):
            count = np.bincount(t[train], minlength=n_types)
            n = np.maximum(count, 1)
            mean_z = np.bincount(t[train], weights=z[train], minlength=n_types) / n
            mean_y = np.bincount(t[train], weights=y[train], minlength=n_types) / n
            dz, dy = z[train] - mean_z[t[train]], y[train] - mean_y[t[train]]
            slope = _ratio((dz * dy).sum(), (dz * dz).sum())
            within = mean_y[t] + slope * (z - mean_z[t])
            return np.where(count[t] >= min_type, within, _line(z, y, train))[~train]

        pred[entries[~train]] = _median_or(y, train, min_train, fit)
    return pred


def select_coexpressed_genes(
    x: sp.csc_matrix, folds: np.ndarray, n_predictors: int = 50
) -> np.ndarray:
    """
    Genes with the highest absolute correlation with each gene, per fold.

    Correlations are over all training cells, zeros included. The gene x gene
    correlation matrix is dense.

    Returns
    -------
        Gene indices of shape (n_genes, n_folds, n_predictors).
    """
    n_genes = x.shape[1]
    if n_predictors >= n_genes:
        raise ValueError("n_predictors must be smaller than the number of genes")
    fold_ids = np.unique(folds)
    rows = x.tocsr()
    selected = np.empty((n_genes, len(fold_ids), n_predictors), dtype=np.int32)
    for i, fold in enumerate(fold_ids):
        train = rows[folds != fold]
        mean = np.asarray(train.mean(axis=0)).ravel()
        cov = _gram(train) / train.shape[0] - np.outer(mean, mean)
        std = np.sqrt(np.maximum(np.diag(cov), 0))
        scale = np.where(std > 0, 1 / np.where(std > 0, std, 1), 0)
        corr = np.abs(cov * scale[:, None] * scale[None, :])
        np.fill_diagonal(corr, -1)
        selected[:, i] = np.argpartition(-corr, n_predictors - 1, axis=1)[
            :, :n_predictors
        ]
    return selected


def predict_coexpression(
    x: sp.csc_matrix, folds: np.ndarray, predictors: np.ndarray, min_train: int = 50
) -> np.ndarray:
    """
    Least squares on co-expressed genes from :func:`select_coexpressed_genes`.

    Genes with too few training cells in any fold are not scored (NaN).
    """
    pred = np.full(x.nnz, np.nan)
    for entries, y, cells, fold_rows, values in _coexpression_genes(
        x, folds, predictors, min_train
    ):
        for i, test in fold_rows:
            design = np.column_stack([values(i), np.ones(len(cells))])
            beta = np.linalg.lstsq(design[~test], y[~test], rcond=RCOND)[0]
            pred[entries[test]] = design[test] @ beta
    return pred


def predict_cell_type_library_size_coexpression(
    x: sp.csc_matrix,
    folds: np.ndarray,
    cell_types,
    library_size,
    predictors: np.ndarray,
    min_type: int = 5,
    min_train: int = 50,
) -> np.ndarray:
    """
    Cell-type intercepts plus library size and co-expressed genes.

    Types with fewer than ``min_type`` training cells share the grand mean as intercept.
    Genes with too few training cells in any fold are not scored (NaN).
    """
    types, n_types = _codes(cell_types)
    log_size = np.log(np.asarray(library_size, dtype=float))
    pred = np.full(x.nnz, np.nan)
    for entries, y, cells, fold_rows, values in _coexpression_genes(
        x, folds, predictors, min_train
    ):
        t = types[cells]
        for i, test in fold_rows:
            train = ~test
            design = np.column_stack([values(i), log_size[cells]])
            count = np.bincount(t[train], minlength=n_types)
            large = count >= min_type
            to_mean = sp.csr_matrix(
                (np.ones(train.sum()), (t[train], np.arange(train.sum()))),
                shape=(n_types, train.sum()),
            )
            n = np.maximum(count, 1)[:, None]
            mean_x = np.where(
                large[:, None], to_mean @ design[train] / n, design[train].mean(axis=0)
            )
            mean_y = np.where(large, to_mean @ y[train] / n[:, 0], y[train].mean())
            beta = np.linalg.lstsq(
                design[train] - mean_x[t[train]],
                y[train] - mean_y[t[train]],
                rcond=RCOND,
            )[0]
            pred[entries[test]] = (
                mean_y[t[test]] + (design[test] - mean_x[t[test]]) @ beta
            )
    return pred


def predict_pca(
    x: sp.csc_matrix,
    folds: np.ndarray,
    rank: int = 512,
    min_train: int = 50,
    seed: int = 42,
) -> np.ndarray:
    """
    Low-rank reconstruction from the whole cell, with a per-gene affine readout.

    Stored values are centred on the gene's training mean and other entries are zero, so
    rank 0 is the gene mean. The readout is fitted on the reconstructions of the cells in
    the other folds.
    """
    cell_of = x.indices
    gene_of = np.repeat(np.arange(x.shape[1]), np.diff(x.indptr))
    recon = np.empty(x.nnz)
    for fold in np.unique(folds):
        test_entry = folds[cell_of] == fold
        mean = _expressed_means(x, ~test_entry)
        z = sp.csc_matrix((x.data - mean[gene_of], x.indices, x.indptr), shape=x.shape)
        z = z.tocsr()
        test_cells = np.flatnonzero(folds == fold)
        _, _, vt = randomized_svd(
            z[folds != fold], n_components=rank, n_iter=4, random_state=seed
        )
        low_rank = np.zeros((x.shape[0], rank))
        low_rank[test_cells] = z[test_cells] @ vt.T
        for rows in _chunks(np.flatnonzero(test_entry)):
            genes = gene_of[rows]
            recon[rows] = mean[genes] + np.einsum(
                "ij,ij->i", low_rank[cell_of[rows]], vt.T[genes]
            )
    pred = np.empty(x.nnz)
    for entries, y, _, train in _gene_folds(x, folds):
        pred[entries[~train]] = _median_or(
            y, train, min_train, lambda: _line(recon[entries], y, train)[~train]
        )
    return pred


def predict_nearest_cells(
    x: sp.csc_matrix,
    folds: np.ndarray,
    n_neighbors: int = 50,
    n_components: int = 50,
    min_neighbors: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """
    Mean over the nearest training cells that express the gene.

    Neighbours are by cosine similarity in a PCA fitted on the training cells, which are
    held dense in memory.
    """
    rows = x.tocsr()
    expressed = rows.copy()
    expressed.data = np.ones_like(expressed.data)
    pred = predict_gene_median(x, folds)
    gene_of = np.repeat(np.arange(x.shape[1]), np.diff(x.indptr))
    for fold in np.unique(folds):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        dense = rows[train].toarray()
        mean = dense.mean(axis=0)
        _, _, vt = randomized_svd(
            dense - mean, n_components=n_components, n_iter=4, random_state=seed
        )
        train_pcs = _unit(rows[train] @ vt.T - mean @ vt.T)
        test_pcs = _unit(rows[test] @ vt.T - mean @ vt.T)
        del dense
        for part in _chunks(np.arange(len(test)), 4096):
            similarity = test_pcs[part] @ train_pcs.T
            near = np.argpartition(-similarity, n_neighbors - 1, axis=1)
            near = train[near[:, :n_neighbors]]
            link = sp.csr_matrix(
                (
                    np.ones(near.size),
                    near.ravel(),
                    np.arange(len(part) + 1) * n_neighbors,
                ),
                shape=(len(part), x.shape[0]),
            )
            totals, counts = (link @ rows).toarray(), (link @ expressed).toarray()
            local = np.full(x.shape[0], -1)
            local[test[part]] = np.arange(len(part))
            entry = np.flatnonzero(local[x.indices] >= 0)
            at = local[x.indices[entry]], gene_of[entry]
            enough = counts[at] >= min_neighbors
            at = at[0][enough], at[1][enough]
            pred[entry[enough]] = totals[at] / counts[at]
    return pred


def compare_to_baseline(
    x: sp.csc_matrix,
    model_pred: np.ndarray,
    baseline_pred: np.ndarray,
    groups: np.ndarray,
    entry_mask: np.ndarray | None = None,
    min_cells: int = 50,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare model error with baseline error under absolute and squared error.

    Args:
    ----
        x: cells x genes matrix the predictions are aligned to
        model_pred: model prediction per stored entry of ``x``
        baseline_pred: baseline prediction per stored entry; genes with a NaN prediction
            are left out
        groups: donor group of each cell, resampled for the confidence intervals
        entry_mask: score only these entries. ``min_cells`` still counts all of them
        min_cells: least number of expressing cells for a gene to be scored
        n_boot: number of bootstrap resamples of the groups
        seed: bootstrap seed

    Returns:
    -------
        A table per (gene, loss) and a summary per loss. A ratio below 1 means the model
        has the lower error. ``pooled_ratio`` weights every entry equally.
    """
    n_genes, n_groups = x.shape[1], int(groups.max()) + 1
    gene_of = np.repeat(np.arange(n_genes), np.diff(x.indptr))
    keep = np.ones(x.nnz, dtype=bool) if entry_mask is None else entry_mask
    scored = np.diff(x.indptr) >= min_cells
    scored &= np.bincount(gene_of[np.isnan(baseline_pred)], minlength=n_genes) == 0
    scored &= np.bincount(gene_of[keep], minlength=n_genes) > 0
    cell = gene_of[keep] * n_groups + groups[x.indices[keep]]

    def totals(pred, power):
        error = np.abs(x.data[keep] - pred[keep]) ** power
        total = np.bincount(cell, weights=error, minlength=n_genes * n_groups)
        return total.reshape(n_genes, n_groups)[scored]

    weights = (
        np.random.default_rng(seed)
        .multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=n_boot)
        .T
    )
    per_gene, summary = [], []
    for loss, power in (("mae", 1), ("mse", 2)):
        model, base = totals(model_pred, power), totals(baseline_pred, power)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = model.sum(axis=1) / base.sum(axis=1)
        boot = (model.sum(axis=0) @ weights) / (base.sum(axis=0) @ weights)
        low, high = np.quantile(boot, [0.025, 0.975])
        genes = np.flatnonzero(scored)
        per_gene.append(pd.DataFrame({"gene_id": genes, "loss": loss, "ratio": ratio}))
        summary.append(
            {
                "loss": loss,
                "n_genes": len(genes),
                "median_ratio": np.median(ratio),
                "frac_genes_better": (ratio < 1).mean(),
                "pooled_ratio": model.sum() / base.sum(),
                "pooled_ci_low": low,
                "pooled_ci_high": high,
            }
        )
    return pd.concat(per_gene, ignore_index=True), pd.DataFrame(summary)


def _gene_folds(x: sp.csc_matrix, folds: np.ndarray) -> Iterator[tuple]:
    """Yield (entry indices, values, cells, training mask) per gene and held-out fold."""
    for start, stop in zip(x.indptr[:-1], x.indptr[1:]):
        if start == stop:
            continue
        cells = x.indices[start:stop]
        y = x.data[start:stop].astype(float)
        for fold in np.unique(folds[cells]):
            yield np.arange(start, stop), y, cells, folds[cells] != fold


def _coexpression_genes(x, folds, predictors, min_train) -> Iterator[tuple]:
    fold_ids = np.unique(folds)
    rows = x.tocsr()
    for gene, (start, stop) in enumerate(zip(x.indptr[:-1], x.indptr[1:])):
        cells = x.indices[start:stop]
        tests = [(i, folds[cells] == fold) for i, fold in enumerate(fold_ids)]
        if min(int((~test).sum()) for _, test in tests) < min_train:
            continue
        union, inverse = np.unique(predictors[gene], return_inverse=True)
        columns = inverse.reshape(predictors[gene].shape)
        block = rows[cells][:, union].toarray().astype(float)
        yield (
            np.arange(start, stop),
            x.data[start:stop].astype(float),
            cells,
            [(i, test) for i, test in tests if test.any()],
            lambda i, block=block, columns=columns: block[:, columns[i]],
        )


def _gram(rows: sp.csr_matrix, size: int = 4096) -> np.ndarray:
    """Dense ``rows.T @ rows``; chunked BLAS is faster than the sparse product."""
    total = np.zeros((rows.shape[1], rows.shape[1]))
    for part in _chunks(np.arange(rows.shape[0]), size):
        dense = rows[part].toarray()
        total += dense.T @ dense
    return total


def _line(z: np.ndarray, y: np.ndarray, train: np.ndarray) -> np.ndarray:
    dz, dy = z[train] - z[train].mean(), y[train] - y[train].mean()
    slope = _ratio((dz * dy).sum(), (dz * dz).sum())
    return y[train].mean() + slope * (z - z[train].mean())


def _median_or(y, train, min_train, fit):
    if train.sum() >= max(min_train, 1):
        return fit()
    return np.median(y[train]) if train.any() else np.nan


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 1e-12 else 0.0


def _expressed_means(x: sp.csc_matrix, entries: np.ndarray) -> np.ndarray:
    gene_of = np.repeat(np.arange(x.shape[1]), np.diff(x.indptr))[entries]
    total = np.bincount(gene_of, weights=x.data[entries], minlength=x.shape[1])
    count = np.bincount(gene_of, minlength=x.shape[1])
    return total / np.maximum(count, 1)


def _codes(labels) -> tuple[np.ndarray, int]:
    codes, names = pd.factorize(np.asarray(labels))
    if (codes < 0).any():
        raise ValueError("Every cell needs a label")
    return codes, len(names)


def _unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def _chunks(index: np.ndarray, size: int = 100_000) -> Iterator[np.ndarray]:
    for start in range(0, len(index), size):
        yield index[start : start + size]
