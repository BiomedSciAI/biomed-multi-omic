import logging
import random
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import stats
from scipy.sparse import random as sparse_random

from bmfm_targets.tests import helpers
from bmfm_targets.transforms import sc_transforms

logger = logging.getLogger(__name__)


def get_random_anndata() -> AnnData:
    np.random.seed(0)
    counts = sparse_random(
        30,
        20,
        density=0.6,
        format="csr",
        data_rvs=stats.poisson(25, loc=10).rvs,
        dtype=int,
    )
    adata = AnnData(counts)
    adata.obs_names = [f"Cell_{i:d}" for i in range(adata.n_obs)]
    adata.var_names = [f"Gene_{i:d}" for i in range(adata.n_vars)]
    return adata


def get_random_anndata_with_gene_names() -> AnnData:
    adata = get_random_anndata()
    adata.var_names = [
        "DDX43",
        "NAA40",
        "ZBTB45P1",
        "MIA2",
        "ASPN",
        "MIR661",
        "CUL5",
        "GPANK1",
        "RPL39P19",
        "TRY2P",
        "HPYR1",
        "ADCK5",
        "PAX7",
        "TAS2R30",
        "ARL4AP2",
        "RPS12P29",
        "WNK2",
        "UQCRHP2",
        "TSSK2",
        "TAS2R30",
    ]
    return adata


def test_filter_cells_transform():
    # Apply the filter cells transformation
    adata = get_random_anndata()
    min_genes = 10
    transform = sc_transforms.FilterCellsTransform(min_genes=min_genes)
    transformed_data = transform(adata)["adata"]
    assert transformed_data.obs.min().squeeze() >= min_genes


def test_filter_genes_transform():
    adata = get_random_anndata()
    min_cells = 5
    transform = sc_transforms.FilterGenesTransform(min_cells=min_cells)
    transformed_data = transform(adata)["adata"]
    assert (transformed_data.X.todense() > 0).sum(axis=1).min() >= 5


def test_median_transform():
    adata = get_random_anndata()
    median_dict = {var_name: random.randint(1, 10) for var_name in adata.var_names}
    transform = sc_transforms.MedianNormalizeTransform(median_dict=median_dict)
    transformed_data = transform(adata)["adata"]
    assert (transformed_data.X.todense() > 0).sum(axis=1).min() > 0


def test_norm_transform():
    adata = get_random_anndata()
    transform = sc_transforms.NormalizeTotalTransform(target_sum=10)
    transformed_data = transform(adata)["adata"]
    assert transformed_data is not None


@pytest.mark.skip(reason="sps integration fails, requires additional review")
def test_TPM_norm_transform():
    adata = get_random_anndata_with_gene_names()
    transform = sc_transforms.TPMNormalizationTransform(
        gtf_file=helpers.TEST_GTF_PATH,
        filter_genes_min_value=1,
        filter_genes_min_percentage=0.25,
    )
    transformed_data = transform(adata)["adata"]
    assert transformed_data is not None
    assert len(transformed_data.var_names) < len(adata.var_names)


def test_scale_counts_transform():
    adata = get_random_anndata()
    adata_pre = adata.copy()
    scale_factor = 10
    transform = sc_transforms.ScaleCountsTransform(scale_factor=scale_factor)
    transformed_data = transform(adata)["adata"]
    np.testing.assert_almost_equal(
        adata_pre.X.data * scale_factor, transformed_data.X.data
    )


def test_log_transform():
    adata = get_random_anndata()
    pre_transform = sc_transforms.NormalizeTotalTransform(target_sum=10)
    transform = sc_transforms.LogTransform()
    transformed_data = pre_transform(adata)["adata"]
    pre_log_values = np.array(transformed_data.X.data)
    transformed_data = transform(transformed_data)["adata"]
    post_log_values = np.array(transformed_data.X.data)
    assert id(pre_log_values) != id(post_log_values)
    np.testing.assert_almost_equal(np.log1p(pre_log_values), post_log_values)


def test_standardize_gene_names_transform():
    adata = get_random_anndata()
    gene_map = {
        f"{adata.var_names[i]}": f"{adata.var_names[i]}_renamed"
        for i in range(adata.n_vars)
    }
    transform = sc_transforms.StandardizeGeneNamesTransform(gene_map=gene_map)
    transformed_data = transform(adata)["adata"]
    assert transformed_data.var_names[0] == f"{adata.var_names[0]}_renamed"

    adata = get_random_anndata()
    transform = sc_transforms.StandardizeGeneNamesTransform()
    transformed_data = transform(adata)["adata"]
    # default gene map is loaded from json
    # protein coding gene has 61,769 (current + prev + alias) symbols in the
    assert len(transform.gene_map) > 50_000
    # should not change the number of observations
    assert transformed_data.n_obs == adata.n_obs


def test_rename_genes_transform():
    adata = get_random_anndata()
    gene_map = {f"Gene_{i:d}": f"Gene_{i:d}_renamed" for i in range(adata.n_vars)}
    logger.info(gene_map)
    transform = sc_transforms.RenameGenesTransform(gene_map=gene_map)
    transformed_data = transform(adata)["adata"]
    assert transformed_data.var_names[0] == "Gene_0_renamed"


def test_keep_genes_transform():
    adata = get_random_anndata()
    genes_to_keep = [f"Gene_{i:d}" for i in range(adata.n_vars)][0:2]
    transform = sc_transforms.KeepGenesTransform(genes_to_keep=genes_to_keep)
    transformed_data = transform(adata)["adata"]
    assert transformed_data.n_vars == len(genes_to_keep)

    with tempfile.NamedTemporaryFile("w") as f:
        pd.Series(genes_to_keep).to_csv(f.name, index=False, header=False)
        transform = sc_transforms.KeepGenesTransform(genes_to_keep=f.name)
    transformed_data = transform(adata)["adata"]
    assert transformed_data.n_vars == len(genes_to_keep)


def test_keep_vocab_genes_transform():
    adata = get_random_anndata()
    transform = sc_transforms.KeepVocabGenesTransform(vocab_identifier="all_genes")
    transformed_data = transform(adata)["adata"]
    assert transformed_data.n_vars == 0


def test_explanatory_and_target_genes_transform():
    adata = get_random_anndata()
    explanatory_gene_set = [f"Gene_{i:d}" for i in range(adata.n_vars)][0:2]
    regression_label_columns = [f"Gene_{i:d}" for i in range(adata.n_vars)][2:4]
    transform = sc_transforms.KeepExplanatoryAndTargetGenesTransform(
        explanatory_gene_set=explanatory_gene_set,
        regression_label_columns=regression_label_columns,
    )
    transformed_data = transform(adata)["adata"]
    assert transformed_data.n_vars == len(
        explanatory_gene_set + regression_label_columns
    )


def test_hvg_transform():
    adata = get_random_anndata()
    pre_transform = sc_transforms.NormalizeTotalTransform(target_sum=10)
    transform = sc_transforms.HighlyVariableGenesTransform()
    transformed_data = pre_transform(adata)["adata"]
    transformed_data = transform(transformed_data)["adata"]
    assert transformed_data is not None


def test_move_expressions_to_labels_transform():
    adata = get_random_anndata()
    transform = sc_transforms.MoveExpressionsToLabels(
        regression_label_columns=[adata.var_names[0]]
    )
    transformed_data = transform(adata)["adata"]
    assert adata.var_names[0] in transformed_data.obs.columns
    assert adata.var_names[0] not in transformed_data.var


def test_int_cast_bin_transform():
    adata = get_random_anndata()
    pre_transform_norm_total = sc_transforms.NormalizeTotalTransform(target_sum=100)
    pre_transform_log = sc_transforms.LogTransform(base=2)
    n_bins = 4
    bin_transform = sc_transforms.BinTransform(
        num_bins=n_bins, binning_method="int_cast"
    )
    transformed_data = pre_transform_norm_total(adata)["adata"]
    transformed_data = pre_transform_log(adata)["adata"]
    pre_binned_data = np.array(transformed_data.X.data)
    transformed_data = bin_transform(transformed_data)["adata"]
    binned_data = np.array(transformed_data.X.data)
    bin_counts = np.bincount(binned_data)[1:]  # first element is zero
    assert 0 < len(bin_counts) <= n_bins


def test_percentile_bin_transform():
    adata = get_random_anndata()
    pre_transform_norm_total = sc_transforms.NormalizeTotalTransform(target_sum=10)
    pre_transform_log = sc_transforms.LogTransform(base=2)
    n_bins = 10
    bin_transform = sc_transforms.BinTransform(
        num_bins=n_bins, binning_method="percentile"
    )
    transformed_data = pre_transform_norm_total(adata)["adata"]
    transformed_data = pre_transform_log(adata)["adata"]
    pre_binned_data = np.array(transformed_data.X.data)
    transformed_data = bin_transform(transformed_data)["adata"]
    binned_data = np.array(transformed_data.X.data)
    bin_counts = np.bincount(binned_data)[1:]  # first element is zero
    equal_bin_size = len(binned_data) / n_bins
    # each bin should have roughly the same number of samples
    assert all(bin_counts > 0.5 * equal_bin_size)
    assert all(bin_counts < 2 * equal_bin_size)


@pytest.mark.skip(reason="sps integration fails, requires additional review")
def test_chained_transform():
    # Apply the filter cells transformation
    adata = get_random_anndata()
    filter_cells_transform = sc_transforms.FilterCellsTransform(min_genes=10)
    filter_genes_transform = sc_transforms.FilterGenesTransform(min_cells=5)
    norm_total_transform = sc_transforms.NormalizeTotalTransform(target_sum=10)
    log_transform = sc_transforms.LogTransform()
    hvg_transform = sc_transforms.HighlyVariableGenesTransform()
    bin_transform = sc_transforms.BinTransform(num_bins=10)

    transformed_data = filter_cells_transform(adata)["adata"]
    transformed_data = filter_genes_transform(transformed_data)["adata"]
    normalized_data = norm_total_transform(transformed_data)["adata"]
    lognormalized_data = log_transform(normalized_data)["adata"]
    hvg_transformed_data = hvg_transform(lognormalized_data)["adata"]
    binned_data = bin_transform(hvg_transformed_data)

    assert binned_data is not None


@pytest.mark.skip(reason="sps integration fails, requires additional review")
def test_chained_transform_bulk_RNA():
    # Apply the filter cells transformation
    adata = get_random_anndata_with_gene_names()

    tpm_transform = sc_transforms.TPMNormalizationTransform(
        gtf_file=helpers.TEST_GTF_PATH,
        filter_genes_min_value=1,
        filter_genes_min_percentage=0.25,
    )

    log_transform = sc_transforms.LogTransform(add_one=False)
    bin_transform = sc_transforms.BinTransform(num_bins=10)

    normalized_data = tpm_transform(adata)["adata"]
    lognormalized_data = log_transform(normalized_data)["adata"]
    binned_data = bin_transform(lognormalized_data)

    assert binned_data is not None


def test_deserialization():
    pre_transforms: list[dict[str, Any]] = [
        {
            "transform_name": "FilterCellsTransform",
            "transform_args": {
                "max_counts": None,
                "max_genes": None,
                "min_counts": None,
                "min_genes": 200,
            },
        },
        {
            "transform_name": "NormalizeTotalTransform",
            "transform_args": {
                "exclude_highly_expressed": False,
                "max_fraction": 0.05,
                "target_sum": 10000.0,
            },
        },
        {
            "transform_name": "LogTransform",
            "transform_args": {"base": None, "chunk_size": None, "chunked": None},
        },
        {
            "transform_name": "HighlyVariableGenesTransform",
            "transform_args": {
                "flavor": "seurat",
                "max_disp": float("inf"),
                "max_mean": 3,
                "min_disp": 0.5,
                "min_mean": 0.0125,
                "n_bins": 20,
                "n_top_genes": None,
                "span": 0.3,
            },
        },
        {
            "transform_name": "BinTransform",
            "transform_args": {
                "num_bins": 10,
            },
        },
    ]
    from bmfm_targets.transforms.compose import Compose
    from bmfm_targets.transforms.sc_transforms import make_transform

    compose = Compose(
        [make_transform(**transform_dict) for transform_dict in pre_transforms]
    )
    assert compose is not None
