from functools import reduce
from itertools import chain

import pytest

from bmfm_targets.config import LabelColumnInfo
from bmfm_targets.datasets.sciplex3 import SciPlex3DataModule
from bmfm_targets.tests import helpers
from bmfm_targets.tokenization import get_all_genes_tokenizer


@pytest.fixture(scope="module")
def dataset_kwargs():
    ds_kwargs = {
        "data_dir": helpers.SciPlex3Paths.root,
        "split_column": "split_random",
    }
    return ds_kwargs


def test_weighted_sampler_sciplex3(dataset_kwargs, gene2vec_fields):
    label_columns = [LabelColumnInfo(label_column_name="control")]
    dm = SciPlex3DataModule(
        dataset_kwargs=dataset_kwargs,
        label_columns=label_columns,
        tokenizer=get_all_genes_tokenizer(),
        balancing_label_column="control",
        fields=gene2vec_fields,
        collation_strategy="sequence_classification",
        batch_size=256,
    )
    dm.setup("fit")
    helpers.update_label_columns(dm.label_columns, dm.label_dict)
    unbalanced_level = dm.train_dataset.processed_data.obs.control.mean()
    for batch in dm.train_dataloader():
        balanced_level = batch["labels"]["control"].to(float).mean()
        tol = 0.1
        assert abs(balanced_level - 0.5) < abs(unbalanced_level - 0.5)
        assert (
            abs(balanced_level - 0.5) < tol
        ), f"control label distribution is unbalanced: {balanced_level}"
        break


def test_bootstrap_sampler_is_resampling(dataset_kwargs, gene2vec_fields):
    label_columns = [LabelColumnInfo(label_column_name="control")]
    dm = SciPlex3DataModule(
        dataset_kwargs=dataset_kwargs,
        tokenizer=get_all_genes_tokenizer(),
        fields=gene2vec_fields,
        label_columns=label_columns,
        collation_strategy="sequence_classification",
        batch_size=2,
        limit_dataset_samples=10,
        bootstrap_test_dataloader=True,
    )

    dm.setup("test")
    all_bootstraps_cell_names = []
    for _ in range(10):
        bootstrap_cell_names = []
        for batch in dm.test_dataloader():
            bootstrap_cell_names.extend([*batch["cell_name"]])
        all_bootstraps_cell_names.append(bootstrap_cell_names)

    # at least some of the runs have the same sample twice
    assert any(len({*i}) < 10 for i in all_bootstraps_cell_names)
    # all together we get exactly 10 unique samples
    assert len({*reduce(chain, all_bootstraps_cell_names, {})}) == 10

    # make sure we don't harm the regular behavior
    dm.bootstrap_test_dataloader = False

    identical_bootstraps_cell_names = []
    for _ in range(10):
        bootstrap_cell_names = []
        for batch in dm.test_dataloader():
            bootstrap_cell_names.extend([*batch["cell_name"]])
        identical_bootstraps_cell_names.append(bootstrap_cell_names)

    # every sample has exactly 10 unique samples
    assert all(len({*i}) == 10 for i in identical_bootstraps_cell_names)
    # all together we get exactly 10 unique samples
    assert len({*reduce(chain, identical_bootstraps_cell_names, {})}) == 10
