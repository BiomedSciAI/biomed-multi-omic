import pytest

from bmfm_targets.config import LabelColumnInfo
from bmfm_targets.datasets.bulk_rna import (
    BulkRnaDataModuleLabeled,
    BulkRnaDatasetLabeled,
    BulkRnaDatasetUnlabeled,
)
from bmfm_targets.tokenization import get_gene2vec_tokenizer

from ..helpers import (
    BulkrnaIBDPaths,
    check_h5ad_file_is_csr,
    check_h5ad_file_structure,
    check_splits,
    ensure_expected_indexes_and_labels_present,
    transform_and_return_args,
)


@pytest.fixture(scope="module")
def bulkrnaIBD_labels():
    bulkrnaIBD_label_dicts = [
        {"label_column_name": "characteristics_ch1_1_demographics_gender"}
    ]
    bulkrnaIBD_labels = [LabelColumnInfo(**ld) for ld in bulkrnaIBD_label_dicts]
    return bulkrnaIBD_labels


@pytest.fixture(scope="module")
def dataset_kwargs(bulkrnaIBD_labels):
    label_columns = [label.label_column_name for label in bulkrnaIBD_labels]
    stratifying_label = next(
        (
            label.label_column_name
            for label in bulkrnaIBD_labels
            if label.is_stratification_label
        ),
        None,
    )
    return transform_and_return_args(
        BulkrnaIBDPaths,
        BulkRnaDatasetLabeled,
        label_columns=label_columns,
        filter_query="~characteristics_ch1_2_ibd_disease.isin(['Control','UC_Pouch','CD_Pouch'])",
    )


def test_bulkrnaibd_unlabeled_picks_correct_split_column(dataset_kwargs):
    unlabeld_ds = BulkRnaDatasetUnlabeled(
        processed_data_source=dataset_kwargs["processed_data_source"], split=None
    )
    assert unlabeld_ds.split_column_name == "split_random"


def test_bulkrnaIBD_filter_query(dataset_kwargs):
    ds_train = BulkRnaDatasetLabeled(**dataset_kwargs, split="train")
    assert (
        "Control"
        not in ds_train.processed_data.obs["characteristics_ch1_2_ibd_disease"]
    )
    assert (
        "UC_Pouch"
        not in ds_train.processed_data.obs["characteristics_ch1_2_ibd_disease"]
    )
    assert (
        "CD_Pouch"
        not in ds_train.processed_data.obs["characteristics_ch1_2_ibd_disease"]
    )


def test_bulkrnaIBD_csr_matrix_load_type(dataset_kwargs):
    ds_train = BulkRnaDatasetLabeled(**dataset_kwargs, split="train")
    check_h5ad_file_is_csr(ds_train.processed_data)


def test_bulkrnaIBD_h5ad_structure():
    h5ad_path = BulkrnaIBDPaths.root / "h5ad" / "bulk_rna.h5ad"
    check_h5ad_file_structure(h5ad_path)


def test_loaded_samples_are_valid(dataset_kwargs):
    ensure_expected_indexes_and_labels_present(
        dataset_kwargs,
        label_column_names=["characteristics_ch1_1_demographics_gender"],
        dataset_factory=BulkRnaDatasetLabeled,
    )


def test_bulkrnaIBD_data_module_sequence_prediction(
    dataset_kwargs, gene2vec_fields, bulkrnaIBD_labels
):
    dm = BulkRnaDataModuleLabeled(
        dataset_kwargs=dataset_kwargs,
        tokenizer=get_gene2vec_tokenizer(),
        fields=gene2vec_fields,
        label_columns=bulkrnaIBD_labels,
        collation_strategy="sequence_classification",
        transform_datasets=False,
    )
    dm.prepare_data()
    dm.setup("fit")
    label = dm.train_dataset.label_columns[0]
    for batch in dm.train_dataloader():
        assert {*batch["labels"][label].numpy()}.issubset(dm.label_dict[label].values())
        break
    for batch in dm.val_dataloader():
        assert {*batch["labels"][label].numpy()}.issubset(dm.label_dict[label].values())
        break


def test_splits():
    check_splits(
        BulkrnaIBDPaths.root / "h5ad" / "bulk_rna.h5ad",
        split_weights={"train": 0.8, "dev": 0.1, "test": 0.1},
        test_needle="GSM6039457",
        balancing_label="characteristics_ch1_1_demographics_gender",
    )
