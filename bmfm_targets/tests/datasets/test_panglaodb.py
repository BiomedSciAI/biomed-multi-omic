import tempfile
from pathlib import Path

import anndata
import pytest
import rdata

from bmfm_targets.datasets.panglaodb import PanglaoDBDataset
from bmfm_targets.datasets.panglaodb.panglaodb_converter import (
    convert_all_rdatas_to_h5ad,
    create_sra_splits,
)

from ..helpers import PanglaoPaths


@pytest.fixture(scope="module")
def splits():
    return create_sra_splits(
        data_info_path=PanglaoPaths.test_metadata,
        rdata_dir=PanglaoPaths.root / "rdata",
    )


def test_sra_splits_no_filter():
    filter_query = None
    splits = create_sra_splits(
        data_info_path=PanglaoPaths.all_metadata,
        rdata_dir=PanglaoPaths.root,
        filter_query=filter_query,
    )
    assert splits["train"].shape == (1082, 13)
    assert splits["dev"].shape == (135, 13)
    assert splits["test"].shape == (136, 13)


def test_sra_splits_test_data(splits):
    assert splits["train"].shape == (8, 13)
    assert splits["dev"].shape == (1, 13)
    assert splits["test"].shape == (2, 13)


def test_sra_splits_homo_sapiens_filter():
    filter_query = 'Species == "Homo sapiens"'
    splits = create_sra_splits(
        data_info_path=PanglaoPaths.all_metadata,
        rdata_dir=PanglaoPaths.root,
        filter_query=filter_query,
    )
    assert splits["train"].shape == (239, 13)
    assert splits["dev"].shape == (29, 13)
    assert splits["test"].shape == (31, 13)


def test_convert_all_rdata(splits):
    input_data = [
        rdata.conversion.convert(rdata.parser.parse_file(i))
        for i in (PanglaoPaths.root / "rdata").glob("*.RData")
    ]
    input_sizes = {tuple(reversed(ns.Dim)) for i in input_data for ns in i.values()}
    input_file_count = sum(map(len, splits.values()))
    with tempfile.TemporaryDirectory() as output_dir:
        convert_all_rdatas_to_h5ad(
            h5ad_dir=output_dir, sra_df_dict=splits, num_workers=0
        )
        output_files = [*Path(output_dir).rglob("*.h5ad")]
        output_file_count = len(output_files)
        assert output_file_count == input_file_count

        output_data = [anndata.read_h5ad(f) for f in output_files]
        output_sizes = {i.shape for i in output_data}
        assert output_sizes == input_sizes


@pytest.mark.usefixtures("_panglao_convert_rdata_and_transform")
def test_panglao_dataset_size_and_needle_train(splits):
    split = "train"
    num_workers = 1
    dataset = PanglaoDBDataset(
        data_dir=PanglaoPaths.root,
        data_info_path=PanglaoPaths.test_metadata,
        convert_rdata_to_h5ad=False,
        transform_datasets=False,
        split=split,
        num_workers=num_workers,
    )
    files_in_split = len(splits[split])

    assert len(dataset) == files_in_split * 50
    check_needle_in_dataset(dataset, "GACGATGCCACG", "ZCCHC10")


@pytest.mark.usefixtures("_panglao_convert_rdata_and_transform")
def test_panglao_dataset_size_and_needle_dev(splits):
    split = "dev"
    num_workers = 1
    dataset = PanglaoDBDataset(
        data_dir=PanglaoPaths.root,
        data_info_path=PanglaoPaths.test_metadata,
        convert_rdata_to_h5ad=False,
        transform_datasets=False,
        split=split,
        num_workers=num_workers,
    )
    files_in_split = len(splits[split])

    assert len(dataset) == files_in_split * 50
    check_needle_in_dataset(dataset, "ATTGTCGTCGTC", "AC010343.1")


def check_needle_in_dataset(dataset, needle_barcode, barcode_first_gene):
    sample_mfi = [mfi for mfi in dataset if mfi.metadata["cell_name"] == needle_barcode]
    assert len(sample_mfi) == 1

    first_read = sample_mfi[0]["genes"][0]
    assert first_read == barcode_first_gene


@pytest.mark.usefixtures("_panglao_convert_rdata_and_transform")
def test_panglao_can_expose_zeros():
    split = "dev"
    dataset = PanglaoDBDataset(
        data_dir=PanglaoPaths.root,
        data_info_path=PanglaoPaths.test_metadata,
        convert_rdata_to_h5ad=False,
        transform_datasets=False,
        split=split,
        num_workers=0,
        expose_zeros="all",
    )
    assert 0 in dataset[0]["expressions"]
    dataset = PanglaoDBDataset(
        data_dir=PanglaoPaths.root,
        data_info_path=PanglaoPaths.test_metadata,
        convert_rdata_to_h5ad=False,
        transform_datasets=False,
        split=split,
        num_workers=0,
        expose_zeros=None,
    )
    assert not 0 in dataset[0]["expressions"]
