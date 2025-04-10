import socket

import pytest

from bmfm_targets.config.main_config import default_fields
from bmfm_targets.datasets.cellxgene.cellxgene_soma_dataset import (
    CellXGeneSOMADataModule,
)
from bmfm_targets.tokenization import MultiFieldInstance, load_tokenizer

if "cccxl" in socket.gethostname() or True:
    pytest.skip(
        "TileDB does not work on CCC login nodes. Use a CCC compute node or local test environment",
        allow_module_level=True,
    )


def test_dataset(pl_module_cellxgene_soma):
    dm = pl_module_cellxgene_soma
    ds = dm.train_dataset
    for i in ds.output_datapipe:
        assert isinstance(i[0], MultiFieldInstance)
        break


def test_multiple_labels_in_dataset(pl_module_cellxgene_soma):
    dm = pl_module_cellxgene_soma
    ds = dm.train_dataset

    assert all(label in ds.label_dict for label in ds.label_columns)

    for i in ds.output_datapipe:
        assert all(label in i[0].metadata for label in ds.label_columns)
        break


def test_train_dataloader(pl_module_cellxgene_soma):
    dm = pl_module_cellxgene_soma
    label = dm.train_dataset.label_columns[0]
    for batch in dm.train_dataloader():
        assert batch["input_ids"].shape[0] == dm.batch_size
        assert batch["labels"][label].shape[0] == dm.batch_size
        break


@pytest.mark.skip(reason="something is wrong with limiting soma")
def test_limit_val_dataloader():
    tokenizer = load_tokenizer()
    limit_dev_samples = 8
    batch_size = 32
    dm = CellXGeneSOMADataModule(
        dataset_kwargs={"uri": None},
        batch_size=batch_size,
        tokenizer=tokenizer,
        fields=default_fields(),
        num_workers=0,
        collation_strategy="sequence_classification",
        limit_dataset_samples={"dev": limit_dev_samples},
    )
    count = 0

    dm.setup("validate")
    for batch in dm.val_dataloader():
        count += batch["input_ids"].shape[0]
    assert count == limit_dev_samples * batch_size
