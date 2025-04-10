import multiprocessing as mp
import os
import shutil
import socket

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import random

from bmfm_targets.config.main_config import default_fields
from bmfm_targets.datasets.cellxgene.cellxgene_nexus_dataset import (
    CellXGeneNexusDataModule,
    CellXGeneNexusDataset,
    get_non_zero_genes_and_expressions,
)
from bmfm_targets.datasets.cellxgene.cellxgene_soma_utils import (
    build_range_index,
    get_split_value_filter,
)
from bmfm_targets.tests.helpers import CellXGenePaths
from bmfm_targets.tokenization.resources import get_protein_coding_genes

if "cccxl" in socket.gethostname():
    pytest.skip(
        "TileDB does not work on CCC login nodes. Use a CCC compute node or local test environment",
        allow_module_level=True,
    )

from logging import getLogger

logger = getLogger()


@pytest.fixture(scope="module")
def _build_nexus_index():
    """Building temporary index for NexusDB."""
    value_filter = "is_primary_data == True " + get_split_value_filter("train")
    uri = None
    index_dir = str(CellXGenePaths.nexus_index_path)
    os.mkdir(index_dir)
    train_index_dir = os.path.join(index_dir, "train")
    build_range_index(
        uri,
        train_index_dir,
        n_records=640 * 2,
        chunk_size=64 * 2,
        label_columns=["cell_type", "tissue"],
        value_filter=value_filter,
    )
    shutil.copytree(train_index_dir, os.path.join(index_dir, "dev"), dirs_exist_ok=True)
    logger.info(f"created index for nexus at {index_dir}")


@pytest.mark.skip(reason="memory usage crashes travis on vpc")
@pytest.mark.usefixtures("_build_nexus_index")
def test_nexus_matches_soma(pl_module_cellxgene_soma):
    soma_dm = pl_module_cellxgene_soma

    label = soma_dm.train_dataset.label_columns[0]
    for batch in soma_dm.train_dataloader():
        assert batch["input_ids"].shape[0] == soma_dm.batch_size
        assert batch["labels"][label].shape[0] == soma_dm.batch_size
        break

    soma_batch = batch

    nexus_dm = CellXGeneNexusDataModule(
        dataset_kwargs={
            "uri": None,
            "index_dir": str(CellXGenePaths.nexus_index_path),
        },
        tokenizer=soma_dm.tokenizer,
        fields=default_fields(),
        label_columns=soma_dm.label_columns,
        num_workers=0,
        collation_strategy="sequence_classification",
    )
    nexus_dm.setup(stage="fit")
    nexus_batch = next(iter(nexus_dm.train_dataloader()))

    assert (soma_batch["input_ids"] == nexus_batch["input_ids"]).all().item()
    assert (soma_batch["attention_mask"] == nexus_batch["attention_mask"]).all().item()
    assert soma_batch["cell_names"] == nexus_batch["cell_names"]
    soma_labels = soma_batch["labels"]
    nexus_labels = nexus_batch["labels"]
    for key in soma_labels.keys():
        assert (soma_labels[key] == nexus_labels[key]).all().item()


@pytest.fixture()
def all_genes_sample(_build_nexus_index):
    index_dir = str(CellXGenePaths.nexus_index_path)
    ds_all_genes = CellXGeneNexusDataset(uri=None, index_dir=index_dir, split="train")
    mfi_all_genes = ds_all_genes[0]
    return pd.Series(dict(zip(mfi_all_genes["genes"], mfi_all_genes["expressions"])))


@pytest.fixture()
def pc_genes_sample(_build_nexus_index):
    pc_genes = get_protein_coding_genes()
    index_dir = str(CellXGenePaths.nexus_index_path)
    ds = CellXGeneNexusDataset(
        uri=None, index_dir=index_dir, limit_genes=pc_genes, split="train"
    )
    mfi = ds[0]
    assert all(g in pc_genes for g in mfi["genes"])
    return pd.Series(dict(zip(mfi["genes"], mfi["expressions"])))


@pytest.mark.skip(
    reason="so slow it crashes travis sometimes, turn on to check intermittently"
)
def test_can_limit_to_protein_coding_genes(all_genes_sample, pc_genes_sample):
    # there are more genes without limiting
    assert len(all_genes_sample) > len(pc_genes_sample)
    # where the genes overlap the expressions are equal
    assert (
        pd.concat([all_genes_sample, pc_genes_sample], axis=1)
        .dropna()
        .diff(axis=1)
        .iloc[:, 1]
        .sum()
        == 0
    )


def test_csr_matrix_access():
    def data_rvs(n):
        return np.random.random(size=n) + 1

    n_genes = 1000
    n_cells = 1000
    test_matrix = random(m=n_cells, n=n_genes, format="csr", data_rvs=data_rvs)
    genes = np.array(range(n_genes))
    data = [
        (i, j)
        for item_index in range(n_cells)
        for i, j in zip(
            *get_non_zero_genes_and_expressions(item_index, test_matrix, genes)
        )
    ]

    dense_array = test_matrix.toarray()
    true_data = []
    for item_index in range(n_cells):
        for index, gene_index in enumerate(range(n_genes)):
            val = dense_array[item_index, gene_index]
            if val > 0.1:
                true_data.append((genes[index], val))

    assert true_data == data


@pytest.fixture()
def _use_spawn_multiprocessing():
    original_method = mp.get_start_method(allow_none=True)
    mp.set_start_method("spawn", force=True)
    yield
    mp.set_start_method(original_method, force=True)
