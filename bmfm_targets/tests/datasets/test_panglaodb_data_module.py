import pytest
import torch

from bmfm_targets.datasets.panglaodb import PanglaoDBDataModule
from bmfm_targets.tests.helpers import check_unk_levels_for_dm
from bmfm_targets.tokenization.resources import get_protein_coding_genes

# pytest fixture `pl_data_module`, `pl_data_module_panglao_limit10`, defined in conftest.py`


def test_train_dataloader(pl_data_module_panglao: PanglaoDBDataModule):
    max_len = pl_data_module_panglao.max_length
    for batch in pl_data_module_panglao.train_dataloader():
        assert tuple(batch["attention_mask"].shape) == (3, max_len)
        assert tuple(batch["input_ids"].shape) == (3, 2, max_len)
        break


def test_val_dataloader(pl_data_module_panglao: PanglaoDBDataModule):
    max_len = pl_data_module_panglao.max_length
    for batch in pl_data_module_panglao.val_dataloader():
        assert tuple(batch["attention_mask"].shape) == (3, max_len)
        assert tuple(batch["input_ids"].shape) == (3, 2, max_len)
        break


def test_predict_dataloader_not_implemented(
    pl_data_module_panglao: PanglaoDBDataModule,
):
    with pytest.raises(NotImplementedError):
        pl_data_module_panglao.predict_dataloader()


# Note this threshold is too high. See issue for details
# https://github.ibm.com/BiomedSciAI-Innersource/bmfm-targets/issues/119
@pytest.mark.skip(reason="Not working yet")
def test_data_tokenization(pl_data_module_panglao: PanglaoDBDataModule):
    check_unk_levels_for_dm(pl_data_module_panglao, 0.05)


def test_subsample_data(pl_data_module_panglao_limit10):
    dm = pl_data_module_panglao_limit10
    assert len(dm.train_dataset) == 10
    assert len(dm.dev_dataset) == 10


def test_panglao_limit_to_protein_coding(pl_data_module_panglao_protein_coding):
    dm = pl_data_module_panglao_protein_coding
    dm.setup("fit")
    train_data = dm.train_dataset.processed_data

    pc_genes = get_protein_coding_genes()
    assert not train_data.var_names.empty
    assert all(i in pc_genes for i in train_data.var_names)


def test_non_binned_data_retains_precision(
    pl_data_module_panglao_regression_no_binning,
):
    mfi = pl_data_module_panglao_regression_no_binning.dev_dataset[0]
    asints = [int(s) for s in mfi["expressions"]]
    assert not all(i == j for i, j in zip(asints, mfi["expressions"]))


def test_non_binned_data_batches_to_float_tensor(
    pl_data_module_panglao_regression_no_binning,
):
    for batch in pl_data_module_panglao_regression_no_binning.train_dataloader():
        assert batch["input_ids"].dtype in (torch.float32, torch.float64)
        break
