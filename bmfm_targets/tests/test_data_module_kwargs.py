"""DataModule must reject config keys it does not consume instead of dropping them."""

import logging
from pathlib import Path

import hydra
import pytest
from omegaconf import OmegaConf

from bmfm_targets.datasets.zheng68k import Zheng68kDataModule
from bmfm_targets.tokenization import load_tokenizer
from bmfm_targets.training.data_module import DataModule

RELEASED_CHECKPOINT_CONFIGS = sorted(
    (Path(__file__).parents[2] / "run" / "checkpoints").glob("*.yaml")
)


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer("protein_coding")


def test_misspelled_key_raises_with_suggestion(tokenizer):
    with pytest.raises(TypeError, match="did you mean 'sequence_dropout_factor'"):
        DataModule(tokenizer=tokenizer, fields=[], sequence_dropout_ratio=0.2)


def test_subclass_rejects_unknown_key(tokenizer):
    """The check runs in the base __init__, after subclasses take their own kwargs."""
    with pytest.raises(TypeError, match="Zheng68kDataModule.*'not_an_option'"):
        Zheng68kDataModule(tokenizer=tokenizer, fields=[], not_an_option=1)


def test_stash_keys_are_accepted(tokenizer):
    stash = {k: "x" for k in DataModule.STASH_KEYS}
    dm = DataModule(tokenizer=tokenizer, fields=[], **stash)
    assert isinstance(dm, DataModule)


def test_retired_key_warns_and_loads(tokenizer, caplog):
    with caplog.at_level(logging.WARNING):
        DataModule(tokenizer=tokenizer, fields=[], collation_strategy="multitask")
    assert "data_module.collation_strategy is ignored" in caplog.text


def test_real_parameter_is_not_flagged(tokenizer):
    dm = DataModule(tokenizer=tokenizer, fields=[], sequence_dropout_factor=0.2)
    assert dm.sequence_dropout_factor == 0.2


@pytest.mark.parametrize(
    "config_path", RELEASED_CHECKPOINT_CONFIGS, ids=lambda p: p.stem
)
def test_released_checkpoint_data_module_configs_construct(config_path, tokenizer):
    """Every key in a released checkpoint's data_module block is one the class uses."""
    cfg = OmegaConf.load(config_path)
    partial_dm = hydra.utils.instantiate(cfg.data_module)
    dm = partial_dm(tokenizer=tokenizer, fields=[], label_columns=None)
    assert isinstance(dm, DataModule)


def test_released_configs_were_found():
    assert len(RELEASED_CHECKPOINT_CONFIGS) >= 6
