import os
import tempfile
from pathlib import Path
from unittest import mock

import hydra
import pytest
from hydra.core.config_store import ConfigStore

from bmfm_targets import config
from bmfm_targets.tasks import task_utils
from bmfm_targets.tests import helpers
from bmfm_targets.training.callbacks import InitialCheckpoint

yaml_dir = Path(__file__).parents[1] / "tasks" / "scbert"
cs = ConfigStore.instance()
cs.store(name="base_scbert_config", node=config.SCBertMainHydraConfigSchema)

TEST_ENV = {
    "BMFM_TARGETS_PANGLAO_DATA": helpers.PanglaoPaths.root,
    "BMFM_TARGETS_PANGLAO_METADATA": helpers.PanglaoPaths.test_metadata,
    "BMFM_TARGETS_ZHENG68K_DATA": helpers.Zheng68kPaths.root,
    "BMFM_TARGETS_HUMANCELLATLAS_DATA": helpers.HumanCellAtlasPaths.root,
    "BMFM_TARGETS_CELLXGENE_DATA": helpers.CellXGenePaths.root,
    "BMFM_TARGETS_SCIBD_DATA": helpers.scIBDPaths.root,
    "BMFM_TARGETS_SCIPLEX3_DATA": helpers.SciPlex3Paths.root,
    "BMFM_TARGETS_SCIBD300K_DATA": helpers.scIBD300kPaths.root,
    "BMFM_TARGETS_ADAMSON_DATA": helpers.ScperturbPerturbationPaths.root,
    "BMFM_TARGETS_NORMAN_DATA": helpers.GearsPerturbationPaths.root,
    "BMFM_TARGETS_TRAINING_DIR": tempfile.TemporaryDirectory().name,
}

TEST_ENV = {k: str(v) for k, v in TEST_ENV.items()}


@mock.patch.dict(os.environ, TEST_ENV)
@pytest.mark.usefixtures("_panglao_convert_rdata_and_transform")
def test_scbert_train_panglaodb_config():
    with hydra.initialize_config_dir(config_dir=str(yaml_dir), version_base="1.2"):
        overrides = [
            "~track_clearml",
            "++model.checkpoint=null",
            "++data_module.dataset_kwargs.processed_dir_name=processed",
        ]
        cfg = hydra.compose(config_name="scbert_train_panglaodb", overrides=overrides)
    task_utils.main_config(cfg)


@mock.patch.dict(os.environ, TEST_ENV)
@pytest.mark.usefixtures("pl_data_module_zheng68k_seq_cls")
def test_scbert_train_zheng68k_config():
    with hydra.initialize_config_dir(config_dir=str(yaml_dir), version_base="1.2"):
        overrides = [
            "~track_clearml",
            "++model.checkpoint=null",
            "++data_module.transform_datasets=false",
        ]
        cfg = hydra.compose(config_name="scbert_train_zheng68k", overrides=overrides)
    task_utils.main_config(cfg)


@mock.patch.dict(os.environ, TEST_ENV)
@pytest.mark.usefixtures("pl_data_module_panglao_dynamic_binning")
def test_scbert_train_panglaodb_dynamic_bins_config():
    with hydra.initialize_config_dir(config_dir=str(yaml_dir), version_base="1.2"):
        overrides = [
            "~track_clearml",
            "++model.checkpoint=null",
            "++data_module.dataset_kwargs.processed_dir_name=processed",
        ]
        cfg = hydra.compose(
            config_name="scbert_train_panglaodb_dynamic_bins", overrides=overrides
        )
    task_utils.main_config(cfg)


@mock.patch.dict(os.environ, TEST_ENV)
def test_scbert_train_panglaodb_dynamic_bins_config_with_init_callback(
    pl_data_module_panglao_dynamic_binning,
):
    with hydra.initialize_config_dir(config_dir=str(yaml_dir), version_base="1.2"):
        overrides = [
            "~track_clearml",
            "++model.checkpoint=null",
            "++data_module.dataset_kwargs.processed_dir_name=processed",
        ]
        cfg = hydra.compose(
            config_name="scbert_train_panglaodb_dynamic_bins", overrides=overrides
        )
    cfg_obj = task_utils.main_config(cfg)
    assert any(
        filter(lambda obj: isinstance(obj, InitialCheckpoint), cfg_obj.task.callbacks)
    )


@mock.patch.dict(os.environ, TEST_ENV)
def test_scbert_interpret_yaml(zheng_seq_cls_ckpt):
    with hydra.initialize_config_dir(config_dir=str(yaml_dir), version_base="1.2"):
        overrides = [
            "++data_module.max_length=32",
            "++data_module.limit_dataset_samples=7",
            f"++task.checkpoint={zheng_seq_cls_ckpt}",
            "++task.accelerator=cpu",
            "++task.precision=32",
            "++data_module.num_workers=0",
            "++task.attribute_kwargs.n_steps=2",
        ]
        cfg = hydra.compose(
            config_name="scbert_interpret_zheng68k", overrides=overrides
        )

    cfg_obj = task_utils.main_config(cfg)
    task_utils.main_run(
        cfg_obj.task,
        cfg_obj.model,
        cfg_obj.data_module,
        cfg_obj.trainer,
        clearml_logger=None,
    )


@mock.patch.dict(os.environ, TEST_ENV)
def test_scbert_predict_yaml(zheng_seq_cls_ckpt):
    with hydra.initialize_config_dir(config_dir=str(yaml_dir), version_base="1.2"):
        overrides = [
            "++data_module.max_length=32",
            "++data_module.limit_dataset_samples=7",
            f"++task.checkpoint={zheng_seq_cls_ckpt}",
            "++task.accelerator=cpu",
            "++data_module.num_workers=0",
            "++task.precision=32",
        ]
        cfg = hydra.compose(config_name="scbert_predict_zheng68k", overrides=overrides)
    cfg_obj = task_utils.main_config(cfg)
    task_utils.main_run(
        cfg_obj.task,
        cfg_obj.model,
        cfg_obj.data_module,
        cfg_obj.trainer,
        clearml_logger=None,
    )


@mock.patch.dict(os.environ, TEST_ENV)
def test_scbert_predict_yaml_mlm(zheng_mlm_ckpt):
    with hydra.initialize_config_dir(config_dir=str(yaml_dir), version_base="1.2"):
        overrides = [
            "++data_module.max_length=32",
            "++data_module.limit_dataset_samples=7",
            "++data_module.collation_strategy='language_modeling'",
            f"++data_module.processed_name={str(helpers.Zheng68kPaths.raw_counts_name)}",
            "++data_module.transform_datasets=False",
            "++data_module.mlm=False",
            "++data_module.num_workers=0",
            f"++task.checkpoint={zheng_mlm_ckpt}",
            "++task.accelerator=cpu",
            "++task.precision=32",
        ]
        cfg = hydra.compose(config_name="scbert_predict_zheng68k", overrides=overrides)
    cfg_obj = task_utils.main_config(cfg)
    task_utils.main_run(
        cfg_obj.task,
        cfg_obj.model,
        cfg_obj.data_module,
        cfg_obj.trainer,
        clearml_logger=None,
    )


@mock.patch.dict(os.environ, TEST_ENV)
def test_scbert_predict_across_datasets(
    sciplex3_seqcls_mt_model_and_ckpt, zheng_mlm_ckpt
):
    assert zheng_mlm_ckpt, "need this for the preprocessed data files"
    _, ckpt = sciplex3_seqcls_mt_model_and_ckpt
    with hydra.initialize_config_dir(config_dir=str(yaml_dir), version_base="1.2"):
        overrides = [
            "++data_module.max_length=32",
            "++data_module.limit_dataset_samples=7",
            f"++data_module.processed_name={str(helpers.Zheng68kPaths.raw_counts_name)}",
            "++data_module.transform_datasets=False",
            "++data_module.mlm=False",
            "++data_module.num_workers=0",
            f"++task.checkpoint={ckpt}",
            "++task.accelerator=cpu",
            "++task.precision=32",
        ]
        cfg = hydra.compose(config_name="scbert_predict_zheng68k", overrides=overrides)
    cfg_obj = task_utils.main_config(cfg)
    task_utils.main_run(
        cfg_obj.task,
        cfg_obj.model,
        cfg_obj.data_module,
        cfg_obj.trainer,
        clearml_logger=None,
    )


@mock.patch.dict(os.environ, TEST_ENV)
def test_pertubation_prediction_config():
    with hydra.initialize_config_dir(config_dir=str(yaml_dir), version_base="1.2"):
        overrides = [
            "++data_module.max_length=32",
            "++data_module.limit_dataset_samples=7",
            "++data_module.batch_size=3",
            "++data_module.num_workers=0",
            "~track_clearml",
            "++model.checkpoint=null",
            "++task.accelerator=cpu",
            "++task.precision=32",
            "++model.num_attention_heads=2",
            "++model.num_hidden_layers=2",
            "++model.intermediate_size=32",
            "++model.hidden_size=16",
            "++task.max_epochs=1",
        ]
        cfg = hydra.compose(
            config_name="scbert_train_perturbation_ft",
            overrides=overrides,
        )
    cfg_obj = task_utils.main_config(cfg)


# TODO: the test hangs up in travis if litdata is unpinned.
@mock.patch.dict(os.environ, TEST_ENV)
def test_scbert_test_yaml(zheng_seq_cls_ckpt):
    with hydra.initialize_config_dir(config_dir=str(yaml_dir), version_base="1.2"):
        overrides = [
            "++data_module.max_length=32",
            "++data_module.limit_dataset_samples=7",
            "++data_module.num_workers=0",
            f"++task.checkpoint={zheng_seq_cls_ckpt}",
            "++task.accelerator=cpu",
            "++task.precision=32",
            "++task.num_bootstrap_runs=3",
        ]
        cfg = hydra.compose(config_name="scbert_test_zheng68k", overrides=overrides)
    cfg_obj = task_utils.main_config(cfg)
    task_utils.main_run(
        cfg_obj.task,
        cfg_obj.model,
        cfg_obj.data_module,
        cfg_obj.trainer,
        clearml_logger=None,
    )


@mock.patch.dict(os.environ, TEST_ENV)
def test_scbert_zheng_mlm_masking_strategy(pl_zheng_mlm_raw_counts):
    with hydra.initialize_config_dir(config_dir=str(yaml_dir), version_base="1.2"):
        overrides = [
            "++data_module.num_workers=0",
            "++data_module.transform_datasets=true",
            "++task.accelerator=cpu",
            "++task.precision=32",
            "++data_module.max_length=32",
            "++data_module.limit_dataset_samples=7",
            "++data_module.batch_size=3",
            "++data_module.transform_datasets=false",
            "~track_clearml",
            "++model.checkpoint=null",
            "++task.accelerator=cpu",
            "++task.precision=32",
            "++task.val_check_interval=2",
            "++model.num_attention_heads=2",
            "++model.num_hidden_layers=2",
            "++model.intermediate_size=32",
            "++model.hidden_size=16",
            "++task.max_epochs=1",
        ]
        cfg = hydra.compose(
            config_name="scbert_train_zheng68k_mlm", overrides=overrides
        )
    cfg_obj = task_utils.main_config(cfg)
    task_utils.main_run(
        cfg_obj.task,
        cfg_obj.model,
        cfg_obj.data_module,
        cfg_obj.trainer,
        clearml_logger=None,
    )


@mock.patch.dict(os.environ, TEST_ENV)
def test_scbert_zheng_wced():
    with hydra.initialize_config_dir(config_dir=str(yaml_dir), version_base="1.2"):
        overrides = [
            "++data_module.num_workers=0",
            "++data_module.transform_datasets=true",
            "++task.accelerator=cpu",
            "++task.precision=32",
            "++data_module.max_length=32",
            "++data_module.limit_dataset_samples=7",
            "++data_module.batch_size=3",
            "~track_clearml",
            "++model.checkpoint=null",
            "++task.accelerator=cpu",
            "++task.precision=32",
            "++task.val_check_interval=2",
            "++model.num_attention_heads=2",
            "++model.num_hidden_layers=2",
            "++model.intermediate_size=32",
            "++model.hidden_size=16",
            "++task.max_epochs=1",
            "++task.limit_train_batches=3",
            "++task.limit_val_batches=3",
        ]
        cfg = hydra.compose(
            config_name="scbert_train_zheng68k_wced", overrides=overrides
        )

    cfg_obj = task_utils.main_config(cfg)
    task_utils.main_run(
        cfg_obj.task,
        cfg_obj.model,
        cfg_obj.data_module,
        cfg_obj.trainer,
        clearml_logger=None,
    )


@mock.patch.dict(os.environ, TEST_ENV)
def test_scbert_zheng_wced_scheduled_dynamic_training():
    with hydra.initialize_config_dir(config_dir=str(yaml_dir), version_base="1.2"):
        overrides = [
            "++data_module.num_workers=0",
            "++data_module.transform_datasets=true",
            "++task.accelerator=cpu",
            "++task.precision=32",
            "++data_module.max_length=scheduled",
            "++data_module.limit_dataset_samples=7",
            "++data_module.batch_size=scheduled",
            "~track_clearml",
            "++model.checkpoint=null",
            "++task.accelerator=cpu",
            "++task.precision=32",
            "++task.val_check_interval=2",
            "++model.num_attention_heads=2",
            "++model.num_hidden_layers=2",
            "++model.intermediate_size=32",
            "++model.hidden_size=16",
            "++task.max_epochs=6",
            "++task.limit_train_batches=6",
            "++task.limit_val_batches=3",
            "task.callbacks=[{_target_: bmfm_targets.training.callbacks.BatchSizeScheduler, schedule: [{max_length: 32, batch_size: 1}, {max_length: 16, batch_size: 2}]}]",
        ]
        cfg = hydra.compose(
            config_name="scbert_train_zheng68k_wced_scheduled_dynamic_training",
            overrides=overrides,
        )

    cfg_obj = task_utils.main_config(cfg)
    task_utils.main_run(
        cfg_obj.task,
        cfg_obj.model,
        cfg_obj.data_module,
        cfg_obj.trainer,
        clearml_logger=None,
    )
