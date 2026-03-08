"""
Unified training tests for all model architectures and training scenarios.

All tests use MultiTaskTrainingModule - the distinction between MLM, seq_cls, and
multitask is just about whether losses target fields, labels, or both.
"""
import copy
import os
import tempfile

import numpy as np
import pytest
import torch
from transformers.models import auto

from bmfm_targets import config
from bmfm_targets.config import LabelColumnInfo
from bmfm_targets.datasets.panglaodb import PanglaoDBDataModule
from bmfm_targets.datasets.SNPdb.streaming_snp_dataset import (
    StreamingHiCDataModule,
    StreamingInsulationDataModule,
    StreamingSNPdbDataModule,
)
from bmfm_targets.datasets.zheng68k import Zheng68kDataModule
from bmfm_targets.models import register_configs_and_models
from bmfm_targets.models.predictive.llama import LlamaForMultiTaskConfig
from bmfm_targets.tasks.task_utils import make_trainer_for_task, train
from bmfm_targets.tests import helpers
from bmfm_targets.tests.helpers import (
    Zheng68kPaths,
    get_test_task_config,
    make_model_config_with_ckpt,
)
from bmfm_targets.tokenization import load_tokenizer
from bmfm_targets.training.callbacks import InitialCheckpoint
from bmfm_targets.training.losses import (
    CrossEntropyObjective,
    IsZeroBCEObjective,
    LossTask,
    MSEObjective,
)
from bmfm_targets.training.modules import MultiTaskTrainingModule

from .helpers import default_mlm_losses_from_fields

register_configs_and_models()


# ============================================================================
# Checkpoint Tests
# ============================================================================


def test_module_saves_all_hyperparameters(zheng_seq_cls_ckpt):
    """Verify checkpoint saves model and trainer configs."""
    ckpt_dict = torch.load(zheng_seq_cls_ckpt, weights_only=False)

    saved_model_config = ckpt_dict["hyper_parameters"]["model_config"]
    assert isinstance(saved_model_config, config.SCBertConfig)

    saved_trainer_config = ckpt_dict["hyper_parameters"]["trainer_config"]
    assert isinstance(saved_trainer_config, config.TrainerConfig)


@pytest.mark.parametrize(
    ("config_cls", "extra_kwargs"),
    [
        pytest.param(config.SCBertConfig, {"intermediate_size": 32}, id="scbert"),
        pytest.param(LlamaForMultiTaskConfig, {"attention": "torch"}, id="llama"),
    ],
)
def test_train_and_initial_checkpoint_save(
    pl_data_module_panglao: PanglaoDBDataModule,
    gene2vec_fields: list[config.FieldInfo],
    config_cls,
    extra_kwargs,
):
    """Test that InitialCheckpoint callback saves model before training."""
    model_config = config_cls(
        fields=gene2vec_fields,
        num_attention_heads=2,
        num_hidden_layers=2,
        hidden_size=32,
        **extra_kwargs,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        callback = InitialCheckpoint(dirpath=tmpdir, filename="initial")
        task_config = get_test_task_config(tmpdir)
        task_config.callbacks = [callback]
        trainer_config = config.TrainerConfig(
            losses=default_mlm_losses_from_fields(gene2vec_fields)
        )
        mlm_training_module = MultiTaskTrainingModule(
            model_config, trainer_config, pl_data_module_panglao.tokenizer
        )

        # Get initial embeddings path (model-specific)
        if hasattr(mlm_training_module.model, "scbert"):
            embed_path = "model.scbert.embeddings.genes_embeddings.weight"
            initial_embeddings = (
                mlm_training_module.model.scbert.embeddings.genes_embeddings.weight.detach()
                .numpy()
                .copy()
            )
        else:  # Llama
            embed_path = "model.core.encoder.embeddings.genes_embeddings.weight"
            initial_embeddings = (
                mlm_training_module.model.core.encoder.embeddings.genes_embeddings.weight.detach()
                .numpy()
                .copy()
            )

        pl_trainer = make_trainer_for_task(task_config)
        train(
            pl_trainer,
            pl_data_module=pl_data_module_panglao,
            pl_module=mlm_training_module,
            task_config=task_config,
        )

        filepath = os.path.join(tmpdir, "initial")
        save_initial = (
            torch.load(filepath, weights_only=False)["state_dict"][embed_path]
            .detach()
            .numpy()
        )
        np.testing.assert_allclose(initial_embeddings, save_initial)


# ============================================================================
# Basic Training Tests (Field Losses Only - "MLM")
# ============================================================================


@pytest.mark.parametrize(
    ("config_cls", "extra_kwargs"),
    [
        pytest.param(config.SCBertConfig, {"intermediate_size": 32}, id="scbert"),
        pytest.param(LlamaForMultiTaskConfig, {"attention": "torch"}, id="llama"),
        pytest.param(
            config.SCModernBertConfig,
            {"intermediate_size": 32, "num_landmarks": 2},
            id="modernbert",
        ),
        pytest.param(
            config.SCPerformerConfig, {"intermediate_size": 32}, id="performer"
        ),
    ],
)
def test_train_with_field_losses(
    pl_data_module_panglao: PanglaoDBDataModule,
    gene2vec_fields: list[config.FieldInfo],
    config_cls,
    extra_kwargs,
):
    """Test training with field losses only (traditional MLM)."""
    model_config = config_cls(
        fields=gene2vec_fields,
        num_attention_heads=2,
        num_hidden_layers=2,
        hidden_size=16,
        **extra_kwargs,
    )
    run_train_checkpoint_test(pl_data_module_panglao, model_config=model_config)


# ============================================================================
# Basic Training Tests (Label Losses Only - "Seq Cls")
# ============================================================================


@pytest.mark.parametrize(
    ("config_cls", "fields_fixture", "extra_kwargs"),
    [
        pytest.param(
            config.SCPerformerConfig,
            "gene2vec_unmasked_fields",
            {"intermediate_size": 64},
            id="performer",
        ),
        pytest.param(
            config.SCModernBertConfig,
            "gene2vec_unmasked_fields",
            {"intermediate_size": 64},
            id="modernbert",
        ),
        pytest.param(
            LlamaForMultiTaskConfig,
            "gene2vec_unmasked_fields",
            {},
            id="llama",
        ),
        pytest.param(
            config.SCNystromformerConfig,
            "gene2vec_fields",
            {"intermediate_size": 64, "num_landmarks": 2},
            id="nystromformer",
        ),
    ],
)
def test_train_with_label_losses(
    pl_data_module_zheng68k_seq_cls,
    request,
    config_cls,
    fields_fixture,
    extra_kwargs,
):
    """Test training with label losses only (traditional sequence classification)."""
    fields = request.getfixturevalue(fields_fixture)

    trainer_config = config.TrainerConfig(
        losses=[LossTask.from_label("celltype", objective=CrossEntropyObjective())]
    )

    model_config = config_cls(
        batch_size=pl_data_module_zheng68k_seq_cls.batch_size,
        fields=fields,
        label_columns=pl_data_module_zheng68k_seq_cls.label_columns,
        num_attention_heads=2,
        num_hidden_layers=2,
        hidden_size=32,
        max_position_embeddings=pl_data_module_zheng68k_seq_cls.max_length,
        **extra_kwargs,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = config.TrainingTaskConfig(
            default_root_dir=tmpdir,
            max_epochs=1,
            max_steps=3,
            val_check_interval=3,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[],
        )

        seq_cls_training_module = MultiTaskTrainingModule(
            model_config,
            trainer_config,
            pl_data_module_zheng68k_seq_cls.tokenizer,
            pl_data_module_zheng68k_seq_cls.label_dict,
        )
        pl_trainer = make_trainer_for_task(task_config)
        train(
            pl_trainer,
            pl_data_module=pl_data_module_zheng68k_seq_cls,
            pl_module=seq_cls_training_module,
            task_config=task_config,
        )


# ============================================================================
# Mixed Training Tests (Field + Label Losses - "Multitask")
# ============================================================================


@pytest.mark.parametrize(
    ("config_cls", "extra_kwargs"),
    [
        (config.SCPerformerConfig, {"intermediate_size": 64}),
        (config.SCModernBertConfig, {"intermediate_size": 64}),
        (config.SCBertConfig, {"intermediate_size": 64}),
        (LlamaForMultiTaskConfig, {}),
    ],
)
def test_train_with_field_and_label_losses(
    pl_data_module_zheng68k_multitask: Zheng68kDataModule,
    config_cls,
    extra_kwargs,
):
    """Test training with both field and label losses (true multitask)."""
    trainer_config = config.TrainerConfig(
        losses=[LossTask.from_label("celltype", objective=CrossEntropyObjective())]
    )

    model_config = config_cls(
        batch_size=pl_data_module_zheng68k_multitask.batch_size,
        fields=pl_data_module_zheng68k_multitask.fields,
        label_columns=pl_data_module_zheng68k_multitask.label_columns,
        num_attention_heads=2,
        num_hidden_layers=2,
        hidden_size=32,
        **extra_kwargs,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = config.TrainingTaskConfig(
            default_root_dir=tmpdir,
            max_epochs=1,
            max_steps=3,
            val_check_interval=3,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[],
        )

        multitask_training_module = MultiTaskTrainingModule(
            model_config,
            trainer_config,
            pl_data_module_zheng68k_multitask.tokenizer,
            pl_data_module_zheng68k_multitask.label_dict,
        )
        pl_trainer = make_trainer_for_task(task_config)
        train(
            pl_trainer,
            pl_data_module=pl_data_module_zheng68k_multitask,
            pl_module=multitask_training_module,
            task_config=task_config,
        )


def test_train_nystromformer_with_mlm_losses(pl_data_module_zheng68k_multitask):
    """Nystromformer uses MLM losses by default."""
    trainer_config = config.TrainerConfig(
        losses=default_mlm_losses_from_fields(pl_data_module_zheng68k_multitask.fields)
    )
    model_config = config.SCNystromformerConfig(
        batch_size=pl_data_module_zheng68k_multitask.batch_size,
        fields=pl_data_module_zheng68k_multitask.fields,
        label_columns=pl_data_module_zheng68k_multitask.label_columns,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        num_landmarks=2,
        max_position_embeddings=pl_data_module_zheng68k_multitask.max_length,
        hidden_size=32,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = config.TrainingTaskConfig(
            default_root_dir=tmpdir,
            max_epochs=1,
            max_steps=3,
            val_check_interval=3,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[],
        )
        multitask_training_module = MultiTaskTrainingModule(
            model_config,
            trainer_config,
            pl_data_module_zheng68k_multitask.tokenizer,
            pl_data_module_zheng68k_multitask.label_dict,
        )
        pl_trainer = make_trainer_for_task(task_config)
        train(
            pl_trainer,
            pl_data_module=pl_data_module_zheng68k_multitask,
            pl_module=multitask_training_module,
            task_config=task_config,
        )


# ============================================================================
# Freeze Layers Tests
# ============================================================================


@pytest.mark.parametrize(
    ("config_cls", "extra_kwargs"),
    [
        pytest.param(config.SCBertConfig, {}, id="scbert"),
        pytest.param(LlamaForMultiTaskConfig, {}, id="llama"),
    ],
)
def test_train_with_frozen_layers(
    pl_data_module_zheng68k_seq_cls,
    gene2vec_fields: list[config.FieldInfo],
    config_cls,
    extra_kwargs,
):
    """Test training with freeze_layers config (verifies training runs, not that freezing works)."""
    trainer_config = config.TrainerConfig(
        losses=[LossTask.from_label("celltype", objective=CrossEntropyObjective())]
    )

    model_config = config_cls(
        batch_size=pl_data_module_zheng68k_seq_cls.batch_size,
        fields=gene2vec_fields,
        label_columns=pl_data_module_zheng68k_seq_cls.label_columns,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        hidden_size=32,
        freeze_layers=[0],  # Freeze first layer
        **extra_kwargs,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = config.TrainingTaskConfig(
            default_root_dir=tmpdir,
            max_epochs=1,
            max_steps=3,
            val_check_interval=3,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[],
        )
        seq_cls_training_module = MultiTaskTrainingModule(
            model_config,
            trainer_config,
            pl_data_module_zheng68k_seq_cls.tokenizer,
            pl_data_module_zheng68k_seq_cls.label_dict,
        )

        pl_trainer = make_trainer_for_task(task_config)
        train(
            pl_trainer,
            pl_data_module=pl_data_module_zheng68k_seq_cls,
            pl_module=seq_cls_training_module,
            task_config=task_config,
        )


# ============================================================================
# Finetune Tests
# ============================================================================


@pytest.mark.parametrize(
    ("pretrain_config_cls", "extra_kwargs"),
    [
        (config.SCBertConfig, {"intermediate_size": 128}),
        (LlamaForMultiTaskConfig, {}),
    ],
)
def test_finetune_from_checkpoint(
    pl_data_module_zheng68k_seq_cls,
    pl_data_module_zheng68k_multitask,
    gene2vec_unmasked_fields: list[config.FieldInfo],
    pretrain_config_cls,
    extra_kwargs,
):
    """Test finetuning from a pretrained checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Pretrain with field+label losses
        trainer_config = config.TrainerConfig(
            losses=[
                LossTask.from_label("celltype", objective=CrossEntropyObjective()),
                LossTask.from_field("expressions", objective=CrossEntropyObjective()),
            ]
        )

        model_config = pretrain_config_cls(
            batch_size=pl_data_module_zheng68k_multitask.batch_size,
            fields=pl_data_module_zheng68k_multitask.fields,
            label_columns=pl_data_module_zheng68k_multitask.label_columns,
            num_attention_heads=2,
            num_hidden_layers=2,
            **extra_kwargs,
        )

        task_config = config.TrainingTaskConfig(
            default_root_dir=tmpdir + "/pretrain",
            max_epochs=1,
            max_steps=3,
            accelerator="cpu",
            val_check_interval=1,
            gradient_clip_val=0.5,
            precision="32",
            enable_model_summary=False,
            enable_progress_bar=False,
            enable_checkpointing=True,
            callbacks=[],
        )
        multitask_training_module = MultiTaskTrainingModule(
            model_config, trainer_config, pl_data_module_zheng68k_multitask.tokenizer
        )
        pl_trainer = make_trainer_for_task(task_config)
        train(
            pl_trainer,
            pl_data_module=pl_data_module_zheng68k_multitask,
            pl_module=multitask_training_module,
            task_config=task_config,
        )

        pretrain_ckpt_path = pl_trainer.checkpoint_callback._last_checkpoint_saved
        pretrain_model = torch.load(pretrain_ckpt_path, weights_only=False)[
            "state_dict"
        ]

        # Finetune with label losses only
        finetune_task_config = config.TrainingTaskConfig(
            default_root_dir=tmpdir + "/finetune",
            max_epochs=1,
            max_steps=3,
            accelerator="cpu",
            val_check_interval=1,
            gradient_clip_val=0.5,
            precision="32",
            enable_model_summary=False,
            enable_progress_bar=False,
            enable_checkpointing=True,
            callbacks=[],
        )

        model_config = config.SCBertConfig(
            batch_size=pl_data_module_zheng68k_seq_cls.batch_size,
            label_columns=pl_data_module_zheng68k_seq_cls.label_columns,
            checkpoint=pretrain_ckpt_path,
            fields=gene2vec_unmasked_fields,
            num_attention_heads=2,
            num_hidden_layers=2,
            intermediate_size=128,
        )

        trainer_config = config.TrainerConfig(
            losses=[LossTask.from_label("celltype", objective=CrossEntropyObjective())]
        )
        sequence_classification_training_module = MultiTaskTrainingModule(
            model_config,
            trainer_config,
            label_dict=pl_data_module_zheng68k_seq_cls.label_dict,
        )
        pl_trainer = make_trainer_for_task(task_config)
        train(
            pl_trainer,
            pl_data_module=pl_data_module_zheng68k_seq_cls,
            pl_module=sequence_classification_training_module,
            task_config=finetune_task_config,
        )

        finetune_ckpt_path = pl_trainer.checkpoint_callback._last_checkpoint_saved
        finetune_model = torch.load(finetune_ckpt_path, weights_only=False)[
            "state_dict"
        ]

        # Verify models have different keys (different decoders)
        assert finetune_model.keys() != pretrain_model.keys()


# ============================================================================
# Regression Tests
# ============================================================================


@pytest.mark.parametrize(
    ("config_cls", "extra_kwargs"),
    [
        pytest.param(config.SCBertConfig, {"intermediate_size": 32}, id="scbert"),
        pytest.param(LlamaForMultiTaskConfig, {}, id="llama"),
        pytest.param(
            config.SCModernBertConfig, {"intermediate_size": 32}, id="modernbert"
        ),
    ],
)
def test_train_with_regression_losses(
    pl_data_module_panglao_regression: PanglaoDBDataModule,
    gene2vec_fields_regression_with_tokenization: list[config.FieldInfo],
    config_cls,
    extra_kwargs,
):
    """Test training with regression (MSE) losses."""
    model_config = config_cls(
        fields=gene2vec_fields_regression_with_tokenization,
        num_attention_heads=2,
        num_hidden_layers=2,
        hidden_size=16,
        **extra_kwargs,
    )
    run_train_checkpoint_test(
        pl_data_module_panglao_regression,
        trainer_config=config.TrainerConfig(
            losses=[
                LossTask.from_field(
                    field_name="genes",
                    objective=CrossEntropyObjective(),
                    weight=1,
                    metrics=[],
                ),
                LossTask.from_field(
                    field_name="expressions",
                    objective=MSEObjective(),
                    weight=1,
                ),
            ],
        ),
        model_config=model_config,
    )


@pytest.mark.parametrize(
    ("config_cls", "extra_kwargs"),
    [
        pytest.param(config.SCBertConfig, {"intermediate_size": 32}, id="scbert"),
        pytest.param(LlamaForMultiTaskConfig, {}, id="llama"),
    ],
)
def test_train_with_continuous_value_encoder(
    pl_zheng_mlm_raw_counts,
    config_cls,
    extra_kwargs,
):
    """Test training with continuous value encoder (CVE) for raw counts."""
    model_config = config_cls(
        fields=pl_zheng_mlm_raw_counts.fields,
        num_attention_heads=2,
        num_hidden_layers=2,
        hidden_size=16,
        pad_token_id=2,
        **extra_kwargs,
    )
    run_train_checkpoint_test(
        pl_zheng_mlm_raw_counts,
        trainer_config=config.TrainerConfig(
            losses=[
                LossTask.from_field(
                    field_name="genes",
                    objective=CrossEntropyObjective(),
                    weight=1,
                    metrics=[],
                ),
                LossTask.from_field(
                    field_name="expressions",
                    objective=MSEObjective(ignore_zero=True),
                    weight=1,
                ),
                LossTask.from_field(
                    field_name="expressions",
                    objective=IsZeroBCEObjective(),
                    weight=1,
                ),
            ],
        ),
        model_config=model_config,
    )


@pytest.mark.parametrize(
    ("config_cls", "extra_kwargs"),
    [
        pytest.param(config.SCBertConfig, {"intermediate_size": 32}, id="scbert"),
        pytest.param(LlamaForMultiTaskConfig, {}, id="llama"),
    ],
)
def test_train_with_cve_plus_mvc(
    all_genes_fields_with_regression_masking_plus_mvc,
    pl_zheng_mlm_raw_counts,
    config_cls,
    extra_kwargs,
):
    """Test training with CVE + Masked Value Completion (MVC)."""
    fields = all_genes_fields_with_regression_masking_plus_mvc

    dm = pl_zheng_mlm_raw_counts.__class__(
        data_dir=pl_zheng_mlm_raw_counts.data_dir,
        processed_name=Zheng68kPaths.raw_counts_name,
        transform_datasets=False,
        tokenizer=pl_zheng_mlm_raw_counts.tokenizer,
        label_columns=pl_zheng_mlm_raw_counts.label_columns,
        fields=fields,
        batch_size=2,
        limit_dataset_samples=8,
        mlm=True,
        rda_transform="auto_align",
        max_length=20,
        pad_to_multiple_of=2,
    )

    model_config = config_cls(
        fields=fields,
        num_attention_heads=2,
        num_hidden_layers=2,
        hidden_size=16,
        pad_token_id=2,
        **extra_kwargs,
    )
    run_train_checkpoint_test(
        dm,
        trainer_config=config.TrainerConfig(
            batch_prediction_behavior="track",
            losses=[
                LossTask.from_field(
                    field_name="expressions",
                    objective=MSEObjective(ignore_zero=True),
                    weight=1,
                    loss_group="group1",
                ),
                LossTask.from_field(
                    field_name="expressions",
                    objective=IsZeroBCEObjective(),
                    weight=1,
                    loss_group="group1",
                ),
                LossTask.from_field(
                    field_name="expressions",
                    objective=MSEObjective(ignore_zero=True),
                    weight=1,
                    loss_group="group2",
                    decoder_key="expressions_mvc_regression",
                ),
                LossTask.from_field(
                    field_name="expressions",
                    objective=IsZeroBCEObjective(),
                    weight=1,
                    loss_group="group2",
                    decoder_key="expressions_mvc_is_zero",
                ),
            ],
        ),
        model_config=model_config,
    )


@pytest.mark.parametrize(
    ("config_cls", "extra_kwargs"),
    [
        pytest.param(config.SCBertConfig, {"intermediate_size": 32}, id="scbert"),
        pytest.param(LlamaForMultiTaskConfig, {}, id="llama"),
    ],
)
def test_train_with_rda(
    pl_data_module_panglao_rda: PanglaoDBDataModule,
    all_genes_fields_with_rda_regression_masking: list[config.FieldInfo],
    mock_clearml_logger,
    config_cls,
    extra_kwargs,
):
    """Test training with Reference Data Alignment (RDA)."""
    model_config = config_cls(
        fields=all_genes_fields_with_rda_regression_masking,
        num_attention_heads=2,
        num_hidden_layers=2,
        hidden_size=16,
        **extra_kwargs,
    )
    run_train_checkpoint_test(
        pl_data_module_panglao_rda,
        trainer_config=config.TrainerConfig(
            losses=[
                LossTask.from_field(
                    field_name="expressions",
                    objective=MSEObjective(),
                    weight=1,
                ),
            ],
        ),
        model_config=model_config,
    )


def test_train_with_regression_label_output(
    pl_data_module_zheng68k_seq_cls,  # Needed for transform_datasets=True side effect
    gene2vec_fields: list[config.FieldInfo],
):
    """Test training with regression label (not field) output."""
    tokenizer = load_tokenizer("gene2vec")
    label_columns = pl_data_module_zheng68k_seq_cls.label_columns + [
        config.LabelColumnInfo("n_counts", is_regression_label=True),
    ]
    dm = Zheng68kDataModule(
        tokenizer=tokenizer,
        fields=gene2vec_fields,
        label_columns=label_columns,
        data_dir=helpers.Zheng68kPaths.root,
        transform_datasets=False,
        mlm=False,
        num_workers=0,
        batch_size=3,
        max_length=16,
        limit_dataset_samples={"train": 12, "dev": 12, "predict": 2},
    )
    dm.setup("fit")

    trainer_config = config.TrainerConfig(
        losses=[LossTask.from_label("n_counts", objective=MSEObjective())]
    )

    model_config = config.SCBertConfig(
        batch_size=dm.batch_size,
        fields=gene2vec_fields,
        label_columns=label_columns,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        hidden_size=32,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = config.TrainingTaskConfig(
            default_root_dir=tmpdir,
            max_epochs=1,
            max_steps=3,
            val_check_interval=3,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[],
        )

        seq_cls_training_module = MultiTaskTrainingModule(
            model_config, trainer_config, tokenizer, dm.label_dict
        )
        pl_trainer = make_trainer_for_task(task_config)
        train(
            pl_trainer,
            pl_data_module=dm,
            pl_module=seq_cls_training_module,
            task_config=task_config,
        )


# ============================================================================
# Gradient Reversal Tests
# ============================================================================


@pytest.mark.parametrize(
    ("config_cls", "extra_kwargs"),
    [
        pytest.param(config.SCBertConfig, {"intermediate_size": 64}, id="scbert"),
        pytest.param(LlamaForMultiTaskConfig, {}, id="llama"),
    ],
)
def test_train_with_gradient_reversal_layer(
    pl_data_module_zheng68k_seq_cls,  # Needed for transform_datasets=True side effect
    gene2vec_fields: list[config.FieldInfo],
    config_cls,
    extra_kwargs,
):
    """Test training with gradient reversal layer for adversarial training."""
    tokenizer = load_tokenizer("gene2vec")
    label_columns = pl_data_module_zheng68k_seq_cls.label_columns + [
        config.LabelColumnInfo(
            "n_counts",
            is_regression_label=True,
            gradient_reversal_coefficient=0.1,
        ),
    ]
    dm = Zheng68kDataModule(
        tokenizer=tokenizer,
        fields=gene2vec_fields,
        label_columns=label_columns,
        data_dir=helpers.Zheng68kPaths.root,
        transform_datasets=False,
        mlm=False,
        num_workers=0,
        batch_size=3,
        max_length=16,
        limit_dataset_samples={"train": 12, "dev": 12, "predict": 2},
    )
    dm.setup("fit")

    trainer_config = config.TrainerConfig(
        losses=[
            LossTask.from_label("celltype", objective=CrossEntropyObjective()),
            LossTask.from_label("n_counts", objective=MSEObjective()),
        ]
    )

    model_config = config_cls(
        batch_size=dm.batch_size,
        fields=gene2vec_fields,
        label_columns=label_columns,
        num_attention_heads=2,
        num_hidden_layers=2,
        hidden_size=32,
        **extra_kwargs,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = config.TrainingTaskConfig(
            default_root_dir=tmpdir,
            max_epochs=1,
            max_steps=3,
            val_check_interval=3,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[],
        )

        pl_trainer = make_trainer_for_task(task_config)
        seq_cls_training_module = MultiTaskTrainingModule(
            model_config, trainer_config, tokenizer, dm.label_dict
        )
        train(
            pl_trainer,
            pl_data_module=dm,
            pl_module=seq_cls_training_module,
            task_config=task_config,
        )


# ============================================================================
# Special Dataset Tests
# ============================================================================


@pytest.mark.usefixtures("_convert_raw_to_lit")
def test_train_snpdb(
    streaming_snpdb_parameters,
    snp2vec_fields: list[config.FieldInfo],
):
    """Test training on SNPdb dataset."""
    datamodule = StreamingSNPdbDataModule(**streaming_snpdb_parameters)
    datamodule.prepare_data()
    datamodule.setup()
    model_config = config.SCNystromformerConfig(
        batch_size=datamodule.batch_size,
        fields=snp2vec_fields,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=32,
        hidden_size=16,
    )
    run_train_checkpoint_test(
        datamodule,
        trainer_config=config.TrainerConfig(
            losses=[
                LossTask.from_field(
                    field_name="dna_chunks",
                    objective=CrossEntropyObjective(),
                    weight=1,
                )
            ]
        ),
        model_config=model_config,
    )


@pytest.mark.usefixtures("_convert_hic_raw_to_lit")
@pytest.mark.parametrize(
    ("config_cls", "extra_kwargs"),
    [
        (config.SCBertConfig, {"intermediate_size": 128}),
        (LlamaForMultiTaskConfig, {}),
    ],
)
def test_train_hic(streaming_hic_parameters, hic_fields, config_cls, extra_kwargs):
    """Test training on HiC dataset with field and label losses."""
    pl_data_module_hic = StreamingHiCDataModule(**streaming_hic_parameters)
    pl_data_module_hic.prepare_data()
    pl_data_module_hic.setup()

    losses = [
        LossTask.from_label("hic_contact", objective=CrossEntropyObjective()),
        LossTask.from_field("dna_chunks", objective=CrossEntropyObjective()),
    ]

    model_config = config_cls(
        batch_size=pl_data_module_hic.batch_size,
        fields=pl_data_module_hic.fields,
        label_columns=pl_data_module_hic.label_columns,
        num_attention_heads=2,
        num_hidden_layers=2,
        hidden_size=32,
        **extra_kwargs,
    )

    run_train_checkpoint_test(
        pl_data_module_hic,
        trainer_config=config.TrainerConfig(losses=losses),
        model_config=model_config,
    )


@pytest.mark.usefixtures("_convert_insulation_raw_to_lit")
@pytest.mark.parametrize(
    ("config_cls", "extra_kwargs"),
    [
        (config.SCBertConfig, {"intermediate_size": 128}),
        (LlamaForMultiTaskConfig, {"intermediate_size": 128}),
    ],
)
def test_train_insulation(
    streaming_insulation_parameters,
    insulation_fields,
    config_cls,
    extra_kwargs,
):
    """Test training on Insulation dataset with regression label."""
    pl_data_module_insulation = StreamingInsulationDataModule(
        **streaming_insulation_parameters
    )
    pl_data_module_insulation.prepare_data()
    pl_data_module_insulation.setup()

    losses = [
        LossTask.from_label("insulation", objective=MSEObjective()),
        LossTask.from_field("dna_chunks", objective=CrossEntropyObjective()),
    ]

    model_config = config_cls(
        batch_size=pl_data_module_insulation.batch_size,
        fields=pl_data_module_insulation.fields,
        label_columns=pl_data_module_insulation.label_columns,
        num_attention_heads=2,
        num_hidden_layers=2,
        hidden_size=32,
        **extra_kwargs,
    )

    run_train_checkpoint_test(
        pl_data_module_insulation,
        trainer_config=config.TrainerConfig(losses=losses),
        model_config=model_config,
    )


# ============================================================================
# Batch Tracking Tests
# ============================================================================


@pytest.mark.parametrize(
    ("config_cls", "extra_kwargs"),
    [
        (config.SCBertConfig, {"intermediate_size": 128}),
        (LlamaForMultiTaskConfig, {}),
    ],
)
def test_batch_prediction_tracking(
    pl_data_module_zheng68k_multitask,
    mock_clearml_logger,
    config_cls,
    extra_kwargs,
):
    """Test that batch predictions are tracked correctly."""
    dm0 = pl_data_module_zheng68k_multitask
    label_columns = pl_data_module_zheng68k_multitask.label_columns + [
        config.LabelColumnInfo(
            label_column_name="n_counts",
            is_regression_label=True,
            gradient_reversal_coefficient=0.1,
        )
    ]
    dm = Zheng68kDataModule(
        tokenizer=dm0.tokenizer,
        fields=dm0.fields,
        label_columns=label_columns,
        data_dir=dm0.data_dir,
        transform_datasets=False,
        mlm=True,
        num_workers=0,
        batch_size=3,
        max_length=16,
        limit_dataset_samples={"train": 12, "dev": 12, "predict": 2},
    )
    dm.setup("fit")

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer_config = config.TrainerConfig(
            losses=[
                LossTask.from_label("celltype", objective=CrossEntropyObjective()),
                LossTask.from_label("n_counts", objective=MSEObjective()),
                LossTask.from_field("expressions", objective=CrossEntropyObjective()),
            ],
            batch_prediction_behavior="track",
        )

        model_config = config_cls(
            batch_size=dm.batch_size,
            fields=dm.fields,
            label_columns=dm.label_columns,
            num_attention_heads=2,
            num_hidden_layers=2,
            **extra_kwargs,
        )

        task_config = config.TrainingTaskConfig(
            default_root_dir=tmpdir + "/pretrain",
            max_epochs=1,
            max_steps=3,
            accelerator="cpu",
            val_check_interval=1,
            gradient_clip_val=0.5,
            precision="32",
            enable_model_summary=False,
            enable_progress_bar=False,
            enable_checkpointing=True,
            callbacks=[],
        )
        multitask_training_module = MultiTaskTrainingModule(
            model_config=model_config,
            trainer_config=trainer_config,
            tokenizer=dm.tokenizer,
            label_dict=dm.label_dict,
        )
        pl_trainer = make_trainer_for_task(task_config)
        train(
            pl_trainer,
            pl_data_module=dm,
            pl_module=multitask_training_module,
            task_config=task_config,
        )

        # Verify batch tracking worked
        val_batches = multitask_training_module.split_batch_predictions("validation")
        assert len(val_batches["expressions"]) > 0
        assert len(val_batches["celltype"]) > 0
        n_counts_lt = [
            lt
            for lt in multitask_training_module.loss_tasks
            if lt.source.name == "n_counts"
        ][0]
        assert n_counts_lt.objective.name == "mse"


# ============================================================================
# Helper Functions
# ============================================================================


def run_train_checkpoint_test(pl_data_module, model_config, trainer_config=None):
    """Helper to run training and verify checkpoint save/load."""
    if trainer_config is None:
        trainer_config = config.TrainerConfig(
            losses=default_mlm_losses_from_fields(model_config.fields)
        )
    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = get_test_task_config(tmpdir)
        mlm_training_module = MultiTaskTrainingModule(
            model_config, trainer_config, pl_data_module.tokenizer
        )
        pl_trainer = make_trainer_for_task(task_config)
        train(
            pl_trainer,
            pl_data_module=pl_data_module,
            pl_module=mlm_training_module,
            task_config=task_config,
        )

        ckpt_path = str(task_config.default_root_dir) + "/last.ckpt"

        trained_model = mlm_training_module.model
        model_config_with_ckpt = make_model_config_with_ckpt(
            mlm_training_module.model_config, ckpt_path
        )
        ensure_all_parameters_load_for_mlm(trained_model, model_config_with_ckpt)
        ensure_shared_parameters_load_for_seq_cls(trained_model, model_config_with_ckpt)

        ckpt_dict = torch.load(ckpt_path, weights_only=False)
        saved_model_config = ckpt_dict["hyper_parameters"]["model_config"]
        assert isinstance(saved_model_config, type(model_config))

        saved_trainer_config = ckpt_dict["hyper_parameters"]["trainer_config"]
        assert isinstance(saved_trainer_config, type(trainer_config))


def ensure_shared_parameters_load_for_seq_cls(trained_model, model_config):
    """Verify shared parameters can be loaded into seq_cls model."""
    seq_cls_model_config = copy.deepcopy(model_config)
    seq_cls_model_config.label_columns = [
        LabelColumnInfo(label_column_name="dummy", n_unique_values=3)
    ]
    seq_cls_from_ckpt = auto.AutoModelForSequenceClassification.from_config(
        seq_cls_model_config
    )

    shared_params = {x[0] for x in trained_model.named_parameters()} & {
        x[0] for x in seq_cls_from_ckpt.named_parameters()
    }
    assert len(shared_params) > 0
    for param in shared_params:
        torch.testing.assert_close(
            trained_model.get_parameter(param), seq_cls_from_ckpt.get_parameter(param)
        )


def ensure_all_parameters_load_for_mlm(trained_model, model_config):
    """Verify all parameters can be loaded into MLM model."""
    loaded_model = auto.AutoModelForMaskedLM.from_config(model_config)

    for t, l in zip(trained_model.named_parameters(), loaded_model.named_parameters()):
        torch.testing.assert_close(t[1], l[1])


# Made with Bob
