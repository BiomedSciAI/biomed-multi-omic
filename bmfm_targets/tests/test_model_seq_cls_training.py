import tempfile

import torch

from bmfm_targets import config
from bmfm_targets.datasets.zheng68k import Zheng68kDataModule

# test_ function args are pytest fixtures defined in conftest.py`
from bmfm_targets.tasks.task_utils import make_trainer_for_task, train, train_run
from bmfm_targets.tests import helpers
from bmfm_targets.training.modules import SequenceClassificationTrainingModule
from bmfm_targets.training.modules.masked_language_modeling import (
    MLMTrainingModule,
)

from .helpers import SciPlex3Paths


def test_module_saves_all_hyperparameters(mock_data_seq_cls_ckpt):
    ckpt_dict = torch.load(mock_data_seq_cls_ckpt, weights_only=False)

    saved_model_config = ckpt_dict["hyper_parameters"]["model_config"]
    assert isinstance(saved_model_config, config.SCBertConfig)

    saved_trainer_config = ckpt_dict["hyper_parameters"]["trainer_config"]
    assert isinstance(saved_trainer_config, config.TrainerConfig)


def test_train_seq_cls_performer(
    pl_data_module_mock_data_seq_cls: Zheng68kDataModule,
    gene2vec_unmasked_fields: list[config.FieldInfo],
):
    trainer_config = config.TrainerConfig(
        losses=[{"label_column_name": "celltype"}], batch_prediction_behavior="track"
    )

    model_config = config.SCPerformerConfig(
        batch_size=pl_data_module_mock_data_seq_cls.batch_size,
        fields=gene2vec_unmasked_fields,
        label_columns=pl_data_module_mock_data_seq_cls.label_columns,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        hidden_size=32,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = config.TrainingTaskConfig(
            accelerator="cpu",
            default_root_dir=tmpdir,
            max_epochs=1,
            max_steps=3,
            val_check_interval=3,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[],
        )
        pl_trainer = make_trainer_for_task(task_config)
        seq_cls_training_module = SequenceClassificationTrainingModule(
            model_config,
            trainer_config,
            label_dict=pl_data_module_mock_data_seq_cls.label_dict,
        )

        train(
            pl_trainer,
            pl_data_module=pl_data_module_mock_data_seq_cls,
            pl_module=seq_cls_training_module,
            task_config=task_config,
        )
        preds_df = seq_cls_training_module.prediction_df["celltype"]
        assert preds_df.shape[0] == len(pl_data_module_mock_data_seq_cls.dev_dataset)
        assert preds_df.shape[1] == 2 + len(
            pl_data_module_mock_data_seq_cls.label_dict["celltype"]
        )


def test_train_seq_cls_nystromformer(
    pl_data_module_mock_data_seq_cls: Zheng68kDataModule,
    gene2vec_fields: list[config.FieldInfo],
):
    trainer_config = config.TrainerConfig(losses=[{"label_column_name": "celltype"}])
    model_config = config.SCNystromformerConfig(
        batch_size=pl_data_module_mock_data_seq_cls.batch_size,
        fields=gene2vec_fields,
        label_columns=pl_data_module_mock_data_seq_cls.label_columns,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        num_landmarks=2,
        max_position_embeddings=pl_data_module_mock_data_seq_cls.max_length,
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
        pl_trainer = make_trainer_for_task(task_config)
        seq_cls_training_module = SequenceClassificationTrainingModule(
            model_config,
            trainer_config,
            label_dict=pl_data_module_mock_data_seq_cls.label_dict,
        )
        train(
            pl_trainer,
            pl_data_module=pl_data_module_mock_data_seq_cls,
            pl_module=seq_cls_training_module,
            task_config=task_config,
        )


def test_train_mlm_finetune_scbert(
    pl_data_module_panglao,
    pl_data_module_mock_data_seq_cls,
    gene2vec_fields: list[config.FieldInfo],
):
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer_config = config.TrainerConfig(
            losses=helpers.default_mlm_losses_from_fields(pl_data_module_panglao.fields)
        )

        model_config = config.SCBertConfig(
            batch_size=pl_data_module_panglao.batch_size,
            fields=gene2vec_fields,
            num_attention_heads=2,
            num_hidden_layers=2,
            intermediate_size=128,
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
        pl_trainer = make_trainer_for_task(task_config)
        mlm_training_module = MLMTrainingModule(
            model_config, trainer_config, pl_data_module_panglao.tokenizer
        )
        train(
            pl_trainer,
            pl_data_module=pl_data_module_panglao,
            pl_module=mlm_training_module,
            task_config=task_config,
        )

        pretrain_ckpt_path = (
            task_config.default_root_dir
            + "/lightning_logs/version_0/checkpoints/epoch=0-step=3.ckpt"
        )
        pretrain_model = torch.load(pretrain_ckpt_path, weights_only=False)

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
        finetune_pl_trainer = make_trainer_for_task(finetune_task_config)
        model_config = config.SCBertConfig(
            batch_size=pl_data_module_mock_data_seq_cls.batch_size,
            label_columns=pl_data_module_mock_data_seq_cls.label_columns,
            checkpoint=pretrain_ckpt_path,
            fields=gene2vec_fields,
            num_attention_heads=2,
            num_hidden_layers=2,
            intermediate_size=128,
        )
        seq_cls_trainer_config = config.TrainerConfig(
            losses=[{"label_column_name": "celltype"}]
        )
        seq_cls_training_module = SequenceClassificationTrainingModule(
            model_config,
            seq_cls_trainer_config,
            label_dict=pl_data_module_mock_data_seq_cls.label_dict,
        )
        train(
            finetune_pl_trainer,
            pl_data_module=pl_data_module_mock_data_seq_cls,
            pl_module=seq_cls_training_module,
            task_config=finetune_task_config,
        )

        finetune_ckpt_path = (
            finetune_task_config.default_root_dir
            + "/lightning_logs/version_0/checkpoints/epoch=0-step=3.ckpt"
        )
        finetune_model = torch.load(finetune_ckpt_path, weights_only=False)

        assert (
            finetune_model["state_dict"].keys() != pretrain_model["state_dict"].keys()
        )


def test_train_mlm_finetune_scbert_with_new_fields(
    pl_data_module_panglao_geneformer,
    pl_data_module_mock_data_seq_cls,
    geneformer_gene2vec_fields: list[config.FieldInfo],
    gene2vec_fields: list[config.FieldInfo],
):
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer_config = config.TrainerConfig(
            losses=[{"field_name": "genes", "name": "cross_entropy", "weight": 1}]
        )

        model_config = config.SCBertConfig(
            batch_size=pl_data_module_panglao_geneformer.batch_size,
            fields=geneformer_gene2vec_fields,
            num_attention_heads=2,
            num_hidden_layers=2,
            intermediate_size=128,
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
        pl_trainer = make_trainer_for_task(task_config)
        mlm_training_module = MLMTrainingModule(
            model_config, trainer_config, pl_data_module_panglao_geneformer.tokenizer
        )
        train(
            pl_trainer,
            pl_data_module=pl_data_module_panglao_geneformer,
            pl_module=mlm_training_module,
            task_config=task_config,
        )

        ckpt_path = (
            task_config.default_root_dir
            + "/lightning_logs/version_0/checkpoints/epoch=0-step=3.ckpt"
        )
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
            batch_size=pl_data_module_mock_data_seq_cls.batch_size,
            checkpoint=ckpt_path,
            fields=gene2vec_fields,
            label_columns=pl_data_module_mock_data_seq_cls.label_columns,
            num_attention_heads=2,
            num_hidden_layers=2,
            intermediate_size=128,
        )
        finetune_trainer_config = config.TrainerConfig(
            losses=[{"label_column_name": "celltype"}]
        )
        seq_cls_training_module = SequenceClassificationTrainingModule(
            model_config,
            finetune_trainer_config,
            label_dict=pl_data_module_mock_data_seq_cls.label_dict,
        )
        finetune_pl_trainer = make_trainer_for_task(finetune_task_config)
        train(
            finetune_pl_trainer,
            pl_data_module=pl_data_module_mock_data_seq_cls,
            pl_module=seq_cls_training_module,
            task_config=finetune_task_config,
        )
        finetune_ckpt_path = (
            finetune_task_config.default_root_dir
            + "/lightning_logs/version_0/checkpoints/epoch=0-step=3.ckpt"
        )
        finetune_model = torch.load(finetune_ckpt_path, weights_only=False)
        assert (
            "model.base_model.embeddings.expressions_embeddings.weight"
            in finetune_model["state_dict"].keys()
        )


def test_train_seq_cls_with_regression_output(gene2vec_fields):
    from bmfm_targets.datasets import sciplex3
    from bmfm_targets.tokenization import load_tokenizer

    tokenizer = load_tokenizer("gene2vec")
    label_columns = [
        config.LabelColumnInfo(
            label_column_name="proliferation_index", is_regression_label=True
        )
    ]
    dm = sciplex3.SciPlex3DataModule(
        tokenizer=tokenizer,
        fields=gene2vec_fields,
        label_columns=label_columns,
        dataset_kwargs={
            "data_dir": SciPlex3Paths.root,
            "split_column": "split_random",
            "transforms": [],
        },
        collation_strategy="sequence_classification",
        shuffle=True,
        batch_size=2,
        limit_dataset_samples={"train": 6, "dev": 1},
        transform_datasets=False,
    )
    dm.setup("fit")
    helpers.update_label_columns(dm.label_columns, dm.label_dict)
    trainer_config = config.TrainerConfig(
        losses=[{"label_column_name": "proliferation_index"}],
        batch_prediction_behavior="track",
    )
    model_config = config.SCNystromformerConfig(
        fields=gene2vec_fields,
        label_columns=dm.label_columns,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        num_landmarks=2,
        max_position_embeddings=dm.max_length,
        hidden_size=32,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = config.TrainingTaskConfig(
            accelerator="cpu",
            default_root_dir=tmpdir,
            max_epochs=1,
            max_steps=3,
            val_check_interval=3,
            enable_progress_bar=False,
            gradient_clip_val=0.5,
            precision="64",
            enable_model_summary=False,
            enable_checkpointing=True,
            callbacks=[],
        )
        pl_trainer = make_trainer_for_task(task_config)
        seq_cls_training_module = SequenceClassificationTrainingModule(
            model_config,
            trainer_config,
            label_dict=dm.label_dict,
        )
        train(
            pl_trainer,
            pl_data_module=dm,
            pl_module=seq_cls_training_module,
            task_config=task_config,
        )


def test_train_seq_cls_with_gradient_reversal_layer(gene2vec_fields):
    from bmfm_targets.datasets import sciplex3
    from bmfm_targets.tokenization import load_tokenizer

    tokenizer = load_tokenizer("gene2vec")
    label_columns = [
        config.LabelColumnInfo(
            label_column_name="proliferation_index", is_regression_label=True
        ),
        config.LabelColumnInfo(
            label_column_name="batch", gradient_reversal_coefficient=0.5
        ),
    ]
    dm = sciplex3.SciPlex3DataModule(
        tokenizer=tokenizer,
        fields=gene2vec_fields,
        label_columns=label_columns,
        dataset_kwargs={
            "data_dir": SciPlex3Paths.root,
            "split_column": "split_random",
            "transforms": [],
        },
        collation_strategy="sequence_classification",
        shuffle=True,
        batch_size=2,
        limit_dataset_samples={"train": 6, "dev": 1},
        transform_datasets=False,
    )
    dm.setup("fit")
    helpers.update_label_columns(dm.label_columns, dm.label_dict)
    trainer_config = config.TrainerConfig(
        losses=[
            {"label_column_name": "proliferation_index"},
            {"label_column_name": "batch"},
        ],  # understood to be grl from label_columns
        batch_prediction_behavior="track",
    )
    model_config = config.SCBertConfig(
        fields=gene2vec_fields,
        label_columns=dm.label_columns,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        hidden_size=32,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = config.TrainingTaskConfig(
            accelerator="cpu",
            default_root_dir=tmpdir,
            max_epochs=1,
            max_steps=3,
            val_check_interval=3,
            enable_progress_bar=False,
            gradient_clip_val=0.5,
            precision="64",
            enable_model_summary=False,
            enable_checkpointing=True,
            callbacks=[],
        )
        pl_trainer = make_trainer_for_task(task_config)
        train_run(
            pl_trainer,
            model_config=model_config,
            data_module=dm,
            trainer_config=trainer_config,
            task_config=task_config,
        )
        assert pl_trainer.model.prediction_df.keys() == {"batch", "proliferation_index"}
