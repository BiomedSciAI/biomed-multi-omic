import tempfile

import pytest
import torch

from bmfm_targets import config
from bmfm_targets.datasets.SNPdb.streaming_snp_dataset import StreamingHiCDataModule
from bmfm_targets.datasets.zheng68k import Zheng68kDataModule

# test_ function args are pytest fixtures defined in conftest.py`
from bmfm_targets.tasks.task_utils import make_trainer_for_task, train
from bmfm_targets.training.modules import (
    MultiTaskTrainingModule,
    SequenceClassificationTrainingModule,
)

from .helpers import default_mlm_losses_from_fields


def test_module_saves_all_hyperparameters(mock_data_multitask_ckpt):
    ckpt_dict = torch.load(mock_data_multitask_ckpt, weights_only=False)

    saved_model_config = ckpt_dict["hyper_parameters"]["model_config"]
    assert isinstance(saved_model_config, config.SCBertConfig)

    saved_trainer_config = ckpt_dict["hyper_parameters"]["trainer_config"]
    assert isinstance(saved_trainer_config, config.TrainerConfig)


def test_train_multitask_performer(
    pl_data_module_mock_data_multitask: Zheng68kDataModule,
):
    trainer_config = config.TrainerConfig(
        losses=[
            {"label_column_name": "celltype"},
        ]
    )

    model_config = config.SCPerformerConfig(
        batch_size=pl_data_module_mock_data_multitask.batch_size,
        fields=pl_data_module_mock_data_multitask.fields,
        label_columns=pl_data_module_mock_data_multitask.label_columns,
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

        multitask_training_module = MultiTaskTrainingModule(
            model_config,
            trainer_config,
            pl_data_module_mock_data_multitask.tokenizer,
            pl_data_module_mock_data_multitask.label_dict,
        )
        pl_trainer = make_trainer_for_task(task_config)
        train(
            pl_trainer,
            pl_data_module=pl_data_module_mock_data_multitask,
            pl_module=multitask_training_module,
            task_config=task_config,
        )


def test_train_multitask_nystromformer(pl_data_module_mock_data_multitask):
    trainer_config = config.TrainerConfig(
        losses=default_mlm_losses_from_fields(pl_data_module_mock_data_multitask.fields)
    )
    model_config = config.SCNystromformerConfig(
        batch_size=pl_data_module_mock_data_multitask.batch_size,
        fields=pl_data_module_mock_data_multitask.fields,
        label_columns=pl_data_module_mock_data_multitask.label_columns,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        num_landmarks=2,
        max_position_embeddings=pl_data_module_mock_data_multitask.max_length,
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
            pl_data_module_mock_data_multitask.tokenizer,
            pl_data_module_mock_data_multitask.label_dict,
        )
        pl_trainer = make_trainer_for_task(task_config)
        train(
            pl_trainer,
            pl_data_module=pl_data_module_mock_data_multitask,
            pl_module=multitask_training_module,
            task_config=task_config,
        )


def test_train_multitask_finetune_scbert(
    pl_data_module_mock_data_seq_cls,
    pl_data_module_mock_data_multitask,
    gene2vec_unmasked_fields: list[config.FieldInfo],
):
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer_config = config.TrainerConfig(
            losses=[
                {"label_column_name": "celltype"},
                {"field_name": "expressions"},
            ]
        )

        model_config = config.SCBertConfig(
            batch_size=pl_data_module_mock_data_multitask.batch_size,
            fields=pl_data_module_mock_data_multitask.fields,
            label_columns=pl_data_module_mock_data_multitask.label_columns,
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
        multitask_training_module = MultiTaskTrainingModule(
            model_config, trainer_config, pl_data_module_mock_data_multitask.tokenizer
        )
        pl_trainer = make_trainer_for_task(task_config)
        train(
            pl_trainer,
            pl_data_module=pl_data_module_mock_data_multitask,
            pl_module=multitask_training_module,
            task_config=task_config,
        )

        pretrain_ckpt_path = pl_trainer.checkpoint_callback._last_checkpoint_saved

        pretrain_model = torch.load(pretrain_ckpt_path, weights_only=False)[
            "state_dict"
        ]

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
            label_columns=pl_data_module_mock_data_seq_cls.label_columns,
            checkpoint=pretrain_ckpt_path,
            fields=gene2vec_unmasked_fields,
            num_attention_heads=2,
            num_hidden_layers=2,
            intermediate_size=128,
        )

        trainer_config = config.TrainerConfig(
            losses=[
                {"label_column_name": "celltype"},
            ]
        )
        sequence_classification_training_module = SequenceClassificationTrainingModule(
            model_config,
            trainer_config,
            label_dict=pl_data_module_mock_data_seq_cls.label_dict,
        )
        pl_trainer = make_trainer_for_task(task_config)
        train(
            pl_trainer,
            pl_data_module=pl_data_module_mock_data_seq_cls,
            pl_module=sequence_classification_training_module,
            task_config=finetune_task_config,
        )

        finetune_ckpt_path = pl_trainer.checkpoint_callback._last_checkpoint_saved

        finetune_model = torch.load(finetune_ckpt_path, weights_only=False)[
            "state_dict"
        ]

        assert finetune_model.keys() != pretrain_model.keys()


@pytest.mark.usefixtures("_convert_hic_raw_to_lit")
def test_train_multitask_hic(streaming_hic_parameters):
    pl_data_module_hic_multitask = StreamingHiCDataModule(**streaming_hic_parameters)
    pl_data_module_hic_multitask.prepare_data()
    pl_data_module_hic_multitask.setup()
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer_config = config.TrainerConfig(
            losses=[
                {"label_column_name": "hic_contact"},
                {"field_name": "dna_chunks"},
            ]
        )

        model_config = config.SCBertConfig(
            batch_size=pl_data_module_hic_multitask.batch_size,
            fields=pl_data_module_hic_multitask.fields,
            label_columns=pl_data_module_hic_multitask.label_columns,
            num_attention_heads=2,
            num_hidden_layers=2,
            intermediate_size=128,
            hidden_size=32,
        )

        task_config = config.TrainingTaskConfig(
            default_root_dir=tmpdir + "/pretrain",
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
            model_config, trainer_config, pl_data_module_hic_multitask.tokenizer
        )
        pl_trainer = make_trainer_for_task(task_config)
        train(
            pl_trainer,
            pl_data_module=pl_data_module_hic_multitask,
            pl_module=multitask_training_module,
            task_config=task_config,
        )


def test_multitask_scbert_tracks_field_and_label_batches(
    pl_data_module_mock_data_multitask, mock_clearml_logger
):
    dm0 = pl_data_module_mock_data_multitask
    label_columns = pl_data_module_mock_data_multitask.label_columns + [
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
        dataset_kwargs={"source_h5ad_file_name":"mock_test_data.h5ad"},
        transform_datasets=False,
        collation_strategy="multitask",
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
                {"label_column_name": "celltype"},
                {"label_column_name": "n_counts"},
                {"field_name": "expressions"},
            ],
            batch_prediction_behavior="track",
        )

        model_config = config.SCBertConfig(
            batch_size=dm.batch_size,
            fields=dm.fields,
            label_columns=dm.label_columns,
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
        val_batches = multitask_training_module.split_batch_predictions("validation")
        assert len(val_batches["expressions"]) > 0
        assert len(val_batches["celltype"]) > 0
        n_counts_lt = [
            lt
            for lt in multitask_training_module.loss_tasks
            if lt.output_key == "n_counts"
        ][0]
        assert n_counts_lt.loss_name == "mse"
