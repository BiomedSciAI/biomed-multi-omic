import tempfile

import pytest
import torch

from bmfm_targets import config
from bmfm_targets.datasets import dnaseq
from bmfm_targets.datasets.SNPdb.streaming_snp_dataset import StreamingSNPdbDataModule
from bmfm_targets.tasks.task_utils import make_trainer_for_task, train
from bmfm_targets.training.losses import CrossEntropyObjective, LossTask, MSEObjective
from bmfm_targets.training.modules import SequenceClassificationTrainingModule
from bmfm_targets.training.modules.masked_language_modeling import MLMTrainingModule


@pytest.mark.skip(reason="skip for open source integration")
@pytest.mark.parametrize(
    "pl_data_module_dnaseq_fixture",
    [
        pytest.param("pl_data_module_dnaseq_lenti_mpra"),
        pytest.param("pl_data_module_dnaseq_drosophila_enhancer"),
    ],
)
def test_train_dnaseq_seq_cls_regression_tasks(
    request,
    pl_data_module_dnaseq_fixture,
):
    pl_data_module_dnaseq = request.getfixturevalue(pl_data_module_dnaseq_fixture)
    ds = pl_data_module_dnaseq.get_dataset_instance()
    trainer_config = config.TrainerConfig(
        losses=[
            LossTask.from_label(
                ds.regression_label_columns[0],
                objective=MSEObjective(),
                metrics=[{"name": "pcc"}],
            )
        ]
    )

    model_config = config.SCBertConfig(
        batch_size=pl_data_module_dnaseq.batch_size,
        fields=pl_data_module_dnaseq.fields,
        label_columns=pl_data_module_dnaseq.label_columns,
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
            accumulate_grad_batches=1,
            val_check_interval=1,
            gradient_clip_val=0.5,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[],
        )
        seq_cls_training_module = SequenceClassificationTrainingModule(
            model_config,
            trainer_config,
            label_dict=pl_data_module_dnaseq.label_dict,
        )
        pl_trainer = make_trainer_for_task(task_config)
        train(
            pl_trainer,
            pl_data_module=pl_data_module_dnaseq,
            pl_module=seq_cls_training_module,
            task_config=task_config,
        )


@pytest.mark.skip(reason="skip for open source integration")
@pytest.mark.parametrize(
    "pl_data_module_dnaseq_fixture",
    [
        pytest.param("pl_data_module_dnaseq_core_promoter"),
        pytest.param("pl_data_module_dnaseq_promoter"),
        pytest.param("pl_data_module_dnaseq_splice"),
        pytest.param("pl_data_module_dnaseq_covid"),
        pytest.param("pl_data_module_dnaseq_chromatin"),
        pytest.param("pl_data_module_dnaseq_epigenetic_marks"),
        pytest.param("pl_data_module_dnaseq_transcription_factor"),
    ],
)
def test_train_dnaseq_seq_cls_classification_tasks(
    request,
    pl_data_module_dnaseq_fixture,
):
    pl_data_module_dnaseq = request.getfixturevalue(pl_data_module_dnaseq_fixture)
    ds = pl_data_module_dnaseq.get_dataset_instance()
    trainer_config = config.TrainerConfig(
        losses=[
            LossTask.from_label(
                ds.label_columns[0],
                objective=CrossEntropyObjective(),
                metrics=[{"name": "f1"}],
            )
        ]
    )

    model_config = config.SCBertConfig(
        batch_size=pl_data_module_dnaseq.batch_size,
        fields=pl_data_module_dnaseq.fields,
        label_columns=pl_data_module_dnaseq.label_columns,
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
            accumulate_grad_batches=1,
            val_check_interval=1,
            gradient_clip_val=0.5,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[],
        )
        seq_cls_training_module = SequenceClassificationTrainingModule(
            model_config,
            trainer_config,
            label_dict=pl_data_module_dnaseq.label_dict,
        )
        pl_trainer = make_trainer_for_task(task_config)
        train(
            pl_trainer,
            pl_data_module=pl_data_module_dnaseq,
            pl_module=seq_cls_training_module,
            task_config=task_config,
        )


@pytest.mark.skip(reason="skip for open source integration")
@pytest.mark.usefixtures("_convert_raw_to_lit")
def test_pretrain_scbert_snpdb_and_finetune_lenti_mpra(
    streaming_snpdb_parameters,
    snp2vec_fields: list[config.FieldInfo],
    pl_data_module_dnaseq_lenti_mpra: dnaseq.DNASeqMPRADataModule,
):
    datamodule = StreamingSNPdbDataModule(**streaming_snpdb_parameters)
    datamodule.prepare_data()
    datamodule.setup()
    model_config = config.SCBertConfig(
        fields=snp2vec_fields,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=32,
        hidden_size=16,
    )
    losses = [LossTask.from_field("dna_chunks", objective=CrossEntropyObjective())]
    trainer_config = config.TrainerConfig(losses=losses)

    with tempfile.TemporaryDirectory() as tmpdir:
        pretrain_task_config = config.TrainingTaskConfig(
            default_root_dir=tmpdir,
            max_epochs=1,
            max_steps=3,
            accelerator="cpu",
            val_check_interval=3,
            gradient_clip_val=0.5,
            precision="32",
            enable_model_summary=False,
            enable_progress_bar=False,
            enable_checkpointing=True,
            resume_training_from_ckpt=False,
            detect_anomaly=True,
            callbacks=[],
        )

        mlm_training_module = MLMTrainingModule(
            model_config, trainer_config, datamodule.tokenizer
        )
        pl_trainer = make_trainer_for_task(pretrain_task_config)
        train(
            pl_trainer,
            pl_data_module=datamodule,
            pl_module=mlm_training_module,
            task_config=pretrain_task_config,
        )

        pretrain_ckpt_path = pl_trainer.checkpoint_callback._last_checkpoint_saved
        ckpt_dict = torch.load(pretrain_ckpt_path, weights_only=False)

        saved_model_config = ckpt_dict["hyper_parameters"]["model_config"]
        assert isinstance(saved_model_config, config.SCBertConfig)

        saved_trainer_config = ckpt_dict["hyper_parameters"]["trainer_config"]
        assert isinstance(saved_trainer_config, config.TrainerConfig)
        trainer_config = config.TrainerConfig(
            losses=[LossTask.from_label("mean_value", objective=MSEObjective())]
        )
        model_config = config.SCBertConfig(
            batch_size=pl_data_module_dnaseq_lenti_mpra.batch_size,
            fields=pl_data_module_dnaseq_lenti_mpra.fields,
            label_columns=pl_data_module_dnaseq_lenti_mpra.label_columns,
            num_attention_heads=2,
            num_hidden_layers=2,
            intermediate_size=64,
            hidden_size=32,
        )
        seq_cls_training_module = SequenceClassificationTrainingModule(
            model_config,
            trainer_config,
            label_dict=pl_data_module_dnaseq_lenti_mpra.label_dict,
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
        pl_trainer = make_trainer_for_task(finetune_task_config)
        train(
            pl_trainer,
            pl_data_module=pl_data_module_dnaseq_lenti_mpra,
            pl_module=seq_cls_training_module,
            task_config=finetune_task_config,
        )

        finetune_ckpt_path = pl_trainer.checkpoint_callback._last_checkpoint_saved
        finetune_model = torch.load(finetune_ckpt_path, weights_only=False)[
            "state_dict"
        ]
        pretrain_model = torch.load(pretrain_ckpt_path, weights_only=False)[
            "state_dict"
        ]
        assert finetune_model.keys() != pretrain_model.keys()
        assert (
            "model.base_model.embeddings.dna_chunks_embeddings.weight"
            in finetune_model.keys()
        )
