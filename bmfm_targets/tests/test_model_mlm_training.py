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
    StreamingSNPdbDataModule,
)
from bmfm_targets.models import register_configs_and_models
from bmfm_targets.models.predictive import (
    MultiTaskClassifier,
)
from bmfm_targets.tasks.task_utils import make_trainer_for_task, train
from bmfm_targets.tests.helpers import (
    get_test_task_config,
    make_model_config_with_ckpt,
)
from bmfm_targets.training.callbacks import InitialCheckpoint
from bmfm_targets.training.modules import MLMTrainingModule

from .helpers import default_mlm_losses_from_fields

register_configs_and_models()


def test_mlm_training_scbert(
    pl_data_module_panglao: PanglaoDBDataModule, gene2vec_fields: list[config.FieldInfo]
):
    model_config = config.SCBertConfig(
        fields=gene2vec_fields,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=32,
        hidden_size=16,
    )
    run_train_checkpoint_test(
        pl_data_module_panglao,
        model_config=model_config,
    )


def test_train_and_initial_checkpoint_save_scbert(
    pl_data_module_panglao: PanglaoDBDataModule, gene2vec_fields: list[config.FieldInfo]
):
    model_config = config.SCBertConfig(
        fields=gene2vec_fields,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        hidden_size=32,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        callback = InitialCheckpoint(
            dirpath=tmpdir,
            filename="initial",
        )
        task_config = get_test_task_config(tmpdir)
        task_config.callbacks = [callback]
        trainer_config = config.TrainerConfig(
            losses=default_mlm_losses_from_fields(gene2vec_fields)
        )
        mlm_training_module = MLMTrainingModule(
            model_config, trainer_config, pl_data_module_panglao.tokenizer
        )
        initial_embeddings = (
            (mlm_training_module.model.scbert.embeddings.genes_embeddings.weight)
            .detach()
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
            torch.load(filepath, weights_only=False)["state_dict"][
                "model.scbert.embeddings.genes_embeddings.weight"
            ]
            .detach()
            .numpy()
        )
        np.testing.assert_allclose(initial_embeddings, save_initial)


def test_mlm_training_performer(
    pl_data_module_panglao: PanglaoDBDataModule, gene2vec_fields: list[config.FieldInfo]
):
    model_config = config.SCPerformerConfig(
        fields=gene2vec_fields,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=32,
        hidden_size=16,
    )
    run_train_checkpoint_test(
        pl_data_module_panglao,
        model_config=model_config,
    )


@pytest.mark.usefixtures("_convert_raw_to_lit")
def test_mlm_training_nystromformer_snpdb(
    streaming_snpdb_parameters,
    snp2vec_fields: list[config.FieldInfo],
):
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
    losses = [{"field_name": "dna_chunks", "name": "cross_entropy", "weight": 1}]
    run_train_checkpoint_test(
        datamodule,
        trainer_config=config.TrainerConfig(losses=losses),
        model_config=model_config,
    )


@pytest.mark.usefixtures("_convert_hic_raw_to_lit")
def test_mlm_training_scbert_hic(
    streaming_hic_parameters,
    snp2vec_fields: list[config.FieldInfo],
):
    datamodule = StreamingHiCDataModule(**streaming_hic_parameters)
    datamodule.prepare_data()
    datamodule.setup()
    model_config = config.SCBertConfig(
        fields=snp2vec_fields,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=32,
        hidden_size=16,
    )
    losses = [{"field_name": "dna_chunks", "name": "cross_entropy", "weight": 1}]
    run_train_checkpoint_test(
        datamodule,
        trainer_config=config.TrainerConfig(losses=losses),
        model_config=model_config,
    )


def test_mlm_training_scnystromformer(
    pl_data_module_panglao: PanglaoDBDataModule, gene2vec_fields: list[config.FieldInfo]
):
    model_config = config.SCNystromformerConfig(
        fields=gene2vec_fields,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=32,
        hidden_size=16,
        num_landmarks=2,
        max_position_embeddings=pl_data_module_panglao.max_length,
    )
    run_train_checkpoint_test(pl_data_module_panglao, model_config=model_config)


def test_mlm_training_scbert_regression(
    pl_data_module_panglao_regression: PanglaoDBDataModule,
    gene2vec_fields_regression_with_tokenization: list[config.FieldInfo],
):
    model_config = config.SCBertConfig(
        fields=gene2vec_fields_regression_with_tokenization,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=32,
        hidden_size=16,
    )
    run_train_checkpoint_test(
        pl_data_module_panglao_regression,
        trainer_config=config.TrainerConfig(
            losses=[
                {"field_name": "genes", "name": "cross_entropy", "weight": 1},
                {"field_name": "expressions", "name": "token_mse", "weight": 1},
            ],
        ),
        model_config=model_config,
    )


def test_train_and_resume_checkpoint_load_scbert_regression(
    pl_data_module_panglao_regression: PanglaoDBDataModule,
    gene2vec_fields_regression_with_tokenization: list[config.FieldInfo],
):
    model_config = config.SCBertConfig(
        fields=gene2vec_fields_regression_with_tokenization,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=32,
        hidden_size=16,
    )
    run_train_resume_checkpoint_test(
        pl_data_module_panglao_regression,
        trainer_config=config.TrainerConfig(
            batch_size=1,
            losses=[
                {"field_name": "genes", "name": "cross_entropy", "weight": 1},
                {"field_name": "expressions", "name": "token_mse", "weight": 1},
            ],
        ),
        model_config=model_config,
    )


def test_mlm_training_scbert_continuous_value_encoder_regression(
    pl_mock_data_mlm_no_binning,
):
    model_config = config.SCBertConfig(
        fields=pl_mock_data_mlm_no_binning.fields,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=32,
        hidden_size=16,
        pad_token_id=2,
    )
    run_train_checkpoint_test(
        pl_mock_data_mlm_no_binning,
        trainer_config=config.TrainerConfig(
            losses=[
                {"field_name": "genes", "name": "cross_entropy", "weight": 1},
                {
                    "field_name": "expressions",
                    "name": "mse",
                    "ignore_zero": True,
                    "weight": 1,
                },
                {"field_name": "expressions", "name": "is_zero_bce", "weight": 1},
            ],
        ),
        model_config=model_config,
    )


def test_train_and_checkpoint_scbert_rda(
    pl_data_module_panglao_rda: PanglaoDBDataModule,
    all_genes_fields_with_rda_regression_masking: list[config.FieldInfo],
    mock_clearml_logger,
):
    model_config = config.SCBertConfig(
        fields=all_genes_fields_with_rda_regression_masking,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=32,
        hidden_size=16,
    )
    run_train_checkpoint_test(
        pl_data_module_panglao_rda,
        trainer_config=config.TrainerConfig(
            batch_size=1,
            losses=[
                {"field_name": "expressions", "name": "token_mse", "weight": 1},
            ],
        ),
        model_config=model_config,
    )


def ensure_shared_parameters_load_for_seq_cls(trained_model, model_config):
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


def ensure_shared_parameters_load_for_multitask(trained_model, model_config):
    tasks = [{"label_column_name": "dummy"}]
    label_output_size_dict = {"dummy": 3}

    mt_from_ckpt = MultiTaskClassifier.from_ckpt(
        model_config.checkpoint,
        loss_tasks=tasks,
        label_output_size_dict=label_output_size_dict,
    )

    base_model_from_trained = getattr(trained_model, model_config.model_type)

    for t, l in zip(
        base_model_from_trained.named_parameters(),
        mt_from_ckpt.base_model.named_parameters(),
    ):
        torch.testing.assert_close(t[1], l[1])


def ensure_all_parameters_load_for_mlm(trained_model, model_config):
    loaded_model = auto.AutoModelForMaskedLM.from_config(model_config)

    for t, l in zip(trained_model.named_parameters(), loaded_model.named_parameters()):
        torch.testing.assert_close(t[1], l[1])


def run_train_resume_checkpoint_test(pl_data_module, trainer_config, model_config):
    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = get_test_task_config(tmpdir)
        mlm_training_module = MLMTrainingModule(
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

        task_config = get_test_task_config(tmpdir, resume_training_from_ckpt=True)
        model_config.checkpoint = ckpt_path
        mlm_training_module = MLMTrainingModule(
            model_config, trainer_config, pl_data_module.tokenizer
        )
        pl_trainer = make_trainer_for_task(task_config)
        train(
            pl_trainer,
            pl_data_module=pl_data_module,
            pl_module=mlm_training_module,
            task_config=task_config,
        )


def run_train_checkpoint_test(
    pl_data_module, model_config: config.SCBertConfig, trainer_config=None
):
    if trainer_config is None:
        trainer_config = config.TrainerConfig(
            losses=default_mlm_losses_from_fields(model_config.fields)
        )
    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = get_test_task_config(tmpdir)
        mlm_training_module = MLMTrainingModule(
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
        # ensure_shared_parameters_load_for_multitask(trained_model, model_config_with_ckpt)
        ckpt_dict = torch.load(ckpt_path, weights_only=False)

        saved_model_config = ckpt_dict["hyper_parameters"]["model_config"]
        assert isinstance(saved_model_config, type(model_config))
        # assert saved_model_config == model_config

        saved_trainer_config = ckpt_dict["hyper_parameters"]["trainer_config"]
        assert isinstance(saved_trainer_config, type(trainer_config))
        # assert saved_trainer_config == trainer_config
