import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytorch_lightning as pl
import torch.nn as nn
import torch.testing as tt
from scanpy import read_h5ad

from bmfm_targets import config
from bmfm_targets.config import LabelColumnInfo
from bmfm_targets.datasets import sciplex3
from bmfm_targets.models import register_configs_and_models
from bmfm_targets.models.predictive import scbert
from bmfm_targets.models.predictive.multitask import (
    ClassificationHead,
    MultiTaskClassifier,
)
from bmfm_targets.tasks.task_utils import make_trainer_for_task, predict_run
from bmfm_targets.tests import helpers
from bmfm_targets.tokenization import load_tokenizer
from bmfm_targets.training.losses import CrossEntropyObjective, LossTask, get_loss_tasks
from bmfm_targets.training.modules import SequenceClassificationTrainingModule

register_configs_and_models()


@pytest.fixture()
def pl_sciplex3_dm(gene2vec_unmasked_fields):
    label_columns = ["target", "cell_type"]
    dm = sciplex3.SciPlex3DataModule(
        dataset_kwargs={
            "data_dir": helpers.SciPlex3Paths.root,
            "split_column": "split_random",
            "label_columns": label_columns,
        },
        tokenizer=load_tokenizer("gene2vec"),
        fields=gene2vec_unmasked_fields,
        batch_size=3,
        max_length=8,
        pad_to_multiple_of=2,
        collation_strategy="sequence_classification",
        limit_dataset_samples=3,
    )
    dm.setup()
    return dm


def test_model_construction(gene2vec_unmasked_fields):
    model_config = config.SCBertConfig(
        fields=gene2vec_unmasked_fields,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        hidden_size=32,
    )
    base_scbert = scbert.modeling_scbert.SCBertModel(
        model_config, add_pooling_layer=True
    )
    label_columns = [
        config.LabelColumnInfo(
            label_column_name="a", n_unique_values=4, task_group="drug"
        ),
        config.LabelColumnInfo(
            label_column_name="b", n_unique_values=5, task_group="drug"
        ),
        config.LabelColumnInfo(
            label_column_name="c", n_unique_values=6, task_group="cell"
        ),
        config.LabelColumnInfo(
            label_column_name="d", n_unique_values=7, task_group="cell"
        ),
    ]
    losses = [
        {"label_column_name": "a"},
        {"label_column_name": "b"},
        {"label_column_name": "c"},
        {"label_column_name": "d"},
    ]

    loss_tasks = get_loss_tasks(losses, label_columns=label_columns)
    model = MultiTaskClassifier(base_scbert, loss_tasks)
    assert {*model.shared_latents.keys()} == {"drug", "cell"}
    assert {*model.classifiers.keys()} == {"a", "b", "c", "d"}
    assert None not in model.shared_latent_map

    label_columns = [
        config.LabelColumnInfo(
            label_column_name="a", n_unique_values=4, task_group="drug"
        ),
        config.LabelColumnInfo(
            label_column_name="b", n_unique_values=5, task_group="drug"
        ),
        config.LabelColumnInfo(label_column_name="c", n_unique_values=6),
        config.LabelColumnInfo(
            label_column_name="d", n_unique_values=7, task_group="cell"
        ),
    ]
    loss_tasks = get_loss_tasks(losses, label_columns=label_columns)
    model = MultiTaskClassifier(base_scbert, loss_tasks)
    assert {*model.shared_latents.keys()} == {"drug", "cell"}
    assert {*model.classifiers.keys()} == {"a", "b", "c", "d"}
    assert None in model.shared_latent_map

    label_columns = [
        config.LabelColumnInfo(label_column_name="a", n_unique_values=4),
        config.LabelColumnInfo(label_column_name="b", n_unique_values=5),
    ]
    losses = [{"label_column_name": "a"}, {"label_column_name": "b"}]
    loss_tasks = get_loss_tasks(losses, label_columns=label_columns)
    model = MultiTaskClassifier(base_scbert, loss_tasks)
    assert {*model.shared_latents.keys()} == set()
    assert {*model.classifiers.keys()} == {"a", "b"}
    assert None in model.shared_latent_map

    label_columns = [
        config.LabelColumnInfo(label_column_name="a", n_unique_values=4),
    ]
    losses = [{"label_column_name": "a"}]
    loss_tasks = get_loss_tasks(losses, label_columns=label_columns)
    model = MultiTaskClassifier(base_scbert, loss_tasks)
    assert {*model.shared_latents.keys()} == set()
    assert {*model.classifiers.keys()} == {"a"}
    assert None in model.shared_latent_map

    label_columns = [
        config.LabelColumnInfo(
            label_column_name="a", n_unique_values=4, classifier_depth=1
        )
    ]
    losses = [{"label_column_name": "a"}]
    loss_tasks = get_loss_tasks(losses, label_columns=label_columns)
    model = MultiTaskClassifier(base_scbert, loss_tasks)
    assert isinstance(model.classifiers["a"], nn.Linear)

    label_columns = [
        config.LabelColumnInfo(
            label_column_name="a", n_unique_values=4, classifier_depth=2
        )
    ]
    losses = [{"label_column_name": "a"}]
    loss_tasks = get_loss_tasks(losses, label_columns=label_columns)
    model = MultiTaskClassifier(base_scbert, loss_tasks)
    assert isinstance(model.classifiers["a"], ClassificationHead)

    label_columns = [
        config.LabelColumnInfo(
            label_column_name="a", n_unique_values=4, classifier_depth=3
        )
    ]
    losses = [{"label_column_name": "a"}]
    loss_tasks = get_loss_tasks(losses, label_columns=label_columns)
    with pytest.raises(NotImplementedError):
        model = MultiTaskClassifier(base_scbert, loss_tasks)


def test_default_classifier_depth_derives_from_pooling(gene2vec_unmasked_fields):
    model_config = config.SCBertConfig(
        fields=gene2vec_unmasked_fields,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        hidden_size=32,
    )
    label_columns = [
        config.LabelColumnInfo(
            label_column_name="a", n_unique_values=4, classifier_depth=1
        )
    ]
    losses = [{"label_column_name": "a"}]
    loss_tasks = get_loss_tasks(losses, label_columns=label_columns)

    base_scbert_no_pooling = scbert.modeling_scbert.SCBertModel(
        model_config, add_pooling_layer=False
    )
    label_columns = [
        config.LabelColumnInfo(
            label_column_name="a", n_unique_values=4, classifier_depth=1
        )
    ]
    losses = [{"label_column_name": "a"}]
    loss_tasks = get_loss_tasks(losses, label_columns=label_columns)
    model = MultiTaskClassifier(base_scbert_no_pooling, loss_tasks)
    assert isinstance(model.classifiers["a"], nn.Linear)

    label_columns = [
        config.LabelColumnInfo(
            label_column_name="a", n_unique_values=4, classifier_depth=2
        )
    ]
    losses = [{"label_column_name": "a"}]
    loss_tasks = get_loss_tasks(losses, label_columns=label_columns)
    model = MultiTaskClassifier(base_scbert_no_pooling, loss_tasks)
    assert isinstance(model.classifiers["a"], ClassificationHead)


def test_add_classification_head_to_model(gene2vec_unmasked_fields):
    label_columns = [
        config.LabelColumnInfo(label_column_name="target"),
        config.LabelColumnInfo(label_column_name="pathway", task_group="drug"),
        config.LabelColumnInfo(label_column_name="product_name", task_group="drug"),
        config.LabelColumnInfo(label_column_name="cell_type", task_group="cell"),
        config.LabelColumnInfo(
            label_column_name="proliferation_index",
            is_regression_label=True,
            task_group="cell",
        ),
    ]
    dm = sciplex3.SciPlex3DataModule(
        dataset_kwargs={
            "data_dir": helpers.SciPlex3Paths.root,
            "split_column": "split_random",
        },
        tokenizer=load_tokenizer("gene2vec"),
        fields=gene2vec_unmasked_fields,
        label_columns=label_columns,
        batch_size=7,
        max_length=16,
        mlm=False,
    )
    dm.setup()
    helpers.update_label_columns(label_columns, dm.label_dict)
    model_config = config.SCBertConfig(
        fields=gene2vec_unmasked_fields,
        label_columns=dm.label_columns,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        hidden_size=32,
    )
    base_scbert = scbert.modeling_scbert.SCBertModel(
        model_config, add_pooling_layer=True
    )

    losses = [
        {"label_column_name": "target"},
        {"label_column_name": "pathway"},
        {"label_column_name": "product_name"},
        {"label_column_name": "cell_type"},
        {"label_column_name": "proliferation_index"},
    ]
    loss_tasks = get_loss_tasks(losses, label_columns=dm.label_columns)

    model = MultiTaskClassifier(base_scbert, loss_tasks)

    assert model.shared_latents["drug"].weight.shape == (
        model_config.hidden_size,
        model_config.hidden_size,
    )

    for batch in dm.train_dataloader():
        output = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        assert {*output.logits.keys()} == {
            "target",
            "pathway",
            "product_name",
            "cell_type",
            "proliferation_index",
        }
        assert output.logits["proliferation_index"].shape[1] == 1
        break


def test_seq_cls_module_with_multitask(gene2vec_unmasked_fields):
    label_columns = [
        config.LabelColumnInfo(label_column_name="target"),
        config.LabelColumnInfo(label_column_name="pathway", task_group="drug"),
        config.LabelColumnInfo(label_column_name="product_name", task_group="drug"),
        config.LabelColumnInfo(label_column_name="cell_type", task_group="cell"),
        config.LabelColumnInfo(
            label_column_name="proliferation_index",
            is_regression_label=True,
            task_group="cell",
        ),
    ]
    dm = sciplex3.SciPlex3DataModule(
        dataset_kwargs={
            "data_dir": helpers.SciPlex3Paths.root,
            "split_column": "split_random",
        },
        tokenizer=load_tokenizer("gene2vec"),
        fields=gene2vec_unmasked_fields,
        label_columns=label_columns,
        batch_size=7,
        max_length=16,
        collation_strategy="sequence_classification",
        limit_dataset_samples={"train": 21, "dev": 7},
    )
    dm.setup()
    helpers.update_label_columns(label_columns, dm.label_dict)
    model_config = config.SCBertConfig(
        fields=gene2vec_unmasked_fields,
        label_columns=label_columns,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        hidden_size=32,
    )
    losses = [
        {"label_column_name": "target"},
        {"label_column_name": "pathway"},
        {"label_column_name": "product_name"},
        {"label_column_name": "cell_type"},
        {"label_column_name": "proliferation_index"},
    ]

    loss_tasks = get_loss_tasks(losses, label_columns=label_columns)
    trainer_config = config.TrainerConfig(losses=loss_tasks)
    pl_module = SequenceClassificationTrainingModule(
        model_config, trainer_config, label_dict=dm.label_dict
    )

    trainer = pl.Trainer(max_epochs=1, accelerator="cpu", precision=64)
    trainer.fit(pl_module, dm)


def test_can_load_training_module_from_ckpt(
    sciplex3_seqcls_mt_model_and_ckpt,
):
    trained_model, ckpt_path = sciplex3_seqcls_mt_model_and_ckpt
    loaded_model = SequenceClassificationTrainingModule.load_from_checkpoint(
        ckpt_path, weights_only=False
    )
    loaded_model = loaded_model.model
    tt.assert_close(loaded_model.state_dict(), trained_model.state_dict())


def test_ckpt_load_with_model_config_checkpoint(
    sciplex3_seqcls_mt_model_and_ckpt, pl_sciplex3_dm
):
    trained_model, ckpt_path = sciplex3_seqcls_mt_model_and_ckpt
    loss_tasks = trained_model.loss_tasks
    model_config = trained_model.config
    model_config_with_ckpt = helpers.make_model_config_with_ckpt(
        model_config, ckpt_path
    )
    loss_tasks = [
        LossTask.from_label(
            l.label_column.label_column_name, objective=CrossEntropyObjective()
        )
        for l in loss_tasks
    ]

    mt_pl_module_from_ckpt = SequenceClassificationTrainingModule(
        model_config_with_ckpt,
        config.TrainerConfig(losses=loss_tasks),
        label_dict=pl_sciplex3_dm.label_dict,
    )

    loaded_model_pt = mt_pl_module_from_ckpt.model
    tt.assert_close(loaded_model_pt.state_dict(), trained_model.state_dict())


def test_updated_model_config_params_are_used_when_loading_ckpt(
    sciplex3_seqcls_mt_model_and_ckpt, pl_sciplex3_dm
):
    trained_model, ckpt_path = sciplex3_seqcls_mt_model_and_ckpt
    loss_tasks = trained_model.loss_tasks
    model_config = trained_model.config
    new_model_config = helpers.make_model_config_with_ckpt(model_config, ckpt_path)
    # new model config should change something
    new_param_val = 0.12345
    new_model_config.hidden_dropout_prob = new_param_val
    loss_task_dicts = [
        {"label_column_name": l.label_column.label_column_name} for l in loss_tasks
    ]
    loss_tasks_new = get_loss_tasks(
        loss_task_dicts, label_columns=new_model_config.label_columns
    )

    mt_pl_module_from_ckpt = SequenceClassificationTrainingModule(
        new_model_config,
        config.TrainerConfig(losses=loss_tasks_new),
        label_dict=pl_sciplex3_dm.label_dict,
    )

    loaded_model_pt = mt_pl_module_from_ckpt.model
    assert loaded_model_pt.config.hidden_dropout_prob == new_param_val


def test_inference_after_train(sciplex3_seqcls_mt_model_and_ckpt, pl_sciplex3_dm):
    trained_model, ckpt_path = sciplex3_seqcls_mt_model_and_ckpt
    dm = pl_sciplex3_dm
    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = config.PredictTaskConfig(
            default_root_dir=tmpdir,
            precision="32",
            accelerator="cpu",
            output_embeddings=True,
            output_predictions=True,
            checkpoint=ckpt_path,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[],
        )
        pl_trainer = make_trainer_for_task(task_config)
        predict_run(
            pl_trainer,
            task_config=task_config,
            data_module=dm,
            trainer_config=None,
            model_config=None,
        )

        output_path = Path(task_config.default_root_dir)
        embeddings_path = output_path / "embeddings.csv"
        predictions_path = output_path / "predictions.csv"

        embeddings_df = pd.read_csv(embeddings_path, header=None, index_col=0)
        predictions_df = pd.read_csv(predictions_path, index_col=0)

        assert embeddings_df.shape == (3, 16)
        assert all(i in dm.label_dict["cell_type"] for i in predictions_df["cell_type"])

        logits_path = output_path / "logits.csv"
        probabilities_path = output_path / "probabilities.csv"

        logits_df = pd.read_csv(logits_path, index_col=0)
        probabilities_df = pd.read_csv(probabilities_path, index_col=0)

        assert logits_df.shape == (
            3,
            91,
        )  # 91 comes from 88 for "target" and 3 for "cell_type"
        assert probabilities_df.shape == (3, 91)
        assert (
            round(probabilities_df.iloc[0, :].sum() * 10000) == 20000
        )  # because we have two labels


def test_seq_cls_module_with_partial_NaN_labels_multitask(gene2vec_unmasked_fields):
    ad = read_h5ad(
        helpers.SciPlex3Paths.root
        / "h5ad"
        / sciplex3.SciPlex3Dataset.source_h5ad_file_name
    )
    half_zero = np.random.random(size=(ad.shape[0])) < 0.5

    ad.obs["sometimes_nan_1"] = ad.obs["cell_type"].copy()
    ad.obs["sometimes_nan_2"] = ad.obs["cell_type"].copy()

    ad.obs["sometimes_nan_1"].loc[half_zero] = np.nan
    ad.obs["sometimes_nan_2"].loc[~half_zero] = np.nan

    assert sum(ad.obs["cell_type"].isna()) == 0
    assert sum(ad.obs["sometimes_nan_1"].isna()) > 0
    assert sum(ad.obs["sometimes_nan_2"].isna()) > 0
    assert (
        sum(ad.obs["sometimes_nan_1"].isna() | ad.obs["sometimes_nan_2"].isna())
        == ad.shape[0]
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        modified_fname = Path(tmpdirname) / "sciplex3_with_nans.h5ad"
        ad.write_h5ad(modified_fname)
        label_columns = [
            LabelColumnInfo("sometimes_nan_1"),
            LabelColumnInfo("sometimes_nan_2"),
        ]
        dm = sciplex3.SciPlex3DataModule(
            dataset_kwargs={
                "source_h5ad_file_name": modified_fname,
                "split_column": "split_random",
                "label_dict_path": Path(tmpdirname) / "labels.json",
            },
            tokenizer=load_tokenizer("gene2vec"),
            fields=gene2vec_unmasked_fields,
            label_columns=label_columns,
            batch_size=7,
            max_length=14,
            pad_to_multiple_of=2,
            collation_strategy="sequence_classification",
        )
        dm.setup()
        helpers.update_label_columns(dm.label_columns, dm.label_dict)
        model_config = config.SCBertConfig(
            fields=gene2vec_unmasked_fields,
            label_columns=dm.label_columns,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            hidden_size=32,
        )

        losses = [
            {"label_column_name": "sometimes_nan_1"},
            {"label_column_name": "sometimes_nan_2"},
        ]

        loss_tasks = get_loss_tasks(losses, label_columns=label_columns)
        trainer_config = config.TrainerConfig(losses=loss_tasks)
        pl_module = SequenceClassificationTrainingModule(
            model_config, trainer_config, label_dict=dm.label_dict
        )

        trainer = pl.Trainer(max_epochs=1, default_root_dir=tmpdirname)
        trainer.fit(pl_module, dm)


def test_partial_NaN_labels_loss(gene2vec_unmasked_fields):
    ad = read_h5ad(
        helpers.SciPlex3Paths.root
        / "h5ad"
        / sciplex3.SciPlex3Dataset.source_h5ad_file_name
    )
    half_zero = np.random.random(size=(ad.shape[0])) < 0.5

    ad.obs["sometimes_nan_1"] = ad.obs["cell_type"].copy()
    ad.obs["sometimes_nan_2"] = ad.obs["cell_type"].copy()

    ad.obs["sometimes_nan_1"].loc[half_zero] = np.nan
    ad.obs["sometimes_nan_2"].loc[~half_zero] = np.nan

    assert sum(ad.obs["cell_type"].isna()) == 0
    assert sum(ad.obs["sometimes_nan_1"].isna()) > 0
    assert sum(ad.obs["sometimes_nan_2"].isna()) > 0
    assert (
        sum(ad.obs["sometimes_nan_1"].isna() | ad.obs["sometimes_nan_2"].isna())
        == ad.shape[0]
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        modified_fname = Path(tmpdirname) / "sciplex3_with_nans.h5ad"
        ad.write_h5ad(modified_fname)
        label_columns = [
            LabelColumnInfo("cell_type"),
            LabelColumnInfo("sometimes_nan_1"),
            LabelColumnInfo("sometimes_nan_2"),
        ]
        dm = sciplex3.SciPlex3DataModule(
            dataset_kwargs={
                "source_h5ad_file_name": modified_fname,
                "split_column": "split_random",
                "label_dict_path": Path(tmpdirname) / "label_columns.json",
            },
            tokenizer=load_tokenizer("gene2vec"),
            fields=gene2vec_unmasked_fields,
            label_columns=label_columns,
            batch_size=1,
            max_length=14,
            pad_to_multiple_of=2,
            limit_dataset_samples=100,
            collation_strategy="sequence_classification",
        )
        dm.setup()
        helpers.update_label_columns(dm.label_columns, dm.label_dict)
        model_config = config.SCBertConfig(
            fields=gene2vec_unmasked_fields,
            label_columns=dm.label_columns,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            hidden_size=32,
            hidden_dropout_prob=0,
            classifier_dropout=0,
            attention_probs_dropout_prob=0,
        )

        losses = [
            {"label_column_name": "sometimes_nan_1"},
            {"label_column_name": "sometimes_nan_2"},
        ]

        loss_tasks_nans = get_loss_tasks(losses, label_columns=dm.label_columns)
        trainer_config_nans = config.TrainerConfig(losses=loss_tasks_nans)
        pl_module_nans = SequenceClassificationTrainingModule(
            model_config, trainer_config_nans, label_dict=dm.label_dict
        )

        losses_cell_type = [{"label_column_name": "cell_type"}]
        loss_tasks_cell_type = get_loss_tasks(
            losses_cell_type, label_columns=dm.label_columns
        )
        trainer_config_cell_type = config.TrainerConfig(losses=loss_tasks_cell_type)
        pl_module_cell_type = SequenceClassificationTrainingModule(
            model_config, trainer_config_cell_type, label_dict=dm.label_dict
        )

        pl_module_nans.model.base_model = pl_module_cell_type.model.base_model
        pl_module_nans.model.classifiers[
            "sometimes_nan_1"
        ] = pl_module_cell_type.model.classifiers["cell_type"]
        pl_module_nans.model.classifiers[
            "sometimes_nan_2"
        ] = pl_module_cell_type.model.classifiers["cell_type"]

        # optimizer_nans = pl_module_nans.configure_optimizers()
        # optimizer_cell_type = pl_module_cell_type.configure_optimizers()

        for batch_idx, batch in enumerate(dm.train_dataloader()):
            loss_nan = pl_module_nans.training_step(batch, batch_idx)
            loss_cell_type = pl_module_cell_type.training_step(batch, batch_idx)
            assert loss_nan == loss_cell_type
