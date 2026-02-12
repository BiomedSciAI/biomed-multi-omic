import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bmfm_targets import config
from bmfm_targets.datasets.sciplex3 import SciPlex3DataModule
from bmfm_targets.datasets.scp1884 import SCP1884DataModule
from bmfm_targets.datasets.zheng68k import Zheng68kDataModule
from bmfm_targets.tasks import task_utils
from bmfm_targets.tests import helpers
from bmfm_targets.tokenization import load_tokenizer
from bmfm_targets.training.losses import CrossEntropyObjective, LossTask
from bmfm_targets.training.metrics.metric_functions import calculate_95_ci
from bmfm_targets.training.modules import SequenceClassificationTrainingModule


def test_zheng68k_predict_no_train(
    pl_data_module_zheng68k_seq_cls: Zheng68kDataModule, gene2vec_fields
):
    data_module = pl_data_module_zheng68k_seq_cls
    trainer_config = config.TrainerConfig(
        losses=[LossTask.from_label("celltype", objective=CrossEntropyObjective())]
    )
    model_config = config.SCBertConfig(
        fields=gene2vec_fields,
        label_columns=data_module.label_columns,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        hidden_size=32,
    )
    training_module = SequenceClassificationTrainingModule(
        model_config, trainer_config, label_dict=data_module.label_dict
    )
    task_config = config.PredictTaskConfig(
        precision="32",
        accelerator="cpu",
        output_embeddings=True,
        output_predictions=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        task_config.default_root_dir = tmpdir
        pl_trainer = task_utils.make_trainer_for_task(task_config)

        results = task_utils.predict(
            pl_trainer,
            pl_module=training_module,
            pl_data_module=pl_data_module_zheng68k_seq_cls,
        )
        assert {*results.keys()} == {
            "embeddings",
            "celltype_predictions",
            "celltype_logits",
            "cell_name",
        }


def test_zheng68k_predict_no_train_including_non_input_fields(
    pl_zheng_mlm_raw_counts: Zheng68kDataModule,
    pl_data_module_zheng68k_seq_cls,
    all_genes_fields_with_rda_regression_masking: list[config.FieldInfo],
):
    model_config = config.SCBertConfig(
        fields=all_genes_fields_with_rda_regression_masking,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=32,
        hidden_size=16,
    )
    data_module = Zheng68kDataModule(
        tokenizer=pl_zheng_mlm_raw_counts.tokenizer,
        fields=all_genes_fields_with_rda_regression_masking,
        data_dir=helpers.Zheng68kPaths.root,
        processed_name=helpers.Zheng68kPaths.raw_counts_name,
        label_columns=pl_data_module_zheng68k_seq_cls.label_columns,
        transform_datasets=False,
        batch_size=2,
        limit_dataset_samples=8,
        mlm=False,
        collation_strategy="sequence_classification",
        max_length=20,
        pad_to_multiple_of=2,
    )
    data_module.prepare_data()
    data_module.setup("predict")

    trainer_config = config.TrainerConfig(
        losses=[LossTask.from_label("celltype", objective=CrossEntropyObjective())]
    )
    model_config = config.SCBertConfig(
        fields=data_module.fields,
        label_columns=pl_data_module_zheng68k_seq_cls.label_columns,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        hidden_size=32,
    )

    task_config = config.PredictTaskConfig(
        precision="32",
        accelerator="cpu",
        output_embeddings=True,
        output_predictions=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[],
    )

    data_module.setup(task_config.setup_stage)

    training_module = SequenceClassificationTrainingModule(
        model_config, trainer_config, label_dict=data_module.label_dict
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        task_config.default_root_dir = tmpdir
        pl_trainer = task_utils.make_trainer_for_task(task_config)
        results = task_utils.predict(
            pl_trainer,
            pl_module=training_module,
            pl_data_module=data_module,
        )
        assert {*results.keys()} == {
            "embeddings",
            "celltype_predictions",
            "celltype_logits",
            "cell_name",
        }


def test_different_pooling_choices_respected(gene2vec_fields, sciplex3_label_columns):
    test_size = 10
    dm = SciPlex3DataModule(
        dataset_kwargs={
            "data_dir": helpers.SciPlex3Paths.root,
            "split_column": "split_random",
        },
        limit_dataset_samples=test_size,
        fields=gene2vec_fields,
        label_columns=sciplex3_label_columns,
        collation_strategy="sequence_classification",
        tokenizer=load_tokenizer("gene2vec"),
        batch_size=10,
        max_length=64,
    )
    dm.setup("predict")
    helpers.update_label_columns(dm.label_columns, dm.label_dict)
    trainer_config = config.TrainerConfig(
        losses=[LossTask.from_label("cell_type", objective=CrossEntropyObjective())]
    )
    model_config = config.SCBertConfig(
        fields=gene2vec_fields,
        label_columns=dm.label_columns,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        hidden_size=32,
    )
    pl_module = SequenceClassificationTrainingModule(
        model_config, trainer_config, label_dict=dm.label_dict
    )
    task_config = config.PredictTaskConfig(
        precision="32",
        accelerator="cpu",
        output_embeddings=True,
        output_predictions=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        task_config.default_root_dir = tmpdir
        hidden_size = model_config.hidden_size

        pooler_layer_names, pooler_layer_embeddings = get_embeddings(
            dm, pl_module, task_config, pooling_method="pooling_layer"
        )
        assert pooler_layer_embeddings.shape == (test_size, hidden_size)

        first_token_names, first_token_embeddings = get_embeddings(
            dm, pl_module, task_config, pooling_method="first_token"
        )
        assert first_token_embeddings.shape == (test_size, model_config.hidden_size)

        mean_pooling_names, mean_pooling_embeddings = get_embeddings(
            dm, pl_module, task_config, pooling_method="mean_pooling"
        )
        assert mean_pooling_embeddings.shape == (test_size, model_config.hidden_size)

        # Test multi-position stacking with list of integers
        multi_position_list = [0, 1]
        multi_position_names, multi_position_embeddings = get_embeddings(
            dm, pl_module, task_config, pooling_method=multi_position_list
        )
        assert multi_position_embeddings.shape == (
            test_size,
            len(multi_position_list) * hidden_size,
        )

        # Test with a different list of positions
        multi_position_list_3 = [0, 2, 3]
        multi_position_names_3, multi_position_embeddings_3 = get_embeddings(
            dm, pl_module, task_config, pooling_method=multi_position_list_3
        )
        assert multi_position_embeddings_3.shape == (
            test_size,
            len(multi_position_list_3) * hidden_size,
        )

        with pytest.raises(ValueError, match=r".*not a pooling.*"):
            get_embeddings(dm, pl_module, task_config, pooling_method="not a pooling")
        np.testing.assert_array_equal(pooler_layer_names, mean_pooling_names)
        np.testing.assert_array_equal(pooler_layer_names, first_token_names)
        np.testing.assert_array_equal(pooler_layer_names, multi_position_names)
        np.testing.assert_array_equal(pooler_layer_names, multi_position_names_3)

        with pytest.raises(AssertionError):
            np.testing.assert_allclose(first_token_embeddings, pooler_layer_embeddings)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(mean_pooling_embeddings, pooler_layer_embeddings)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(first_token_embeddings, mean_pooling_embeddings)

        # Verify multi-position embeddings are different from single-position embeddings
        # The first half of multi_position_embeddings should match first_token (position 0)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                multi_position_embeddings, pooler_layer_embeddings
            )
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                multi_position_embeddings_3, multi_position_embeddings
            )


def get_embeddings(dm, pl_module, task_config, pooling_method):
    pl_module.trainer_config.pooling_method = pooling_method
    pl_trainer = task_utils.make_trainer_for_task(task_config)
    results = task_utils.predict(pl_trainer, pl_module=pl_module, pl_data_module=dm)
    names = results["cell_name"].astype(str)
    embeddings = results["embeddings"].astype(float)
    return names, embeddings


def test_zheng68k_inference_scbert_after_train(
    pl_data_module_zheng68k_seq_cls: Zheng68kDataModule,
    zheng_seq_cls_ckpt,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = config.PredictTaskConfig(
            default_root_dir=tmpdir,
            precision="32",
            accelerator="cpu",
            output_embeddings=True,
            output_predictions=True,
            checkpoint=zheng_seq_cls_ckpt,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[],
        )
        pl_trainer = task_utils.make_trainer_for_task(task_config)
        task_utils.predict_run(
            pl_trainer,
            task_config=task_config,
            data_module=pl_data_module_zheng68k_seq_cls,
            trainer_config=None,
            model_config=None,
        )

        output_path = Path(task_config.default_root_dir)
        embeddings_path = output_path / "embeddings.csv"
        predictions_path = output_path / "predictions.csv"

        embeddings_df = pd.read_csv(embeddings_path, header=None, index_col=0)
        predictions_df = pd.read_csv(predictions_path, index_col=0)

        assert embeddings_df.shape == (2, 32)
        assert all(
            i in pl_data_module_zheng68k_seq_cls.label_dict["celltype"]
            for i in predictions_df["celltype"]
        )

        logits_path = output_path / "logits.csv"
        probabilities_path = output_path / "probabilities.csv"

        logits_df = pd.read_csv(logits_path, index_col=0)
        probabilities_df = pd.read_csv(probabilities_path, index_col=0)

        assert logits_df.shape == (2, 11)
        assert probabilities_df.shape == (2, 11)
        assert round(probabilities_df.iloc[0, :].sum() * 10000) == 10000


def test_zheng68k_inference_scbert_after_train_mlm_mode(
    pl_zheng_mlm_raw_counts: Zheng68kDataModule,
    zheng_mlm_ckpt,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = config.PredictTaskConfig(
            default_root_dir=tmpdir,
            precision="32",
            accelerator="cpu",
            output_embeddings=True,
            output_predictions=True,
            checkpoint=zheng_mlm_ckpt,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[],
        )
        pl_trainer = task_utils.make_trainer_for_task(task_config)
        task_utils.predict_run(
            pl_trainer=pl_trainer,
            task_config=task_config,
            data_module=pl_zheng_mlm_raw_counts,
            trainer_config=None,
            model_config=None,
        )

        output_path = Path(task_config.default_root_dir)
        embeddings_path = output_path / "embeddings.csv"

        embeddings_df = pd.read_csv(embeddings_path, header=None, index_col=0)

        assert embeddings_df.shape == (8, 32)


def test_zheng68k_scbert_test_run_after_train(
    pl_data_module_zheng68k_seq_cls: Zheng68kDataModule,
    zheng_seq_cls_ckpt,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = config.TestTaskConfig(
            default_root_dir=tmpdir,
            precision="32",
            accelerator="cpu",
            checkpoint=zheng_seq_cls_ckpt,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[],
            num_bootstrap_runs=5,
        )
        pl_trainer = task_utils.make_trainer_for_task(task_config)

        runs_list = task_utils.test_run(
            pl_trainer,
            task_config=task_config,
            data_module=pl_data_module_zheng68k_seq_cls,
            trainer_config=None,
            model_config=None,
            clearml_logger=None,
        )
        eps = 1e-6
        assert len(runs_list) == 5
        # runs should not all return the same results
        assert (
            np.linalg.norm(np.diff([i["celltype_accuracy"] for i in runs_list])) > eps
        )


def test_zheng68k_scbert_test_run_after_train_scheduled(
    pl_data_module_zheng68k_seq_cls: Zheng68kDataModule,
    zheng_seq_cls_ckpt,
):
    from bmfm_targets.training.callbacks import BatchSizeScheduler

    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = config.TestTaskConfig(
            default_root_dir=tmpdir,
            precision="32",
            accelerator="cpu",
            checkpoint=zheng_seq_cls_ckpt,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[BatchSizeScheduler(test_batch_size=1, test_max_length=32)],
            num_bootstrap_runs=5,
        )
        pl_trainer = task_utils.make_trainer_for_task(task_config)

        runs_list = task_utils.test_run(
            pl_trainer,
            task_config=task_config,
            data_module=pl_data_module_zheng68k_seq_cls,
            trainer_config=None,
            model_config=None,
            clearml_logger=None,
        )
        eps = 1e-6
        assert len(runs_list) == 5
        # runs should not all return the same results
        assert (
            np.linalg.norm(np.diff([i["celltype_accuracy"] for i in runs_list])) > eps
        )
        assert pl_trainer.datamodule.max_length == 32
        assert pl_trainer.datamodule.batch_size == 1


def test_scp1884_prediction_on_zheng_ckpt(
    gene2vec_unmasked_fields,
    zheng_seq_cls_ckpt,
):
    # SCP1884 has "Celltype" (capital C), Zheng68k has "celltype" (lowercase)
    label_columns = [config.LabelColumnInfo(label_column_name="Celltype")]

    dm = SCP1884DataModule(
        data_dir=helpers.SCP1884Paths.root,
        mlm=False,
        fields=gene2vec_unmasked_fields,
        label_columns=label_columns,
        collation_strategy="sequence_classification",
        tokenizer=load_tokenizer("gene2vec"),
    )
    dm.prepare_data()
    dm.setup()
    helpers.update_label_columns(dm.label_columns, dm.label_dict)

    # Verify SCP1884 has "Celltype" with 68 classes
    assert "Celltype" in dm.label_dict
    assert len(dm.label_dict["Celltype"]) == 68

    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = config.PredictTaskConfig(
            default_root_dir=tmpdir,
            precision="32",
            accelerator="cpu",
            checkpoint=zheng_seq_cls_ckpt,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[],
            output_predictions=True,
            output_embeddings=True,
        )

        pl_trainer = task_utils.make_trainer_for_task(task_config)
        task_utils.predict_run(
            pl_trainer,
            task_config,
            model_config=None,
            data_module=dm,
            trainer_config=None,
        )

        output_path = Path(task_config.default_root_dir)
        embeddings_path = output_path / "embeddings.csv"
        predictions_path = output_path / "predictions.csv"

        embeddings_df = pd.read_csv(embeddings_path, header=None, index_col=0)
        predictions_df = pd.read_csv(predictions_path, index_col=0)

        assert embeddings_df.shape == (1020, 32)


def test_scp1884_prediction_on_zheng_multitask_ckpt(
    gene2vec_unmasked_fields,
    zheng_multitask_ckpt,
):
    dm = SCP1884DataModule(
        data_dir=helpers.SCP1884Paths.root,
        mlm=False,
        fields=gene2vec_unmasked_fields,
        collation_strategy="multitask",
        tokenizer=load_tokenizer("gene2vec"),
    )
    dm.prepare_data()
    dm.setup()
    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = config.PredictTaskConfig(
            default_root_dir=tmpdir,
            precision="32",
            accelerator="cpu",
            checkpoint=zheng_multitask_ckpt,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[],
        )

        pl_module = task_utils.instantiate_module_from_checkpoint(
            task_config, dm, model_config=None, trainer_config=None
        )
        pl_trainer = task_utils.make_trainer_for_task(task_config)
        results = task_utils.predict(pl_trainer, pl_module, dm)

        assert results["celltype_predictions"] is not None
        assert results["embeddings"] is not None

        task_utils.save_prediction_results(
            task_config.default_root_dir,
            pl_module.label_dict,
            results,
        )
        task_utils.save_embeddings_results(task_config.default_root_dir, results)

        output_path = Path(task_config.default_root_dir)
        embeddings_path = output_path / "embeddings.csv"
        predictions_path = output_path / "predictions.csv"

        embeddings_df = pd.read_csv(embeddings_path, header=None, index_col=0)
        predictions_df = pd.read_csv(predictions_path, index_col=0)

        assert embeddings_df.shape == (1020, 32)
        assert all(
            i in pl_module.label_dict["celltype"] for i in predictions_df["celltype"]
        )


def test_generic_dataset_with_no_label_dict_prediction_on_zheng_ckpt(
    gene2vec_fields,
    zheng_seq_cls_ckpt,
):
    from bmfm_targets.training.data_module import DataModule

    dm = SCP1884DataModule(
        data_dir=helpers.SCP1884Paths.root,
        mlm=False,
        fields=gene2vec_fields,
        collation_strategy="sequence_classification",
        tokenizer=load_tokenizer("gene2vec"),
    )
    dm.prepare_data()
    dm.setup("predict")
    generic_dm = DataModule(
        transform_datasets=False,
        dataset_kwargs={
            "processed_data_source": dm.dataset_kwargs["processed_data_source"]
        },
        max_length=16,
        tokenizer=dm.tokenizer,
        fields=dm.fields,
        collation_strategy="sequence_classification",
    )
    generic_dm.prepare_data()
    generic_dm.setup("predict")

    assert generic_dm.label_dict is None

    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = config.PredictTaskConfig(
            default_root_dir=tmpdir,
            precision="32",
            accelerator="cpu",
            checkpoint=zheng_seq_cls_ckpt,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[],
        )

        pl_module = task_utils.instantiate_module_from_checkpoint(
            task_config, generic_dm, model_config=None, trainer_config=None
        )
        pl_trainer = task_utils.make_trainer_for_task(task_config)
        results = task_utils.predict(pl_trainer, pl_module, generic_dm)

        assert results["celltype_predictions"] is not None
        assert results["embeddings"] is not None

        task_utils.save_prediction_results(
            task_config.default_root_dir,
            pl_module.label_dict,
            results,
        )
        task_utils.save_embeddings_results(task_config.default_root_dir, results)

        output_path = Path(task_config.default_root_dir)
        embeddings_path = output_path / "embeddings.csv"
        predictions_path = output_path / "predictions.csv"

        embeddings_df = pd.read_csv(embeddings_path, header=None, index_col=0)
        predictions_df = pd.read_csv(predictions_path, index_col=0)

        assert embeddings_df.shape == (1020, 32)
        assert all(
            i in pl_module.label_dict["celltype"] for i in predictions_df["celltype"]
        )


def test_calculate_95_ci_scalar():
    data = 1
    mean_quan, lower_bound_quan, upper_bound_quan = calculate_95_ci(
        data, n=1, ci_method="bootstrap_quantiles"
    )
    mean_bin, lower_bound_bin, upper_bound_bin = calculate_95_ci(
        data, n=1, ci_method="binomial"
    )
    mean_t, lower_bound_t, upper_bound_t = calculate_95_ci(
        data, n=1, ci_method="bootstrap_t_interval"
    )
    mean_wil, lower_bound_wil, upper_bound_wil = calculate_95_ci(
        data, n=1, ci_method="wilson"
    )
    assert mean_quan == mean_bin
    assert mean_quan == mean_t
    assert mean_wil == mean_bin
    assert lower_bound_quan == data
    assert upper_bound_quan == data
    assert lower_bound_bin == data
    assert upper_bound_bin == data
    assert lower_bound_wil > 0
    assert upper_bound_wil == data
    assert np.isnan(lower_bound_t)
    assert np.isnan(upper_bound_t)


def test_calculate_95_ci_list():
    data = [i / 100 for i in range(0, 101)]
    mean_quan, lower_bound_quan, upper_bound_quan = calculate_95_ci(
        data, n=100, ci_method="bootstrap_quantiles"
    )
    mean_bin, lower_bound_bin, upper_bound_bin = calculate_95_ci(
        data, n=100, ci_method="binomial"
    )
    mean_t, lower_bound_t, upper_bound_t = calculate_95_ci(
        data, n=100, ci_method="bootstrap_t_interval"
    )
    mean_wil, lower_bound_wil, upper_bound_wil = calculate_95_ci(
        data, n=1, ci_method="wilson"
    )

    assert mean_quan == mean_bin
    assert mean_quan == mean_t
    assert mean_wil == mean_bin
    assert lower_bound_quan == 0.025
    assert upper_bound_quan == 0.975
    np.testing.assert_almost_equal(
        lower_bound_bin, mean_bin - 1.96 * np.sqrt(0.5 * (1 - 0.5) / 100), decimal=3
    )
    np.testing.assert_almost_equal(
        upper_bound_bin, mean_bin + 1.96 * np.sqrt(0.5 * (1 - 0.5) / 100), decimal=3
    )
    assert lower_bound_t < mean_t
    assert upper_bound_t > mean_t
    assert lower_bound_bin >= 0
    assert upper_bound_wil <= 1


def test_instantiate_inherits_checkpoint_losses_when_override_is_none(
    zheng_seq_cls_ckpt, pl_data_module_zheng68k_seq_cls
):
    """When trainer_config.losses is None, inherit from checkpoint."""
    task_config = config.PredictTaskConfig(checkpoint=zheng_seq_cls_ckpt)
    override_config = config.TrainerConfig(losses=None)  # Signal: use checkpoint losses

    pl_module = task_utils.instantiate_module_from_checkpoint(
        task_config,
        pl_data_module_zheng68k_seq_cls,
        model_config=None,
        trainer_config=override_config,
    )

    # Should have inherited checkpoint's losses
    assert len(pl_module.loss_tasks) > 0


def test_instantiate_uses_empty_losses_for_embedding_only(
    zheng_multitask_ckpt, pl_data_module_zheng68k_multitask
):
    """When trainer_config.losses is [], no losses (embedding-only)."""
    task_config = config.PredictTaskConfig(checkpoint=zheng_multitask_ckpt)
    override_config = config.TrainerConfig(losses=[])  # Explicit: no losses

    pl_module = task_utils.instantiate_module_from_checkpoint(
        task_config,
        pl_data_module_zheng68k_multitask,
        model_config=None,
        trainer_config=override_config,
    )

    assert pl_module.loss_tasks == []
