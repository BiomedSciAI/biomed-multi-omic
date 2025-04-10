"""Tests for training sequence labeling model."""

import tempfile
from pathlib import Path

import pandas as pd
import torch

from bmfm_targets import config
from bmfm_targets.datasets.base_rna_dataset import BaseRNAExpressionDataset
from bmfm_targets.datasets.perturbation import ScperturbDataModule
from bmfm_targets.tasks.task_utils import make_trainer_for_task, train_run
from bmfm_targets.tokenization import load_tokenizer


def test_sort_by_expression_keeps_perturbed_gene_in_list(
    pl_data_module_adamson_weissman_seq_labeling,
):
    dm = pl_data_module_adamson_weissman_seq_labeling
    dm.sequence_order = "sorted"
    perturbed_id = dm.tokenizer.get_field_vocab("perturbations")["1"]

    for batch in dm.train_dataloader():
        # perturbed gene first (after cls)
        assert (batch["input_ids"][:, 2, 1] == perturbed_id).bool().all()
        # the rest of the expressions in descending order (not counting sep at end)
        assert (batch["input_ids"][:, 1, 2:-2] >= batch["input_ids"][:, 1, 3:-1]).all()


def test_train_seq_label(pl_data_module_adamson_weissman_seq_labeling):
    finetune_losses = [
        {
            "field_name": "label_expressions",
            "name": "mse",
            "weight": 1,
            "ignore_zero": True,
        },
        {
            "field_name": "label_expressions",
            "name": "is_zero_bce",
            "weight": 1,
        },
    ]
    trainer_config = config.TrainerConfig(
        losses=finetune_losses,
        batch_prediction_behavior="dump",
    )
    model_config = config.SCBertConfig(
        fields=pl_data_module_adamson_weissman_seq_labeling.fields,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=128,
        hidden_size=64,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = config.TrainingTaskConfig(
            default_root_dir=tmpdir + "/finetune",
            max_epochs=1,
            max_steps=3,
            accelerator="cpu",
            val_check_interval=1,
            gradient_clip_val=0.5,
            precision="32",
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=True,
            callbacks=[],
        )

        pl_trainer = make_trainer_for_task(task_config)
        train_run(
            pl_trainer,
            data_module=pl_data_module_adamson_weissman_seq_labeling,
            model_config=model_config,
            task_config=task_config,
            trainer_config=trainer_config,
        )
        assert pl_trainer.logged_metrics

        ckpt_dict = torch.load(
            pl_trainer.checkpoint_callback._last_checkpoint_saved, weights_only=False
        )

        saved_model_config = ckpt_dict["hyper_parameters"]["model_config"]
        assert isinstance(saved_model_config, type(model_config))

        saved_trainer_config = ckpt_dict["hyper_parameters"]["trainer_config"]
        assert isinstance(saved_trainer_config, type(trainer_config))

        dumped_batches = (
            Path(task_config.default_root_dir) / "lightning_logs" / "version_0"
        ).glob("validation_label_expressions_iteration*.csv")
        dumped_batches = [*dumped_batches]
        assert len(dumped_batches) > 0
        preds_df = pd.read_csv(dumped_batches[0], index_col=0)
        # check that it is indexing by cell name, they are ATCG barcodes
        assert all(i[0] in "ATCG" for i in preds_df.index)
        assert preds_df.shape[0] > 1
        assert preds_df.shape[1] == 7


def test_train_from_ckpt_tokenized(
    pl_data_module_adamson_weissman_seq_labeling,
    mock_data_seq_cls_ckpt,
    perturbation_fields_tokenized,
):
    finetune_losses = [
        {
            "field_name": "label_expressions",
            "name": "token_mse",
            "weight": 1,
            "label_smoothing": 0.01,
        },
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        do_load_and_train_from_ckpt(
            pl_data_module_adamson_weissman_seq_labeling,
            mock_data_seq_cls_ckpt,
            perturbation_fields_tokenized,
            finetune_losses,
            tmpdir,
        )


def test_train_from_ckpt_rda(
    pl_data_module_adamson_weissman_seq_labeling,
    mock_data_mlm_rda_ckpt,
):
    dm = pl_data_module_adamson_weissman_seq_labeling
    seq_lab_losses = [{"field_name": "label_expressions", "name": "mse"}]
    with tempfile.TemporaryDirectory() as tmpdir:
        do_load_and_train_from_ckpt(
            dm,
            mock_data_mlm_rda_ckpt,
            dm.fields,
            seq_lab_losses,
            tmpdir,
        )


def do_load_and_train_from_ckpt(
    seq_labeling_dm,
    ckpt_path,
    perturbation_fields,
    seq_labeling_losses,
    tmpdir,
):
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
    dm2 = ScperturbDataModule(
        data_dir=seq_labeling_dm.data_dir,
        tokenizer=load_tokenizer(Path(ckpt_path).parent),
        transform_datasets=False,
        mlm=False,
        collation_strategy="sequence_labeling",
        num_workers=0,
        batch_size=3,
        fields=perturbation_fields,
        max_length=128,
        limit_dataset_samples=12,
        sequence_order="sorted",
    )
    dm2.prepare_data()
    dm2.setup("fit")
    model_config = torch.load(ckpt_path, weights_only=False)["hyper_parameters"][
        "model_config"
    ]
    model_config.checkpoint = ckpt_path
    model_config.fields = perturbation_fields

    finetune_trainer_config = config.TrainerConfig(
        batch_size=dm2.batch_size,
        losses=seq_labeling_losses,
        batch_prediction_behavior="track",
    )
    pl_trainer = make_trainer_for_task(finetune_task_config)
    train_run(
        pl_trainer,
        task_config=finetune_task_config,
        model_config=model_config,
        data_module=dm2,
        trainer_config=finetune_trainer_config,
    )


def test_load_perturbation_dataset_from_test_resources(
    pl_data_module_adamson_weissman_seq_labeling,
):
    dataset = pl_data_module_adamson_weissman_seq_labeling.train_dataset
    for i in range(10):
        mfi = dataset[i]
        pert_ds = pd.Series(data=mfi["label_expressions"], index=mfi["genes"])
        pert_orig = dataset.processed_data[mfi.metadata["cell_name"]].to_df().T
        pert_both = pd.concat([pert_ds, pert_orig], axis=1).fillna(0)
        assert pert_both.diff(axis=1).iloc[:, -1].sum() == 0

        control_ds = pd.Series(data=mfi["expressions"], index=mfi["genes"])
        control_orig = (
            dataset.processed_data[mfi.metadata["control_cell_name"]].to_df().T
        )
        control_both = pd.concat([control_ds, control_orig], axis=1).fillna(0)
        assert control_both.diff(axis=1).iloc[:, -1].sum() == 0


def test_mean_expressions_available_in_dataset_including_limit_genes(
    pl_data_module_adamson_weissman_seq_labeling,
):
    seq_labeling_dm = pl_data_module_adamson_weissman_seq_labeling

    protein_coding_dm = ScperturbDataModule(
        data_dir=seq_labeling_dm.data_dir,
        tokenizer=seq_labeling_dm.tokenizer,
        transform_datasets=False,
        mlm=False,
        collation_strategy="sequence_labeling",
        num_workers=0,
        batch_size=3,
        fields=seq_labeling_dm.fields,
        max_length=128,
        limit_dataset_samples=12,
        sequence_order="sorted",
        limit_genes="protein_coding",
    )
    protein_coding_dm.setup("fit")

    _assert_de_genes_valid(pl_data_module_adamson_weissman_seq_labeling.train_dataset)
    _assert_de_genes_valid(protein_coding_dm.train_dataset)


def _assert_de_genes_valid(dataset):
    assert isinstance(dataset.group_means, BaseRNAExpressionDataset)

    non_dropout_20 = dataset.processed_data.uns["top_non_dropout_de_20"]
    non_zero_20 = dataset.processed_data.uns["top_non_zero_de_20"]
    all_vals_genes = lambda v: all(v_i in dataset.processed_data.var_names for v_i in v)
    assert all_vals_genes(non_zero_20)
    assert all_vals_genes(non_dropout_20)
    assert all(all_vals_genes(v) for v in non_zero_20.values())
    assert all(all_vals_genes(v) for v in non_dropout_20.values())
