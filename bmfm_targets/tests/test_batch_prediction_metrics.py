import random
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

from bmfm_targets.datasets.zheng68k import Zheng68kDataModule
from bmfm_targets.tests import helpers
from bmfm_targets.training.metrics import (
    batch_prediction_metrics,
    perturbation_metrics,
    plots,
)


@pytest.fixture(scope="module")
def label_predictions_df():
    labels = ["A", "A", "A", "B", "B", "B", "B", "C", "C", "C"]
    labels = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    label_column_name = "target"
    this_label_dict = {"A": 0, "B": 1, "C": 2}
    sample_names = [f"cell_{i}" for i in range(len(labels))]
    predictions_list = []
    for label in labels:
        if label == 0:
            this_pred = 1
        elif label == 1:
            this_pred = random.choice([0, 1])
        else:
            this_pred = 2
        dummy_logits = torch.rand(len(this_label_dict)).tolist()
        predictions_list.append(torch.tensor([[this_pred, label] + dummy_logits]))

    preds_df = batch_prediction_metrics.create_label_predictions_df(
        predictions_list, label_column_name, sample_names, this_label_dict
    )
    return preds_df


@pytest.fixture(scope="module")
def label_predictions_with_silent_labels_df():
    labels = ["A", "A", "A", "B", "B", "B", "B", "C", "C", "C"]
    # treat "C" as a silent label value, set it to -100
    labels = [0, 0, 0, 1, 1, 1, 1, -100, -100, -100]
    label_column_name = "target"
    this_label_dict = {"A": 0, "B": 1, "C": 2}
    sample_names = [f"cell_{i}" for i in range(len(labels))]
    predictions_list = []
    for label in labels:
        if label == 0:
            this_pred = 1
        elif label == 1:
            this_pred = random.choice([0, 1])
        else:
            this_pred = 2
        dummy_logits = torch.rand(len(this_label_dict)).tolist()
        predictions_list.append(torch.tensor([[this_pred, label] + dummy_logits]))

    preds_df = batch_prediction_metrics.create_label_predictions_df(
        predictions_list, label_column_name, sample_names, this_label_dict
    )
    return preds_df


@pytest.fixture(scope="module")
def grouped_ground_truth_expressions(pl_data_module_adamson_weissman_seq_labeling):
    return perturbation_metrics.get_grouped_ground_truth(
        pl_data_module_adamson_weissman_seq_labeling.get_dataset_instance().group_means
    )


@pytest.fixture(scope="module")
def perturbations_predictions_list(pl_data_module_adamson_weissman_seq_labeling):
    dm = pl_data_module_adamson_weissman_seq_labeling
    predictions_list = []
    cell_names = []
    perturbed_genes = []
    for batch in dm.train_dataloader():
        input_ids = batch["input_ids"]
        labels = batch["labels"]["label_expressions"]
        # logits are basically the same as labels with some noise
        # we don't care about -100, it will be eliminated later
        logits = torch.tensor(np.random.normal(labels, 0.01))
        predictions_list.append(
            batch_prediction_metrics.concat_field_loss_batch_tensors(
                input_ids, labels, logits
            )
        )
        cell_names.append(batch["cell_name"])
        perturbed_genes.append(batch["perturbed_genes"])
    return (
        dm.tokenizer,
        list(chain.from_iterable(cell_names)),
        list(chain.from_iterable(perturbed_genes)),
        predictions_list,
    )


@pytest.fixture(scope="module")
def mlm_predictions_list(pl_zheng_mlm_raw_counts):
    dm = Zheng68kDataModule(
        data_dir=helpers.Zheng68kPaths.root,
        processed_name=helpers.Zheng68kPaths.raw_counts_name,
        transform_datasets=False,
        tokenizer=pl_zheng_mlm_raw_counts.tokenizer,
        label_columns=pl_zheng_mlm_raw_counts.label_columns,
        batch_size=2,
        fields=pl_zheng_mlm_raw_counts.fields,
        mlm=True,
        collation_strategy="language_modeling",
        rda_transform=2000,
        max_length=128,
        pad_to_multiple_of=2,
    )
    dm.prepare_data()
    dm.setup("fit")
    field = [f for f in dm.fields if f.field_name == "expressions"][0]
    predictions_list = []
    cell_names = []
    for batch in dm.train_dataloader():
        input_ids = batch["input_ids"]
        labels = batch["labels"]["expressions"]
        in_use_vals = labels[labels != -100]
        in_use_vals[: len(in_use_vals) // 2] = 0
        labels[labels != -100] = in_use_vals
        # logits are basically the same as labels with some noise
        # we don't care about -100, it will be eliminated later
        mse_logits = torch.tensor(np.random.normal(labels.unsqueeze(-1), 0.1))
        is_zero_logits = 2 * torch.rand_like(labels.unsqueeze(-1)) - 1
        predictions_list.append(
            batch_prediction_metrics.concat_field_loss_batch_tensors(
                input_ids,
                labels,
                predictions=mse_logits,
                expressions_regression=mse_logits,
                expressions_is_zero=is_zero_logits,
            )
        )
        cell_names.append(batch["cell_name"])

    return list(chain.from_iterable(cell_names)), predictions_list


@pytest.fixture(scope="module")
def mlm_predictions_df(pl_zheng_mlm_raw_counts, mlm_predictions_list):
    dm = pl_zheng_mlm_raw_counts
    tokenizer = dm.tokenizer
    cell_names, predictions = mlm_predictions_list
    id2gene = {v: k for k, v in tokenizer.get_field_vocab("genes").items()}
    field = [f for f in dm.fields if f.field_name == "expressions"][0]
    columns = batch_prediction_metrics.field_predictions_df_columns(
        dm.fields, field, "mlm"
    )
    preds = batch_prediction_metrics.create_field_predictions_df(
        predictions, id2gene, columns=columns, sample_names=cell_names
    )
    # if there are no masked values because the seq is too short in testing
    # this will be a subset, otherwise it will be identical
    assert {*preds.index}.issubset({*cell_names})
    return preds


@pytest.fixture(scope="module")
def perturbations_predictions_df(perturbations_predictions_list):
    tokenizer, cell_names, perturbed_genes, predictions = perturbations_predictions_list
    id2gene = {v: k for k, v in tokenizer.get_field_vocab("genes").items()}
    return batch_prediction_metrics.create_field_predictions_df(
        predictions,
        id2gene,
        columns=[
            "gene_id",
            "control_expressions",
            "is_perturbed",
            "predicted_expressions",
            "label_expressions",
        ],
        sample_names=cell_names,
        sample_level_metadata={"perturbed_genes": perturbed_genes},
    )


@pytest.fixture(scope="module")
def grouped_predicted_expressions(
    perturbations_predictions_df, grouped_ground_truth_expressions
):
    return perturbation_metrics.get_grouped_predictions(
        perturbations_predictions_df, grouped_ground_truth_expressions
    )


def test_gene_level_errors(mlm_predictions_df):
    gene_level_error = batch_prediction_metrics.get_gene_level_expression_error(
        mlm_predictions_df
    )
    assert gene_level_error.shape[0] > 5
    gene_metrics = batch_prediction_metrics.get_gene_metrics_from_gene_errors(
        gene_level_error
    )
    assert all(isinstance(x, float) for x in gene_metrics.values())


def test_perturbation_grouped_df(
    grouped_predicted_expressions, grouped_ground_truth_expressions
):
    n_perts = grouped_predicted_expressions.reset_index()["perturbed_genes"].nunique()
    n_genes = grouped_ground_truth_expressions.shape[0]
    assert grouped_predicted_expressions.shape[0] == n_perts * n_genes


def test_aggregated_perturbation_metrics(grouped_predicted_expressions):
    pert_groups = grouped_predicted_expressions.reset_index().perturbed_genes.unique()
    assert len(pert_groups) > 1
    for pert_group in pert_groups:
        agg_metrics = perturbation_metrics.get_aggregated_perturbation_metrics(
            grouped_predicted_expressions.loc[pert_group]
        )
        assert all(isinstance(k, str) for k in agg_metrics.keys())
        assert all(isinstance(v, float | np.float32) for v in agg_metrics.values())


def test_gt_density_plot(perturbations_predictions_df):
    fig = plots.make_predictions_gt_density_plot(perturbations_predictions_df)
    plt.close(fig)


def test_distances_heatmap_plot(grouped_predicted_expressions):
    (
        real_effects,
        pred_effects,
        perturbations,
        overlap_genes,
    ) = perturbation_metrics.prepare_args_for_discrimination_score(
        grouped_predicted_expressions
    )
    _, distances = perturbation_metrics.discrimination_score(
        real_effects, pred_effects, perturbations, overlap_genes
    )
    plots.make_heatmap_plot(
        distances,
        "ground truth",
        "predictions",
        "predicted vs ground truth L1 distances of pseudo bulks",
    )


def test_silent_label_handling(label_predictions_with_silent_labels_df):
    vc = label_predictions_with_silent_labels_df.target_label.value_counts()
    assert vc["Silenced Label Value"] > 0


def test_accuracy_w_ci_plot(label_predictions_df):
    fig = plots.make_accuracy_by_target_plot(
        label_predictions_df, label_column_name="target"
    )
    plt.close(fig)

    # Test accuracy values
    preds = pd.Series(["A", "A", "A", "A"])
    label = "A"
    assert plots.calc_accuracy(preds, label) == 1.0

    preds = pd.Series(["B", "C", "D", "E"])
    label = "A"
    assert plots.calc_accuracy(preds, label) == 0.0

    preds = pd.Series(["A", "B", "A", "C"])
    label = "A"
    assert plots.calc_accuracy(preds, label) == 0.5


def test_top20_deg_perturbation_plot(
    perturbations_predictions_df,
    grouped_ground_truth_expressions,
    pl_data_module_adamson_weissman_seq_labeling,
    grouped_predicted_expressions,
):
    ds = pl_data_module_adamson_weissman_seq_labeling.get_dataset_instance()
    top20 = ds.processed_data.uns["top_non_zero_de_20"]

    assert "perturbed_genes" in grouped_predicted_expressions.reset_index().columns, (
        f"Index names: {grouped_predicted_expressions.index.names}, "
        f"Columns after reset_index: {grouped_predicted_expressions.reset_index().columns.tolist()}"
    )

    perturbations = grouped_predicted_expressions.reset_index()[
        "perturbed_genes"
    ].unique()
    for pert in perturbations:
        top20_df = perturbations_predictions_df[
            perturbations_predictions_df.input_genes.isin(top20[pert])
        ]
        mean_control = grouped_ground_truth_expressions["Control"].loc[top20[pert]]
        fig = plots.make_top20_deg_perturbation_plot(top20_df, mean_control)
        plt.close(fig)


def test_filling_predictions_to_anndata(
    perturbations_predictions_df, grouped_predicted_expressions
):
    from bmfm_targets.training.callbacks import create_adata_from_predictions_df

    adata = create_adata_from_predictions_df(
        perturbations_predictions_df, grouped_predicted_expressions
    )

    for sample_id, row in perturbations_predictions_df.iterrows():
        assert (
            adata[sample_id, row["input_genes"]].X.data[0]
            == row["predicted_expressions"]
        )
