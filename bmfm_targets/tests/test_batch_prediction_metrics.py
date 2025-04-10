from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from bmfm_targets.tokenization import load_tokenizer
from bmfm_targets.training.metrics import (
    batch_prediction_metrics,
    perturbation_metrics,
    plots,
)


@pytest.fixture(scope="module")
def mean_expressions(pl_data_module_adamson_weissman_seq_labeling):
    return perturbation_metrics.get_mean_expressions(
        pl_data_module_adamson_weissman_seq_labeling.get_dataset_instance().group_means.processed_data
    )


@pytest.fixture(scope="module")
def perturbations_predictions_list(pl_data_module_adamson_weissman_seq_labeling):
    dm = pl_data_module_adamson_weissman_seq_labeling
    predictions_list = []
    cell_names = []
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
        cell_names.append(batch["cell_names"])
    return list(chain.from_iterable(cell_names)), predictions_list


@pytest.fixture(scope="module")
def mlm_predictions_list(pl_mock_data_mlm_no_binning):
    dm = pl_mock_data_mlm_no_binning
    field = [f for f in dm.fields if f.field_name == "expressions"][0]
    predictions_list = []
    cell_names = []
    for batch in dm.train_dataloader():
        input_ids = batch["input_ids"]
        labels = batch["labels"]["expressions"]
        # logits are basically the same as labels with some noise
        # we don't care about -100, it will be eliminated later
        logits = torch.tensor(np.random.normal(labels.unsqueeze(-1), 0.01))
        predictions_list.append(
            batch_prediction_metrics.concat_field_loss_batch_tensors(
                input_ids,
                labels,
                predictions=logits,
                **{f"expressions_{m}": logits for m in field.decode_modes},
            )
        )
        cell_names.append(batch["cell_names"])

    return list(chain.from_iterable(cell_names)), predictions_list


@pytest.fixture(scope="module")
def mlm_predictions_df(pl_mock_data_mlm_no_binning, mlm_predictions_list):
    dm = pl_mock_data_mlm_no_binning
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
    tokenizer = load_tokenizer("all_genes")
    id2gene = {v: k for k, v in tokenizer.get_field_vocab("genes").items()}
    cell_names, predictions = perturbations_predictions_list
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
    )


@pytest.fixture(scope="module")
def perturbation_grouped_df(perturbations_predictions_df):
    return perturbation_metrics.get_group_average_expressions(
        perturbations_predictions_df
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


def test_aggregated_perturbation_metrics(perturbation_grouped_df, mean_expressions):
    pert_groups = perturbation_grouped_df.reset_index().perturbed_set.unique()
    assert len(pert_groups) > 1
    for pert_group in pert_groups:
        agg_metrics = perturbation_metrics.get_aggregated_perturbation_metrics(
            perturbation_grouped_df.loc[pert_group], mean_expressions, pert_group
        )
        assert all(isinstance(k, str) for k in agg_metrics.keys())
        assert all(isinstance(v, float) for v in agg_metrics.values())


def test_gt_density_plot(perturbations_predictions_df):
    fig = plots.make_predictions_gt_density_plot(perturbations_predictions_df)
    plt.close(fig)


def test_top20_deg_perturbation_plot(
    perturbations_predictions_df,
    mean_expressions,
    pl_data_module_adamson_weissman_seq_labeling,
    perturbation_grouped_df,
):
    ds = pl_data_module_adamson_weissman_seq_labeling.get_dataset_instance()
    top20 = ds.processed_data.uns["top_non_zero_de_20"]
    for pert in perturbation_grouped_df.reset_index().perturbed_set.unique():
        top20_df = perturbations_predictions_df[
            perturbations_predictions_df.input_genes.isin(top20[pert])
        ]
        mean_control = mean_expressions["Control"].loc[top20[pert]]
        fig = plots.make_top20_deg_perturbation_plot(top20_df, mean_control)
        plt.close(fig)
