"""
Functions for calculating perturbation-specific metrics.

Perturbation metrics are different because they involve various levels of aggregation
across samples.

Typically the predictions for a given perturbation are averaged into one group prediction
and compared with the ground-truth prediction.

Also, the predictions are compared against the average, unperturbed expression.
"""

import numpy as np
import pandas as pd
from scanpy import AnnData
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances as pwd


def prepare_args_for_discrimination_score(group_expressions: pd.DataFrame):
    """
    Prepare arguments for discrimination score calculation.

    Args:
    ----
        group_expressions (pd.DataFrame): DataFrame produced by function
            `get_group_average_expressions`

    """
    perturbations = group_expressions.index.get_level_values(0).unique().sort_values()

    genes_sets = [
        set(group_expressions.loc[pert].index.get_level_values(0))
        for pert in perturbations
    ]
    genes = np.array(sorted(set.intersection(*genes_sets)))

    real_effects = np.vstack(
        [
            group_expressions.loc[(pert, genes), "label_expressions"].to_numpy()
            for pert in perturbations
        ]
    )
    pred_effects = np.vstack(
        [
            group_expressions.loc[(pert, genes), "predicted_expressions"].to_numpy()
            for pert in perturbations
        ]
    )
    return real_effects, pred_effects, perturbations, genes


def discrimination_score(real, pred, perts, genes, metric="l1", exclude=True):
    """
    Compute discrimination scores for perturbation prediction.

    For each perturbation, we compare its predicted effect vector to the
    real effect vectors of all perturbations using a given distance metric.
    The score is 1.0 if the correct perturbation is ranked closest (best match),
    and 0.0 if it is ranked farthest (worst match). If `exclude` is True,
    the target gene for each perturbation is omitted from the comparison.

    Args:
    ----
        real (ndarray): Real effects, shape (P, G) — P perturbations, G genes/features.
        pred (ndarray): Predicted effects, shape (P, G).
        perts (ndarray): Perturbation identifiers, shape (P,).
        genes (ndarray): Gene/feature identifiers, shape (G,).
        metric (str): Distance metric for `sklearn.metrics.pairwise_distances`.
        exclude (bool): Whether to exclude the target gene from comparisons.

    Returns:
    -------
        dict[str, float]: Mapping perturbation → discrimination score in [0, 1].
    """
    num_perts, num_genes = real.shape
    assert pred.shape == (num_perts, num_genes)
    assert len(perts) == num_perts
    assert len(genes) == num_genes

    max_rank = max(num_perts - 1, 1)

    if not exclude:
        distance_matrix = pwd(real, pred, metric=metric)  # shape (P, P)
        order = np.argsort(distance_matrix, axis=0)  # row order per column
        ranks = np.empty_like(order)
        ranks[order, np.arange(num_perts)] = np.arange(num_perts)  # invert permutation
        rank_positions = ranks[np.arange(num_perts), np.arange(num_perts)]
        scores = 1 - rank_positions / max_rank
        return dict(zip(map(str, perts), scores))

    results = {}
    for pert_idx, pert_name in enumerate(perts):
        if isinstance(pert_name, str) and "_" in pert_name:
            gene_mask = ~np.isin(genes, pert_name.split("_"))
        elif isinstance(pert_name, str):
            gene_mask = genes != pert_name
        else:
            raise ValueError(f"Unexpected perturbation name type: {type(pert_name)}")
        masked_real = real[:, gene_mask]
        masked_pred = pred[pert_idx : pert_idx + 1, gene_mask]
        distances = pwd(masked_real, masked_pred, metric=metric).ravel()
        num_better_matches = (distances < distances[pert_idx]).sum()
        score = 1 - num_better_matches / max_rank
        results[str(pert_name)] = score

    return results


def get_aggregated_perturbation_metrics(
    grouped_predictions: pd.DataFrame,
    grouped_ground_truth: pd.DataFrame,
    perturbation_group: str,
):
    """
    Compute Pearson correlations between predicted, control, and ground-truth
    perturbation expression profiles over pseudobulk (mean expressions).

    Includes calculation of delta expressions, which are the differences between the
    perturbed samples and the control mean expressions.

    Parameters
    ----------
    grouped_predictions : pd.DataFrame
        Observed expressions with rows as genes and columns including
        'predicted_expressions', 'is_perturbed', and one column per perturbation.
        Already averaged across samples with the same perturbation (pseudobulk).
    grouped_ground_truth : pd.DataFrame
        Reference mean expressions with rows as genes and columns including
        'Control' and perturbation groups. These are the ground-truth pseudobulk
        expressions for each perturbation.
    perturbation_group : str
        Name of the perturbation column to evaluate.

    Returns
    -------
    dict
        {
            "agg_pcc": correlation(predicted, gt),
            "delta_agg_pcc": correlation(predicted_delta, gt_delta),
            "baseline_agg_pcc": correlation(control, gt),
        }
    """
    aligned = grouped_predictions.merge(
        grouped_ground_truth, how="outer", left_index=True, right_index=True
    ).fillna(0)
    aligned = aligned.drop(columns=["is_perturbed"], errors="ignore")
    deltas = aligned.subtract(aligned["Control"], axis=0).drop(columns=["Control"])

    predicted = aligned["predicted_expressions"]
    control = aligned["Control"]
    gt = aligned[perturbation_group]

    predicted_delta = deltas["predicted_expressions"]
    gt_delta = deltas[perturbation_group]

    return {
        "agg_pcc": pearsonr(predicted, gt)[0],
        "delta_agg_pcc": pearsonr(predicted_delta, gt_delta)[0],
        "baseline_agg_pcc": pearsonr(control, gt)[0],
    }


def get_group_average_expressions(preds_df: pd.DataFrame):
    # can't use .mean() because we must calculate
    # sum / n_counts (not n nonzero occurrences )
    sample_counts = (
        preds_df.assign(sample_id_x=preds_df.index)
        .groupby("perturbed_genes")["sample_id_x"]
        .nunique()
    )
    grouped = preds_df.groupby(["perturbed_genes", "input_genes"]).sum(
        numeric_only=True
    )
    sample_counts_aligned = grouped.index.get_level_values("perturbed_genes").map(
        sample_counts
    )
    grouped = grouped.div(sample_counts_aligned, axis=0)
    return grouped


def get_mean_expressions(group_means_ad: AnnData):
    mean_expressions = group_means_ad.to_df().T
    all_zero = (mean_expressions == 0).all(axis=1)
    mean_expressions = mean_expressions[~all_zero]
    return mean_expressions
