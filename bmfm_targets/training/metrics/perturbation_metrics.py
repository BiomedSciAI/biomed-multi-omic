"""
Functions for calculating perturbation-specific metrics.

Perturbation metrics are different because they involve various levels of aggregation
across samples.

Typically the predictions for a given perturbation are averaged into one group prediction
and compared with the ground-truth prediction.

Also, the predictions are compared against the average, unperturbed expression.
"""
import logging

import numpy as np
import pandas as pd
from scanpy import AnnData
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import pairwise_distances as pwd


def prepare_args_for_discrimination_score(group_expressions: pd.DataFrame):
    """
    Prepare arguments for discrimination score calculation.

    Args:
    ----
        group_expressions (pd.DataFrame): DataFrame produced by function
            `get_grouped_predictions`

    """
    group_expressions = group_expressions.copy()  # so we can change the index

    perturbations = group_expressions.index.get_level_values(0).unique().sort_values()

    # Get the set of genes available for each perturbation
    genes_sets = [
        set(group_expressions.xs(pert, level=0).index) for pert in perturbations
    ]
    genes = np.array(sorted(set.intersection(*genes_sets)))

    group_expressions.index = group_expressions.index.set_names(
        ["perturbation", "gene"]
    )

    # Real effects
    label_flat = group_expressions.loc[
        (slice(None), genes), ["label_expressions"]
    ].reset_index()
    real_df = (
        label_flat.pivot_table(
            index="gene",
            columns="perturbation",
            values="label_expressions",
        )
        .reindex(index=genes)  # enforce gene order
        .T.reindex(index=perturbations)  # enforce perturbation order
    )
    real_effects = real_df.to_numpy()

    # Predicted effects
    pred_flat = group_expressions.loc[
        (slice(None), genes), ["predicted_expressions"]
    ].reset_index()
    pred_df = (
        pred_flat.pivot_table(
            index="gene",
            columns="perturbation",
            values="predicted_expressions",
        )
        .reindex(index=genes)
        .T.reindex(index=perturbations)
    )
    pred_effects = pred_df.to_numpy()
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

    # Uniqueness checks
    assert len(np.unique(perts)) == len(perts), "Perturbation indices must be unique"
    assert len(np.unique(genes)) == len(genes), "Gene/variable indices must be unique"

    # max_rank = max(num_perts - 1, 1)
    max_rank = num_perts

    all_distances = []

    if not exclude:
        distance_matrix = pwd(real, pred, metric=metric)  # shape (P, P)
        order = np.argsort(distance_matrix, axis=0)  # row order per column
        ranks = np.empty_like(order)
        ranks[order, np.arange(num_perts)] = np.arange(num_perts)  # invert permutation
        rank_positions = ranks[np.arange(num_perts), np.arange(num_perts)]
        scores = 1 - rank_positions / max_rank
        distance_df = pd.DataFrame(distance_matrix, index=perts, columns=perts)
        return dict(zip(map(str, perts), scores)), distance_df

    results = {}
    all_distances = []

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
        all_distances.append(distances)

        num_better_matches = (distances < distances[pert_idx]).sum()
        score = 1 - num_better_matches / max_rank
        results[str(pert_name)] = score
    distance_df = pd.DataFrame(np.vstack(all_distances), index=perts, columns=perts)
    return results, distance_df


def get_aggregated_perturbation_metrics(grouped_predictions: pd.DataFrame):
    """
    Compute Pearson correlations between predicted, control, and ground-truth
    perturbation expression profiles over pseudobulk (mean expressions).

    Includes calculation of delta expressions, which are the differences between the
    perturbed samples and the control mean expressions.

    Parameters
    ----------
    grouped_predictions : pd.DataFrame
        Observed expressions with rows as genes and columns including
        'predicted_expressions', 'label_expressions', and "control_expressions".
        Already averaged across samples with the same perturbation (pseudobulk).

    Returns
    -------
    dict
        {
            "agg_pcc": correlation(predicted, gt),
            "delta_agg_pcc": correlation(predicted_delta, gt_delta),
            "baseline_agg_pcc": correlation(control, gt),
        }
    """
    deltas = grouped_predictions.subtract(
        grouped_predictions["control_expressions"], axis=0
    ).drop(columns=["control_expressions"])

    predicted = grouped_predictions["predicted_expressions"]
    control = grouped_predictions["control_expressions"]
    gt = grouped_predictions["label_expressions"]

    predicted_delta = deltas["predicted_expressions"]
    gt_delta = deltas["label_expressions"]

    baseline = grouped_predictions["baseline_expressions"]

    return {
        "agg_pcc": pearsonr(predicted, gt)[0],
        "baseline_agg_pcc": pearsonr(control, gt)[0],
        "baseline_agg_pcc_from_avg_perturbation": pearsonr(baseline, gt)[0],
        # delta metrics (baseline not possible)
        "delta_agg_pcc": pearsonr(predicted_delta, gt_delta)[0],
        # mae metrics
        "agg_mae": mean_absolute_error(gt, predicted),
        "baseline_agg_mae": mean_absolute_error(gt, control),
        "baseline_agg_mae_from_avg_perturbation": mean_absolute_error(gt, baseline),
    }


def get_grouped_predictions(
    preds_df: pd.DataFrame, grouped_ground_truth: pd.DataFrame
) -> pd.DataFrame:
    """
    Group predictions by perturbed genes, filling in missing predictions with baseline.

    Baseline is either the "Average_Perturbation_Train" or "Control" column from
    the grouped_ground_truth DataFrame.

    Args:
    ----
        preds_df (pd.DataFrame): DataFrame containing predicted expressions for all
          genes and samples.
        grouped_ground_truth (pd.DataFrame): DataFrame containing mean ground truth expressions
            for each of the perturbations, the whole train set, and the control.
            The values in the column corresponding to each perturbation are treated
            as the ground truth for that perturbation.
            We also use the "Average_Perturbation_Train" or "Control" column from this
            dataframe to fill in our best guess prediction for genes that were not
            predicted.

    Returns:
    -------
        pd.DataFrame: A DataFrame with a MultiIndex of (perturbed_genes, input_genes)
        and columns for 'predicted_expressions' and 'label_expressions'.
        Also included are "control_expressions" and "baseline_expressions" for downstream
        metric calculation.

    """
    # Extract unique perturbed genes from the set of predictinos
    perts = preds_df["perturbed_genes"].unique()
    # use genes in grouped_ground_truth will be to create grouped predictions
    genes = grouped_ground_truth.index

    # Determine baseline (Average_Perturbation_Train or Control)
    if "Average_Perturbation_Train" in grouped_ground_truth.columns:
        baseline = grouped_ground_truth["Average_Perturbation_Train"]
    else:
        baseline = grouped_ground_truth["Control"]
        logging.warning(
            "Average_Perturbation_Train not found, using Control as baseline"
        )

    # Initialize group_averages DataFrame
    # Use np.tile to repeat the baseline array for each perturbation
    initial_data = {
        "predicted_expressions": np.tile(baseline.to_numpy(), len(perts)),
        "control_expressions": np.tile(
            grouped_ground_truth["Control"].to_numpy(), len(perts)
        ),
        "baseline_expressions": np.tile(baseline.to_numpy(), len(perts)),
    }
    if all(pert in grouped_ground_truth.columns for pert in perts):
        initial_data["label_expressions"] = np.hstack(
            [grouped_ground_truth[pert].to_numpy() for pert in perts]
        )
    else:
        logging.warning(
            "Not all perturbations have label_expressions, "
            "pseudobulk metrics cannot be calculated"
        )
    group_averages = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [perts, genes], names=["perturbed_genes", "input_genes"]
        ),
        data=initial_data,
    )
    if "predicted_expressions" in preds_df.columns:
        # Compute mean predictions for each group
        mean_preds = preds_df.groupby(["perturbed_genes", "input_genes"])[
            "predicted_expressions"
        ].mean()

        # Update group_averages with mean predictions
        group_averages.loc[mean_preds.index, "predicted_expressions"] = mean_preds

    elif "predicted_delta_baseline_expressions" in preds_df.columns:
        mean_predicted_delta = preds_df.groupby(["perturbed_genes", "input_genes"])[
            "predicted_delta_baseline_expressions"
        ].mean()

        # Update group_averages with mean predictions
        group_averages.loc[
            mean_predicted_delta.index, "predicted_expressions"
        ] += mean_predicted_delta
    else:
        raise ValueError(
            f"No usable predictions found in dataframe. Columns: {preds_df.columns}"
        )

    assert not group_averages.isna().any().any(), f"{group_averages.isna().any()}"

    if not all(p in grouped_ground_truth.columns for p in perts):
        logging.warning(
            "No ground truth pseudobulk found for predictions, "
            "dataframe will not have 'label_expressions' column"
        )
        return group_averages

    return group_averages


def get_grouped_ground_truth(group_means_ad: AnnData, remove_always_zero=False):
    """
    Get ground truth group means from AnnData object.

    Removes genes that are always zero to avoid NaNs in metrics.

    Args:
    ----
        group_means_ad (AnnData): group_means attribute from a BasePerturbationDataset instance.
        remove_always_zero (bool): whether to remove genes that are always zero in all
            pseudobulks (ie genes in the library that were never measured in any experiment)
            For producing final predictions to match a gene list this should be false,
            but for calculating metrics it should be true as these genes do not participate
            in the study.

    Returns:
    -------
        DataFrame: dataframe of mean expressions, keyed for joining downstream

    """
    mean_expressions = group_means_ad.to_df().T
    if remove_always_zero:
        all_zero = (mean_expressions == 0).all(axis=1)
        mean_expressions = mean_expressions[~all_zero]
    return mean_expressions
