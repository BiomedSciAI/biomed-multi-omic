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


def _core_metrics(predicted: pd.Series, gt: pd.Series) -> dict:
    """
    Compute core aggregated PCC and MAE between predictions and ground truth.

    Args:
    ----
        predicted (pd.Series): Predicted expression values.
        gt (pd.Series): Ground-truth expression values.

    Returns:
    -------
        dict: ``{"agg_pcc": float, "agg_mae": float}``
    """
    return {
        "agg_pcc": pearsonr(predicted, gt)[0],
        "agg_mae": mean_absolute_error(gt, predicted),
    }


def _baseline_metrics(control: pd.Series, gt: pd.Series, predicted: pd.Series) -> dict:
    """
    Compute baseline and delta metrics using unperturbed control expressions.

    Args:
    ----
        control (pd.Series): Unperturbed (control / input) expression values.
        gt (pd.Series): Ground-truth (label) expression values.
        predicted (pd.Series): Predicted expression values.

    Returns:
    -------
        dict: ``baseline_agg_pcc``, ``baseline_agg_mae``, ``delta_agg_pcc``.
    """
    predicted_delta = predicted - control
    gt_delta = gt - control
    return {
        "baseline_agg_pcc": pearsonr(control, gt)[0],
        "baseline_agg_mae": mean_absolute_error(gt, control),
        "delta_agg_pcc": pearsonr(predicted_delta, gt_delta)[0],
    }


def _delta_metrics(baseline: pd.Series, gt: pd.Series) -> dict:
    """
    Compute PCC and MAE against the average-perturbation (train) baseline.

    Args:
    ----
        baseline (pd.Series): Average-perturbation baseline expression values.
        gt (pd.Series): Ground-truth (label) expression values.

    Returns:
    -------
        dict: ``baseline_agg_pcc_from_avg_perturbation``,
            ``baseline_agg_mae_from_avg_perturbation``.
    """
    return {
        "baseline_agg_pcc_from_avg_perturbation": pearsonr(baseline, gt)[0],
        "baseline_agg_mae_from_avg_perturbation": mean_absolute_error(gt, baseline),
    }


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

    Core metrics (``agg_pcc``, ``agg_mae``) are always computed.  Baseline and
    delta metrics are only included when the required columns are present in
    *grouped_predictions*, so callers that lack ``input_expressions`` or
    ``baseline_expressions`` (e.g. ChIP tissue predictions) never raise.

    Parameters
    ----------
    grouped_predictions : pd.DataFrame
        Observed expressions with rows as genes and columns including at minimum
        ``predicted_expressions`` and ``label_expressions``.  May also contain
        ``input_expressions`` and/or ``baseline_expressions`` for the full
        perturbation metric set.  Already averaged across samples with the same
        condition (pseudobulk).

    Returns
    -------
    dict
        Always contains ``agg_pcc`` and ``agg_mae``.
        Contains ``baseline_agg_pcc``, ``baseline_agg_mae``, ``delta_agg_pcc``
        only when ``input_expressions`` is present.
        Contains ``baseline_agg_pcc_from_avg_perturbation`` and
        ``baseline_agg_mae_from_avg_perturbation`` only when
        ``baseline_expressions`` is present.
    """
    cols = grouped_predictions.columns
    predicted = grouped_predictions["predicted_expressions"]
    gt = grouped_predictions["label_expressions"]

    result: dict = _core_metrics(predicted, gt)

    if "input_expressions" in cols:
        control = grouped_predictions["input_expressions"]
        result |= _baseline_metrics(control, gt, predicted)

    if "baseline_expressions" in cols:
        baseline = grouped_predictions["baseline_expressions"]
        result |= _delta_metrics(baseline, gt)

    return result


def get_grouped_predictions(
    preds_df: pd.DataFrame,
    grouped_ground_truth: pd.DataFrame,
    *,
    group_column: str = "perturbed_genes",
    feature_column: str = "input_genes",
) -> pd.DataFrame:
    """
    Group predictions by condition, filling in missing predictions with baseline.

    Baseline is either the "Average_Perturbation_Train" or "Control" column from
    the grouped_ground_truth DataFrame.

    Args:
    ----
        preds_df (pd.DataFrame): DataFrame containing predicted expressions for all
          genes and samples.
        grouped_ground_truth (pd.DataFrame): DataFrame containing mean ground truth
            expressions for each of the conditions, the whole train set, and the
            control.  The values in the column corresponding to each condition are
            treated as the ground truth for that condition.
            We also use the "Average_Perturbation_Train" or "Control" column from
            this dataframe to fill in our best guess prediction for genes that were
            not predicted.
        group_column (str): Column in *preds_df* that identifies the condition/group
            axis of the MultiIndex (e.g. ``"perturbed_genes"``, ``"tissue_label"``).
            Defaults to ``"perturbed_genes"`` to preserve existing behavior.
        feature_column (str): Column in *preds_df* that identifies the feature axis
            of the MultiIndex (e.g. ``"input_genes"``).
            Defaults to ``"input_genes"`` to preserve existing behavior.

    Returns:
    -------
        pd.DataFrame: A DataFrame with a MultiIndex of (group_column, feature_column)
        and columns for 'predicted_expressions' and 'label_expressions'.
        Also included are "input_expressions" and "baseline_expressions" for downstream
        metric calculation.

    """
    # Extract unique group identifiers from the predictions
    perts = preds_df[group_column].unique()
    # use genes in grouped_ground_truth will be to create grouped predictions
    genes = grouped_ground_truth.index

    # Determine baseline (Average_Perturbation_Train or Control or None)
    has_apt = "Average_Perturbation_Train" in grouped_ground_truth.columns
    has_control = "Control" in grouped_ground_truth.columns
    if has_apt:
        baseline = grouped_ground_truth["Average_Perturbation_Train"]
    elif has_control:
        baseline = grouped_ground_truth["Control"]
        logging.warning(
            "Average_Perturbation_Train not found, using Control as baseline"
        )
    else:
        baseline = None

    # Initialize group_averages DataFrame
    # Use np.tile to repeat the baseline array for each perturbation when available.
    # When no baseline exists (e.g. ChIP tissue case), initialize with zeros so that
    # actual predictions overwrite the placeholder completely.
    n_total = len(perts) * len(genes)
    initial_data: dict = {}
    if baseline is not None:
        initial_data["predicted_expressions"] = np.tile(baseline.to_numpy(), len(perts))
        initial_data["baseline_expressions"] = np.tile(baseline.to_numpy(), len(perts))
    else:
        initial_data["predicted_expressions"] = np.zeros(n_total)
    if has_control:
        initial_data["input_expressions"] = np.tile(
            grouped_ground_truth["Control"].to_numpy(), len(perts)
        )
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
            [perts, genes], names=[group_column, feature_column]
        ),
        data=initial_data,
    )
    if "predicted_expressions" in preds_df.columns:
        # Compute mean predictions for each group
        mean_preds = preds_df.groupby([group_column, feature_column])[
            "predicted_expressions"
        ].mean()

        # Update group_averages with mean predictions
        group_averages.loc[mean_preds.index, "predicted_expressions"] = mean_preds

    elif "predicted_delta_baseline_expressions" in preds_df.columns:
        mean_predicted_delta = preds_df.groupby([group_column, feature_column])[
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
