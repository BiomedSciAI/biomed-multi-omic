import logging
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata


def fast_group_roc_auc_vectorized(
    predictions: pd.DataFrame, group_col, true_col, score_col
):
    predictions = predictions[[group_col, true_col, score_col]].copy()

    # 1 = zero, 0 = non-zero
    predictions[true_col] = (predictions[true_col] == 0).astype(int)
    results = {}

    for g, group in predictions.groupby(group_col):
        results[g] = mann_whitney_roc_auc(
            group[true_col].values, group[score_col].values
        )

    return pd.Series(results)


def mann_whitney_roc_auc(y_true, y_score):
    # Skip groups with only one class
    if y_true.min() == y_true.max():
        return np.nan
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    ranks = rankdata(y_score)
    pos_ranks = ranks[y_true == 1]

    auc = (pos_ranks.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc


def _resolve_input_expression_column(exp_preds: pd.DataFrame) -> str:
    for candidate in ["input_expressions", "control_expressions", "label_expressions"]:
        if candidate in exp_preds.columns:
            return candidate
    raise ValueError("No input_expressions column found.")


def get_gene_level_expression_error(exp_preds: pd.DataFrame) -> pd.DataFrame:
    metrics = {}
    input_expr_col = _resolve_input_expression_column(exp_preds)

    # === Precompute frequency ===
    grouped_all = exp_preds.groupby("input_genes")
    metrics["gene_freq"] = grouped_all[input_expr_col].count()

    # === Null baselines ===
    avg_label_expr = grouped_all["label_expressions"].mean()
    avg_nz_label_expr = (
        exp_preds.query("label_expressions > 0")
        .groupby("input_genes")["label_expressions"]
        .mean()
    )

    exp_preds = exp_preds.assign(
        avg_expressions=exp_preds["input_genes"].map(avg_label_expr),
        avg_nz_expressions=exp_preds["input_genes"].map(avg_nz_label_expr),
    )

    # === Absolute Errors ===
    exp_preds = exp_preds.assign(
        abs_diff=(
            exp_preds["predicted_expressions"] - exp_preds["label_expressions"]
        ).abs(),
        null_diff=(exp_preds["avg_expressions"] - exp_preds["label_expressions"]).abs(),
        nz_null_diff=(
            exp_preds["avg_nz_expressions"] - exp_preds["label_expressions"]
        ).abs(),
    )

    if "logits_expressions_regression" in exp_preds.columns:
        exp_preds["abs_diff_regression_logits"] = (
            exp_preds["logits_expressions_regression"] - exp_preds["label_expressions"]
        ).abs()
        metrics["gene_err_nz_by_logits"] = (
            exp_preds.query("label_expressions > 0")
            .groupby("input_genes")["abs_diff_regression_logits"]
            .mean()
        )

    # === Now do all groupings on the updated DataFrame ===
    grouped_all = exp_preds.groupby("input_genes")
    grouped_nz = exp_preds.query("label_expressions > 0").groupby("input_genes")

    metrics["gene_nz_freq"] = grouped_nz[input_expr_col].count()
    metrics["gene_err_nz"] = grouped_nz["abs_diff"].mean()
    metrics["gene_err_nz_null"] = grouped_nz["nz_null_diff"].mean()
    metrics["gene_err"] = grouped_all["abs_diff"].mean()
    metrics["gene_err_null"] = grouped_all["null_diff"].mean()

    # === Zero classification ===
    is_label_zero = exp_preds["label_expressions"] == 0
    is_pred_zero = exp_preds["predicted_expressions"] == 0

    zero_df = pd.DataFrame(
        {
            "input_genes": exp_preds["input_genes"],
            "tp": (is_label_zero & is_pred_zero).astype(int),
            "fp": (~is_label_zero & is_pred_zero).astype(int),
            "fn": (is_label_zero & ~is_pred_zero).astype(int),
        }
    )
    zero_counts = zero_df.groupby("input_genes")[["tp", "fp", "fn"]].sum()

    precision = zero_counts["tp"] / (zero_counts["tp"] + zero_counts["fp"]).replace(
        0, np.nan
    )
    recall = zero_counts["tp"] / (zero_counts["tp"] + zero_counts["fn"]).replace(
        0, np.nan
    )
    f1 = 2 * (precision * recall) / (precision + recall).replace(0, np.nan)
    metrics["is_zero_f1"] = f1

    is_zero_logits_col = next(
        (c for c in exp_preds.columns if "logits_expressions_is_zero" in c), None
    )
    if is_zero_logits_col is not None:
        try:
            metrics["is_zero_roc_auc"] = fast_group_roc_auc_vectorized(
                exp_preds,
                group_col="input_genes",
                true_col="label_expressions",
                score_col=is_zero_logits_col,
            )

        except Exception as e:
            logging.warning(f"Failed to calculate roc_auc with error {e}")

    return pd.DataFrame(metrics)


def get_best_and_worst_genes(
    gene_level_err: pd.DataFrame, topk=30, commonness_quantile=0.9
):
    common_genes = gene_level_err[
        gene_level_err.gene_nz_freq
        > gene_level_err.gene_nz_freq.quantile(commonness_quantile)
    ]
    common_genes = common_genes.assign(
        gene_err_avg=0.5 * (gene_level_err["gene_err_nz"] + gene_level_err["gene_err"])
    )
    worst_genes = common_genes.sort_values("gene_err_avg", ascending=False).head(topk)
    best_genes = common_genes.sort_values("gene_err_avg").head(topk)
    return best_genes, worst_genes


def _get_label_column_idx(columns: list) -> int:
    """Find the index of the label column for filtering (-100)."""
    label_col = next((c for c in columns if c.startswith("label_")), None)
    if label_col is None:
        raise ValueError("No label_ column found in columns list.")
    return columns.index(label_col)


def create_field_predictions_df(
    predictions_list, id2gene, columns, sample_names=None, include_nonmasked=False
):
    logging.info(f"Preparing to concat {len(predictions_list)} batches")
    predictions_array = torch.concat([*predictions_list]).to(torch.float32).numpy()
    reshaped = predictions_array.reshape(-1, predictions_array.shape[-1])

    n_samples, n_genes = predictions_array.shape[:2]
    sample_ids = np.repeat(
        sample_names if sample_names is not None else np.arange(n_samples), n_genes
    )

    # Adjust columns to match prediction shape
    columns = list(columns)
    while len(columns) < reshaped.shape[1]:
        columns.append(f"logits_{len(columns)}")
    while len(columns) > reshaped.shape[1]:
        dropped = columns.pop()
        warnings.warn(f"Too many columns; dropped {dropped}")

    # Mask -100 entries early
    if not include_nonmasked:
        label_idx = _get_label_column_idx(columns)
        mask = reshaped[:, label_idx] != -100
        reshaped = reshaped[mask]
        sample_ids = sample_ids[mask]
        logging.info(f"Filtered -100s, remaining shape: {reshaped.shape}")

    # Create DataFrame
    preds_df = pd.DataFrame(reshaped, columns=columns)
    preds_df["sample_id"] = sample_ids

    # Gene ID mapping
    if "gene_id" not in preds_df.columns:
        raise ValueError("Expected 'gene_id' column missing")

    preds_df["gene_id"] = preds_df["gene_id"].astype(int)
    preds_df["input_genes"] = preds_df["gene_id"].map(id2gene)
    preds_df = preds_df.drop(columns=["gene_id"])

    # Map other *_genes columns if needed
    for col in preds_df.columns:
        if "genes" in col and col != "input_genes":
            preds_df[col] = pd.to_numeric(preds_df[col], errors="coerce")
            preds_df[col] = preds_df[col].map(id2gene)

    preds_df = preds_df.set_index("sample_id")
    logging.info(f"Final predictions_df shape: {preds_df.shape}")
    return preds_df


def concat_field_loss_batch_tensors(
    input_ids: torch.Tensor, labels: torch.Tensor, predictions: torch.Tensor, **kwargs
):
    batch_size, num_fields, sequence_length = input_ids.shape

    tensors_to_cat = [
        input_ids.permute(0, 2, 1),
        predictions.reshape(batch_size, sequence_length, 1),
        labels.reshape(batch_size, sequence_length, 1),
    ]
    for key, value in sorted(kwargs.items()):
        tensors_to_cat.append(value.reshape(batch_size, sequence_length, 1))

    batch_tensor = torch.concat(tensors_to_cat, dim=-1)
    return batch_tensor


def concat_wced_field_loss_batch_tensors(
    labels: torch.Tensor, predictions: torch.Tensor, **kwargs
):
    """
    Process WCED batch tensors assuming:
    - predictions: [batch_size, vocab_size] - regression-like predictions
    - labels: [batch_size, vocab_size] - with -100 for tokens with no label values.

    """
    batch_size, vocab_size = predictions.shape

    # Create vocab indices tensor [batch_size, vocab_size]
    # This replaces the role of input_ids from MLM batch tracking
    vocab_indices = (
        torch.arange(vocab_size, device=predictions.device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )

    # Shape: [batch_size, vocab_size, num_features]
    tensors_to_stack = [
        vocab_indices.unsqueeze(-1),  # gene_id (vocab index)
        predictions.unsqueeze(-1),  # Prediction values
        labels.unsqueeze(-1),  # Labels (including -100)
    ]

    # Add any additional tensors from kwargs (e.g., logits)
    for key, value in sorted(kwargs.items()):
        # Assume kwargs tensors are also [batch_size, vocab_size]
        if len(value.shape) < 3:
            value = value.unsqueeze(-1)
        tensors_to_stack.append(value)

    # Shape: [batch_size, vocab_size, num_features]
    stacked_tensor = torch.cat(tensors_to_stack, dim=-1)

    return stacked_tensor


def concat_label_loss_batch_tensors(
    input_ids: torch.Tensor, labels: torch.Tensor, predictions: torch.Tensor, **kwargs
):
    batch_size, num_fields, sequence_length = input_ids.shape

    tensors_to_cat = [
        predictions.reshape(batch_size, 1),
        labels.reshape(batch_size, 1),
    ]
    for key, value in sorted(kwargs.items()):
        tensors_to_cat.append(value.reshape(batch_size, -1))

    batch_tensor = torch.concat(tensors_to_cat, dim=-1)
    return batch_tensor


def field_predictions_df_columns(fields, this_field, modeling_strategy):
    one_dim_decode_modes = ["regression", "is_zero", "mvc_regression", "mvc_is_zero"]
    logits_columns = [
        f"logits_{this_field.field_name}_{m}"
        for m in sorted(this_field.decode_modes)
        if m in one_dim_decode_modes
    ]
    if "wced" in this_field.decode_modes:
        logits_columns = [
            f"logits_{this_field.field_name}_{m}"
            for m in this_field.decode_modes["wced"].get("logit_outputs", [])
        ]
    input_field_names = [f.field_name for f in fields if f.is_input]
    field_column_map = {
        "mlm": {"genes": "gene_id", "expressions": "input_expressions"},
        "sequence_labeling": {
            "genes": "gene_id",
            "expressions": "control_expressions",
            "perturbations": "is_perturbed",
        },
    }
    field_column_map["multitask"] = field_column_map["mlm"]

    # check if we are in wced mode
    if not this_field.is_masked and this_field.is_input:
        input_columns = ["gene_id"]
    else:
        input_columns = [
            field_column_map[modeling_strategy][fn] for fn in input_field_names
        ]
    if modeling_strategy in ("mlm", "multitask"):
        output_columns = [
            f"predicted_{this_field.field_name}",
            f"label_{this_field.field_name}",
        ]

    elif modeling_strategy == "sequence_labeling":
        output_columns = ["predicted_expressions", "label_expressions"]
    else:
        raise ValueError("No known predictions df columns for modeling strategy")

    return input_columns + output_columns + logits_columns


def get_gene_metrics_from_gene_errors(gene_level_err: pd.DataFrame):
    err_cols = [
        c for c in gene_level_err.columns if not "_freq" in c and not "null" in c
    ]
    metrics = gene_level_err[err_cols].mean().to_dict()
    metrics["gene_fraction_worse_than_null_nz"] = (
        gene_level_err.gene_err_nz > gene_level_err.gene_err_nz_null
    ).mean()
    metrics["gene_fraction_worse_than_null"] = (
        gene_level_err.gene_err > gene_level_err.gene_err_null
    ).mean()
    return metrics


def create_label_predictions_df(
    predictions_list, label_name, sample_names, this_label_dict
):
    preds_array = (
        torch.concat([*predictions_list]).detach().cpu().to(torch.float32).numpy()
    )
    columns = [f"{label_name}_prediction", f"{label_name}_label"]

    if preds_array.shape[1] == 3:  # regression_task
        # usually logits and predictions are the same for regression task
        # but link_function or other steps could impact it
        columns += [f"{label_name}_logits"]
    elif preds_array.shape[1] > 3:  # classification task
        label_values = [
            i[0] for i in sorted(this_label_dict.items(), key=lambda x: x[1])
        ]
        columns += [f"{label_value}_logits" for label_value in label_values]
        preds_array = preds_array.astype(object)
        preds_array[:, :2] = np.array(label_values)[preds_array[:, :2].astype(int)]
    label_preds_df = pd.DataFrame(index=sample_names, data=preds_array, columns=columns)
    return label_preds_df
