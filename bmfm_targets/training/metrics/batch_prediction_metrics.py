import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score


def group_is_zero_f1(group: pd.Series):
    y_true = (group["label_expressions"] == 0).astype(int)
    y_pred = (group["predicted_expressions"] == 0).astype(int)
    return f1_score(y_true, y_pred, zero_division=np.nan)


def group_roc_auc(group: pd.Series):
    y_true = (group["label_expressions"] == 0).astype(int)
    y_pred = group["logits_expressions_is_zero"]
    if len({*y_true}) == 1:
        return np.nan
    return roc_auc_score(y_true, y_pred)


def get_gene_level_expression_error(exp_preds: pd.DataFrame):
    metrics = {}
    # first get counts total and non-zero
    if "input_expressions" in exp_preds.columns:
        input_expressions_col_name = "input_expressions"
    elif "control_expressions" in exp_preds.columns:
        input_expressions_col_name = "control_expressions"
    else:
        raise ValueError("No input_expressions_column available!")
    metrics["gene_freq"] = (
        exp_preds.groupby("input_genes").count()[input_expressions_col_name].astype(int)
    )
    metrics["gene_nz_freq"] = (
        exp_preds.query("label_expressions > 0")
        .groupby("input_genes")
        .count()[input_expressions_col_name]
        .astype(int)
    )

    # for null approximation consider the avg value and avg nonzero value
    avg_label_expressions = exp_preds.groupby("input_genes").label_expressions.mean()
    avg_nz_label_expressions = (
        exp_preds.query("label_expressions > 0")
        .groupby("input_genes")
        .label_expressions.mean()
    )

    # align null predictions with gene names
    exp_preds_avg = exp_preds.assign(
        avg_nz_expressions=exp_preds.input_genes.map(avg_nz_label_expressions),
        avg_expressions=exp_preds.input_genes.map(avg_label_expressions),
    )
    # calculate pointwise abs error
    exp_preds_diff = exp_preds_avg.assign(
        abs_diff=(
            exp_preds_avg["predicted_expressions"] - exp_preds_avg["label_expressions"]
        ).abs(),
        nz_null_diff=(
            exp_preds_avg["avg_nz_expressions"] - exp_preds_avg["label_expressions"]
        ).abs(),
        null_diff=(
            exp_preds_avg["avg_expressions"] - exp_preds_avg["label_expressions"]
        ).abs(),
    )
    if "logits_expressions_regression" in exp_preds.columns:
        exp_preds_diff = exp_preds_diff.assign(
            abs_diff_regression_logits=(
                exp_preds_avg["logits_expressions_regression"]
                - exp_preds_avg["label_expressions"]
            ).abs()
        )
        metrics["gene_err_nz_by_logits"] = (
            exp_preds_diff.query("label_expressions > 0")
            .groupby("input_genes")
            .abs_diff_regression_logits.mean()
        )

    # calculate the average error limited to the nonzero expressions
    metrics["gene_err_nz_null"] = (
        exp_preds_diff.query("label_expressions > 0")
        .groupby("input_genes")
        .nz_null_diff.mean()
    )
    metrics["gene_err_nz"] = (
        exp_preds_diff.query("label_expressions > 0")
        .groupby("input_genes")
        .abs_diff.mean()
    )

    # calculate the average error across all label expressions
    metrics["gene_err_null"] = exp_preds_diff.groupby("input_genes").null_diff.mean()
    metrics["gene_err"] = exp_preds_diff.groupby("input_genes").abs_diff.mean()

    # calculate gene-wise f1 score
    metrics["is_zero_f1"] = exp_preds.groupby("input_genes").apply(group_is_zero_f1)
    if "logits_expressions_is_zero" in exp_preds.columns:
        metrics["is_zero_roc_auc"] = exp_preds.groupby("input_genes").apply(
            group_roc_auc
        )

    gene_level_err = pd.DataFrame(metrics)

    return gene_level_err


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


def create_field_predictions_df(
    predictions_list, id2gene, columns, sample_names=None, include_nonmasked=False
):
    predictions_array = (
        torch.concat([*predictions_list]).detach().cpu().to(torch.float32).numpy()
    )
    reshaped_arr = predictions_array.reshape(-1, predictions_array.shape[-1])
    if sample_names is not None:
        assert len(sample_names) == predictions_array.shape[0]
    else:
        sample_names = np.arange(predictions_array.shape[0])
    sample_ids = np.repeat(sample_names, predictions_array.shape[1])
    predictions_df = pd.DataFrame(reshaped_arr, columns=columns, index=sample_ids)
    predictions_df = (
        predictions_df.reset_index()
        .rename(columns={"index": "sample_id"})
        .astype({"gene_id": int})
        .set_index("sample_id")
    )
    if include_nonmasked == False:
        label_column = [c for c in predictions_df.columns if c.startswith("label_")][0]
        predictions_df = predictions_df[predictions_df[label_column] != -100]

    input_genes = predictions_df.pop("gene_id").map(id2gene)

    for col in predictions_df.columns:
        if "genes" in col:
            predictions_df[col] = predictions_df[col].map(id2gene)

    return predictions_df.assign(input_genes=input_genes)


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
    one_dim_decode_modes = ["regression", "is_zero"]
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
    logits_columns = [
        f"logits_{this_field.field_name}_{m}"
        for m in sorted(this_field.decode_modes)
        if m in one_dim_decode_modes
    ]
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


def concat_batch_tensors(batch, outputs, predictions, loss_task):
    output_key = loss_task.output_key
    from .loss_handling import FieldLossTask, LabelLossTask

    if isinstance(loss_task, LabelLossTask):
        logits_to_record = {k: v for k, v in outputs.logits.items() if k == output_key}
        batch_tensors = (
            concat_label_loss_batch_tensors(
                input_ids=batch["input_ids"],
                predictions=predictions[output_key],
                labels=batch["labels"][output_key],
                **logits_to_record,
            )
            .cpu()
            .detach()
        )
    elif isinstance(loss_task, FieldLossTask):
        logits_to_record = {
            k: v
            for k, v in outputs.logits.items()
            if k.startswith(output_key) and (v.shape[-1] == 1)
        }
        batch_tensors = (
            concat_field_loss_batch_tensors(
                input_ids=batch["input_ids"],
                predictions=predictions[output_key],
                labels=batch["labels"][output_key],
                **logits_to_record,
            )
            .cpu()
            .detach()
        )

    return batch_tensors


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
