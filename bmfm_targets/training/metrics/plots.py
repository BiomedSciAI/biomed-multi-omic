"""
Plots that are generated in the course of training.

Generally based on the collected labels and predictions of a validation run.
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

from bmfm_targets.training.metrics.metric_functions import calculate_95_ci


def make_heatmap_plot(distances_df, x_label, y_label, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(distances_df, cmap="coolwarm", aspect="auto", origin="lower")
    plt.colorbar(im, ax=ax, label="L1 Distance")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if distances_df.shape[0] <= 60:  # skip labels for large data
        ax.set_xticks(range(len(distances_df.columns)))
        ax.set_xticklabels(distances_df.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(distances_df.index)))
        ax.set_yticklabels(distances_df.index, fontsize=8)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title(title)
    plt.tight_layout()
    return plt.gcf()


def make_predictions_gt_density_plot(
    predictions_df: pd.DataFrame,
    predicted_label="predicted_expressions",
    gt_label="label_expressions",
    kind="hist",
    include_x_y_line=True,
    max_points=1_000_000,
):
    sb.set_context("paper")
    if predictions_df.shape[0] > max_points:
        sampled_df = predictions_df.sample(n=max_points, random_state=42)
    else:
        sampled_df = predictions_df
    g = sb.jointplot(
        data=sampled_df,
        y=predicted_label,
        x=gt_label,
        height=8,
        kind=kind,
        color="blue",
        marginal_kws={"bins": 40, "fill": True},
    )
    if include_x_y_line:
        x_min, x_max = (
            predictions_df[predicted_label].min(),
            predictions_df[predicted_label].max(),
        )
        y_min, y_max = predictions_df[gt_label].min(), predictions_df[gt_label].max()

        min_val = min(x_min, y_min)
        max_val = max(x_max, y_max)

        g.ax_joint.plot(
            [min_val, max_val],
            [min_val, max_val],
            color="gray",
            linestyle="--",
            linewidth=1,
        )

    g.set_axis_labels(gt_label, predicted_label, fontsize=12)
    plt.subplots_adjust(top=0.9)
    return plt.gcf()


def make_top20_deg_perturbation_plot(
    predictions_top20: pd.DataFrame,
    mean_control: pd.Series,
    double_boxplot=False,
    title="",
):
    predictions_top20 = predictions_top20.rename(columns={"input_genes": "gene"})
    predictions_top20 = predictions_top20.set_index("gene", append=True)[
        ["predicted_expressions", "label_expressions"]
    ]

    # index names must match for `.sub` broadcast to work
    mean_control.index.name = "gene"
    top20_genes_delta = predictions_top20.sub(mean_control, axis=0).reset_index("gene")

    sb.set_context("talk")
    plt.figure(figsize=(18, 6))
    if double_boxplot:
        melted_df = pd.melt(
            top20_genes_delta,
            id_vars=["gene"],
            value_vars=["predicted_expressions", "label_expressions"],
            var_name="expression_type",
            value_name="expression_value",
        )
        sb.boxplot(
            data=melted_df, x="gene", y="expression_value", hue="expression_type"
        )
    else:
        sb.boxplot(
            data=top20_genes_delta, x="gene", y="label_expressions", color="lightblue"
        )

    means = top20_genes_delta.groupby("gene")["predicted_expressions"].mean()
    plt.plot(means.index, means.values, "o", color="red", label="Predicted Mean")
    plt.legend(loc="best")
    plt.axhline(0, ls="--", color="green")
    plt.suptitle(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    return plt.gcf()


def make_accuracy_by_target_plot(preds_df, label_column_name):
    prediction_col = label_column_name + "_prediction"
    label_col = label_column_name + "_label"

    if prediction_col is None or label_col is None:
        return None

    if not isinstance(prediction_col, str) or not isinstance(label_col, str):
        return None

    if preds_df[label_col].nunique() > 50:
        return None

    accuracy_df = (
        preds_df.sort_values(label_col)
        .groupby(label_col)
        .apply(
            lambda group: pd.Series(
                {
                    "accuracy": calc_accuracy(group[prediction_col], group.name),
                    "count": len(group),
                }
            )
        )
        .reset_index()
    )

    # Compute estimates and confidence intervals
    estimates = []
    lower_errors = []
    upper_errors = []

    for value, n in zip(accuracy_df["accuracy"], accuracy_df["count"]):
        estimate, lower_CI, upper_CI = calculate_95_ci(value, n, ci_method="wilson")
        estimates.append(estimate)
        lower_errors.append(estimate - lower_CI)
        upper_errors.append(upper_CI - estimate)

    lower_errors = [max(0, err) for err in lower_errors]
    upper_errors = [max(0, err) for err in upper_errors]
    # Create error bars
    display_labels = accuracy_df[label_col]
    fig, ax = plt.subplots(figsize=(10, 5))
    y_positions = np.arange(len(display_labels))  # Positions for x-axis

    ax.errorbar(
        y_positions,
        estimates,
        yerr=np.array([lower_errors, upper_errors]),
        fmt="o",
        capsize=5,
        color="b",
    )

    ax.set_xticks(y_positions)
    ax.set_xticklabels(display_labels, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")

    return fig


def calc_accuracy(series, label):
    return (series == label).mean()
