#!/usr/bin/env python3
"""
MoA Classification Comparison Script for Multiple Embeddings.

Compares the embeddings named by --embedding_keys for mechanism of action (MoA)
classification using stratified drug group cross-validation, and writes metrics, CSVs and
diagnostic plots. Each CV fold leaves out at least one drug from each MoA group. The
filtering and CV code is in moa_lib.py; the Figure 6 pipeline is the run_*.py scripts.

Usage:
    python moa_predict.py --input data.h5ad --output results/ --embedding_keys X_a X_b
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scib.metrics  # noqa: F401  importing scib sets a larger plot style; kept so these plots are unchanged
import seaborn as sns
from moa_lib import (  # noqa: F401  (re-exported for callers of this script)
    create_stratified_drug_folds,
    load_and_prepare_data,
    scib_bio_metrics,
    stratified_drug_group_cv,
)
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder, label_binarize

warnings.filterwarnings("ignore")

# Set style for Nature Methods-style plots
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
sns.set_palette("colorblind")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare embedding performance for MoA classification"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input AnnData file (.h5ad)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/moa_comparison",
        help="Output directory for results",
    )
    parser.add_argument(
        "--embedding_keys",
        type=str,
        nargs="+",
        default=["X_bmfm_wced_cls_0", "X_bmfm_mlm"],
        help="List of embedding keys to compare (default: X_bmfm_wced_cls_0 X_bmfm_mlm)",
    )
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=5,
        help="Number of cross-validation repeats (default: 5)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Maximum iterations for logistic regression (default: 1000)",
    )
    parser.add_argument(
        "--include_baseline",
        action="store_true",
        help="Include raw gene expression baseline in comparison (default: False)",
    )

    # Data filtering parameters
    parser.add_argument(
        "--cell_type",
        type=str,
        default=None,
        help="Cell type to subset data by (filters adata.obs['cell_type'] column)",
    )
    parser.add_argument(
        "--skip_vehicle_filter",
        action="store_true",
        help="Skip filtering out vehicle control (default: False)",
    )
    parser.add_argument(
        "--dose",
        type=float,
        nargs="+",
        default=[10000],
        help="Dose values to filter for (default: 10000)",
    )
    parser.add_argument(
        "--viability_threshold",
        type=float,
        default=0.85,
        help="Viability threshold for filtering cells (default: 0.85)",
    )
    parser.add_argument(
        "--skip_viability_filter",
        action="store_true",
        help="Skip viability filtering (default: False)",
    )
    parser.add_argument(
        "--skip_others_filter",
        action="store_true",
        help="Skip removing 'Others' MoA class (default: False)",
    )
    parser.add_argument(
        "--min_drugs_per_moa",
        type=int,
        default=2,
        help="Minimum number of drugs per MoA class (default: 2)",
    )

    return parser.parse_args()


def calculate_metrics(results: dict[str, list]) -> dict[str, float]:
    """
    Calculate classification metrics from CV results.

    Parameters
    ----------
    results : Dict[str, List]
        Results from cross-validation

    Returns
    -------
    metrics : Dict[str, float]
        Dictionary of calculated metrics
    """
    y_true = np.array(results["y_true"])
    y_pred = np.array(results["y_pred"])

    # Concatenate probability arrays
    y_prob = np.vstack(results["y_prob"])

    # Calculate metrics
    metrics = {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "macro_recall": recall_score(y_true, y_pred, average="macro"),
        "weighted_recall": recall_score(y_true, y_pred, average="weighted"),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    # ROC AUC (one-vs-rest for multiclass)
    try:
        metrics["roc_auc_ovr"] = roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="macro"
        )
    except ValueError:
        metrics["roc_auc_ovr"] = np.nan

    return metrics


def calculate_per_repeat_metrics(results: dict[str, list]) -> pd.DataFrame:
    """
    Calculate metrics for each repeat separately.

    Parameters
    ----------
    results : Dict[str, List]
        Results from cross-validation

    Returns
    -------
    metrics_df : pd.DataFrame
        DataFrame with metrics per repeat
    """
    repeats = np.array(results["repeat"])
    folds = np.array(results["fold"])
    unique_repeats = np.unique(repeats)

    metrics_list = []
    for repeat in unique_repeats:
        # Get indices for this repeat
        repeat_mask = repeats == repeat
        y_true = np.array(results["y_true"])[repeat_mask]
        y_pred = np.array(results["y_pred"])[repeat_mask]

        # Get probabilities for this repeat
        # Find which fold indices belong to this repeat
        repeat_folds = np.unique(folds[repeat_mask])
        y_prob_list = [results["y_prob"][fold] for fold in repeat_folds]
        y_prob = np.vstack(y_prob_list)

        metrics = {
            "repeat": repeat,
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
            "macro_recall": recall_score(y_true, y_pred, average="macro"),
            "weighted_recall": recall_score(y_true, y_pred, average="weighted"),
            "mcc": matthews_corrcoef(y_true, y_pred),
        }

        try:
            metrics["roc_auc_ovr"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"
            )
        except ValueError:
            metrics["roc_auc_ovr"] = np.nan

        metrics_list.append(metrics)

    return pd.DataFrame(metrics_list)


def plot_data_distribution(adata: sc.AnnData, output_dir: Path):
    """
    Create plots showing distribution of drugs and MoA classes.

    Parameters
    ----------
    adata : AnnData
        Filtered AnnData object
    output_dir : Path
        Output directory
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Drugs per MoA class
    ax1 = axes[0]
    moa_drug_counts = (
        adata.obs.groupby("target")["product_name"]
        .nunique()
        .sort_values(ascending=False)
    )

    # Create bar plot with MoA names on x-axis
    x_pos = range(len(moa_drug_counts))
    ax1.bar(x_pos, moa_drug_counts.values, color="steelblue", alpha=0.7)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(moa_drug_counts.index, rotation=45, ha="right", fontsize=8)
    ax1.set_xlabel("MoA Class", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Number of Drugs", fontsize=11, fontweight="bold")
    ax1.set_title(
        f"Drugs per MoA Class (n={len(moa_drug_counts)} classes)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Add statistics text
    stats_text = f"Mean: {moa_drug_counts.mean():.1f}\nMedian: {moa_drug_counts.median():.1f}\nMin: {moa_drug_counts.min()}\nMax: {moa_drug_counts.max()}"
    ax1.text(
        0.95,
        0.95,
        stats_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        fontsize=9,
    )

    # Plot 2: Cells per MoA class
    ax2 = axes[1]
    moa_cell_counts = adata.obs["target"].value_counts().sort_values(ascending=False)

    # Create bar plot with MoA names on x-axis
    x_pos = range(len(moa_cell_counts))
    ax2.bar(x_pos, moa_cell_counts.values, color="coral", alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(moa_cell_counts.index, rotation=45, ha="right", fontsize=8)
    ax2.set_xlabel("MoA Class", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Number of Cells", fontsize=11, fontweight="bold")
    ax2.set_title(
        f"Cells per MoA Class (n={adata.n_obs} cells)", fontsize=12, fontweight="bold"
    )
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Add statistics text
    stats_text = f"Mean: {moa_cell_counts.mean():.0f}\nMedian: {moa_cell_counts.median():.0f}\nMin: {moa_cell_counts.min()}\nMax: {moa_cell_counts.max()}"
    ax2.text(
        0.95,
        0.95,
        stats_text,
        transform=ax2.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(output_dir / "data_distribution.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "data_distribution.png", dpi=300, bbox_inches="tight")
    print(f"Saved data distribution plot to {output_dir / 'data_distribution.pdf'}")
    plt.close()


def plot_roc_pr_curves(
    all_results: dict[str, dict],
    embedding_names: list[str],
    output_dir: Path,
    n_classes: int,
):
    """
    Create ROC and Precision-Recall curves comparing all models.

    Parameters
    ----------
    all_results : Dict[str, Dict]
        Results for each embedding
    embedding_names : List[str]
        Names of embeddings
    output_dir : Path
        Output directory
    n_classes : int
        Number of classes
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(embedding_names)))

    # Plot 1: ROC Curves (micro-average)
    ax_roc = axes[0]

    for idx, (emb_name, color) in enumerate(zip(embedding_names, colors)):
        results = all_results[emb_name]
        y_true = np.array(results["y_true"])
        y_prob = np.vstack(results["y_prob"])

        # Binarize labels for multiclass ROC
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        # Compute micro-average ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        roc_auc = auc(fpr, tpr)

        label = (
            emb_name.replace("X_bmfm_", "")
            .replace("X_raw_", "RAW ")
            .replace("_", " ")
            .upper()
        )
        ax_roc.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC = {roc_auc:.3f})")

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    ax_roc.set_ylabel("True Positive Rate", fontsize=11, fontweight="bold")
    ax_roc.set_title("ROC Curve (Micro-Average)", fontsize=12, fontweight="bold")
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_roc.grid(alpha=0.3, linestyle="--")
    ax_roc.spines["top"].set_visible(False)
    ax_roc.spines["right"].set_visible(False)

    # Plot 2: Precision-Recall Curves (micro-average)
    ax_pr = axes[1]

    for idx, (emb_name, color) in enumerate(zip(embedding_names, colors)):
        results = all_results[emb_name]
        y_true = np.array(results["y_true"])
        y_prob = np.vstack(results["y_prob"])

        # Binarize labels for multiclass PR
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        # Compute micro-average precision-recall curve
        precision, recall, _ = precision_recall_curve(
            y_true_bin.ravel(), y_prob.ravel()
        )
        avg_precision = average_precision_score(y_true_bin, y_prob, average="micro")

        label = (
            emb_name.replace("X_bmfm_", "")
            .replace("X_raw_", "RAW ")
            .replace("_", " ")
            .upper()
        )
        ax_pr.plot(
            recall,
            precision,
            color=color,
            lw=2,
            label=f"{label} (AP = {avg_precision:.3f})",
        )

    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_xlabel("Recall", fontsize=11, fontweight="bold")
    ax_pr.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax_pr.set_title(
        "Precision-Recall Curve (Micro-Average)", fontsize=12, fontweight="bold"
    )
    ax_pr.legend(loc="lower left", fontsize=9)
    ax_pr.grid(alpha=0.3, linestyle="--")
    ax_pr.spines["top"].set_visible(False)
    ax_pr.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "roc_pr_curves.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "roc_pr_curves.png", dpi=300, bbox_inches="tight")
    print(f"Saved ROC and PR curves to {output_dir / 'roc_pr_curves.pdf'}")
    plt.close()


def plot_umap_embeddings(
    adata: sc.AnnData,
    embedding_keys: list[str],
    output_dir: Path,
    random_seed: int = 42,
):
    """
    Create UMAP visualizations of BMFM embeddings colored by MoA class using scanpy.

    Parameters
    ----------
    adata : AnnData
        AnnData object with embeddings in obsm
    embedding_keys : List[str]
        Names of BMFM embeddings (excluding raw expression)
    output_dir : Path
        Output directory
    random_seed : int
        Random seed for UMAP
    """
    n_embeddings = len(embedding_keys)
    n_cols = 5
    n_rows = (n_embeddings + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    # Flatten axes array for easier indexing
    if n_embeddings == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

    for idx, emb_key in enumerate(embedding_keys):
        print(f"  Computing UMAP for {emb_key}...")

        # Create a copy of adata for this embedding
        adata_temp = adata.copy()

        # Set the embedding as X for UMAP computation
        adata_temp.obsm["X_embedding"] = adata_temp.obsm[emb_key]

        # Compute neighbors and UMAP
        sc.pp.neighbors(adata_temp, use_rep="X_embedding", random_state=random_seed)
        sc.tl.umap(adata_temp, random_state=random_seed)

        # Plot on the corresponding axis
        ax = axes[idx]

        # Get unique MoA classes and create color palette
        moa_classes = adata_temp.obs["target"].unique()
        n_classes = len(moa_classes)
        palette = dict(zip(moa_classes, plt.cm.tab10(np.linspace(0, 1, n_classes))))

        # Plot each MoA class
        for moa in moa_classes:
            mask = adata_temp.obs["target"] == moa
            umap_coords = adata_temp.obsm["X_umap"][mask]
            ax.scatter(
                umap_coords[:, 0],
                umap_coords[:, 1],
                c=[palette[moa]],
                label=moa,
                alpha=0.6,
                s=10,
                edgecolors="none",
            )

        # Format plot
        title = emb_key.replace("X_bmfm_", "").replace("_", " ").upper()
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=10)
        ax.set_ylabel("UMAP 2", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.2, linestyle="--")

        # Add legend only to the last plot
        if idx == n_embeddings - 1:
            ax.legend(
                bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, markerscale=2
            )

    plt.tight_layout()
    plt.savefig(output_dir / "umap_embeddings.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "umap_embeddings.png", dpi=300, bbox_inches="tight")
    print(f"Saved UMAP embeddings to {output_dir / 'umap_embeddings.pdf'}")
    plt.close()


def plot_publication_panel(
    adata: sc.AnnData,
    all_results: dict[str, dict],
    embedding_keys: list[str],
    label_encoder: LabelEncoder,
    n_classes: int,
    output_dir: Path,
    random_seed: int = 42,
    baseline_key: str | None = None,
    bio_scores: dict[str, dict[str, float]] | None = None,
):
    """
    Create a comprehensive publication-quality panel figure.

    Includes: Combined ROC curve, Recall & MCC bar plots, Avg Bio, UMAPs, and Confusion matrices
    for the BMFM embeddings and optionally a baseline.

    Parameters
    ----------
    all_results : Dict[str, Dict]
        Results for each embedding
    embedding_keys : List[str]
        Names of BMFM embeddings (excluding raw expression)
    n_classes : int
        Number of MoA classes
    output_dir : Path
        Output directory
    baseline_key : str, optional
        Key for baseline results to include in plots
    bio_scores : Dict[str, Dict[str, float]], optional
        Bio conservation scores for embeddings
    """
    # Determine which embeddings to plot
    plot_keys = list(embedding_keys)
    if baseline_key and baseline_key in all_results:
        plot_keys.append(baseline_key)

    n_embeddings = len(plot_keys)

    # Calculate AUC for each embedding and sort by AUC (best to worst)
    auc_scores = {}
    for emb_key in plot_keys:
        results = all_results[emb_key]
        y_true = np.array(results["y_true"])
        y_prob = np.vstack(results["y_prob"])
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        auc_scores[emb_key] = auc(fpr, tpr)

    # Sort embedding keys by AUC (descending)
    sorted_embedding_keys = sorted(plot_keys, key=lambda k: auc_scores[k], reverse=True)

    # Create figure with mixed grid layout
    fig = plt.figure(figsize=(24, 20))

    colors = plt.cm.tab10(np.linspace(0, 1, n_embeddings))

    # Create separate gridspecs for each row with adjusted spacing
    # Top row now has 4 columns (ROC, Recall, MCC, Avg Bio)
    gs_top = fig.add_gridspec(
        1, 4, left=0.05, right=0.95, top=0.95, bottom=0.70, hspace=0.4, wspace=0.3
    )
    gs_mid = fig.add_gridspec(
        1,
        n_embeddings,
        left=0.05,
        right=0.95,
        top=0.65,
        bottom=0.38,
        hspace=0.4,
        wspace=0.3,
    )
    gs_bot = fig.add_gridspec(
        1,
        n_embeddings,
        left=0.05,
        right=0.95,
        top=0.33,
        bottom=0.06,
        hspace=0.4,
        wspace=0.3,
    )

    # Row 1, Column 1: Combined ROC Curve
    ax_roc = fig.add_subplot(gs_top[0, 0])

    for idx, emb_key in enumerate(sorted_embedding_keys):
        results = all_results[emb_key]
        y_true = np.array(results["y_true"])
        y_prob = np.vstack(results["y_prob"])

        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        roc_auc = auc_scores[emb_key]

        title = emb_key.replace("X_bmfm_", "").replace("_", " ").upper()
        ax_roc.plot(
            fpr, tpr, color=colors[idx], lw=2, label=f"{title} (AUC={roc_auc:.3f})"
        )

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
    ax_roc.set_title("ROC Curves", fontsize=12, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate", fontsize=10)
    ax_roc.set_ylabel("True Positive Rate", fontsize=10)
    ax_roc.legend(loc="lower right", fontsize=8)
    ax_roc.grid(alpha=0.3, linestyle="--")
    ax_roc.set_xlim([0, 1])
    ax_roc.set_ylim([0, 1.05])
    ax_roc.spines["top"].set_visible(False)
    ax_roc.spines["right"].set_visible(False)

    # Row 1, Column 2: Recall Bar Plot
    ax_recall = fig.add_subplot(gs_top[0, 1])

    # Prepare recall data (using sorted order)
    recall_data = []
    for emb_key in sorted_embedding_keys:
        per_repeat = calculate_per_repeat_metrics(all_results[emb_key])
        recall_mean = per_repeat["weighted_recall"].mean()
        recall_std = per_repeat["weighted_recall"].std()
        recall_data.append(
            {
                "Embedding": emb_key.replace("X_bmfm_", "").replace("_", " ").upper(),
                "Mean": recall_mean,
                "Std": recall_std,
            }
        )

    df_recall = pd.DataFrame(recall_data)
    x_pos = np.arange(len(df_recall))

    ax_recall.bar(
        x_pos,
        df_recall["Mean"],
        yerr=df_recall["Std"],
        color=colors,
        capsize=5,
        width=0.5,
        error_kw={"linewidth": 1.5},
    )
    ax_recall.set_xticks(x_pos)
    ax_recall.set_xticklabels(
        df_recall["Embedding"], rotation=45, ha="right", fontsize=9
    )
    ax_recall.set_title("Weighted Recall", fontsize=12, fontweight="bold")
    ax_recall.set_ylabel("Recall", fontsize=10)
    ax_recall.set_ylim([0, 1])
    ax_recall.grid(axis="y", alpha=0.3, linestyle="--")
    ax_recall.spines["top"].set_visible(False)
    ax_recall.spines["right"].set_visible(False)

    # Row 1, Column 3: MCC Bar Plot
    ax_mcc = fig.add_subplot(gs_top[0, 2])

    # Prepare MCC data (using sorted order)
    mcc_data = []
    for emb_key in sorted_embedding_keys:
        per_repeat = calculate_per_repeat_metrics(all_results[emb_key])
        mcc_mean = per_repeat["mcc"].mean()
        mcc_std = per_repeat["mcc"].std()
        mcc_data.append(
            {
                "Embedding": emb_key.replace("X_bmfm_", "").replace("_", " ").upper(),
                "Mean": mcc_mean,
                "Std": mcc_std,
            }
        )

    df_mcc = pd.DataFrame(mcc_data)

    ax_mcc.bar(
        x_pos,
        df_mcc["Mean"],
        yerr=df_mcc["Std"],
        color=colors,
        capsize=5,
        width=0.5,
        error_kw={"linewidth": 1.5},
    )
    ax_mcc.set_xticks(x_pos)
    ax_mcc.set_xticklabels(df_mcc["Embedding"], rotation=45, ha="right", fontsize=9)
    ax_mcc.set_title("Matthews Correlation Coefficient", fontsize=12, fontweight="bold")
    ax_mcc.set_ylabel("MCC", fontsize=10)
    ax_mcc.set_ylim([0, 1])
    ax_mcc.grid(axis="y", alpha=0.3, linestyle="--")
    ax_mcc.spines["top"].set_visible(False)
    ax_mcc.spines["right"].set_visible(False)

    # Row 1, Column 4: Average Bio Score Bar Plot
    if bio_scores:
        ax_bio = fig.add_subplot(gs_top[0, 3])

        # Prepare bio data (using sorted order, excluding baseline)
        bio_data = []
        for emb_key in sorted_embedding_keys:
            if emb_key in bio_scores:
                avg_bio = bio_scores[emb_key]["avg_bio"]
                bio_data.append(
                    {
                        "Embedding": emb_key.replace("X_bmfm_", "")
                        .replace("_", " ")
                        .upper(),
                        "Score": avg_bio,
                    }
                )

        if bio_data:
            df_bio = pd.DataFrame(bio_data)
            x_pos_bio = np.arange(len(df_bio))

            # Use same colors as other plots
            bio_colors = [colors[i] for i in range(len(df_bio))]

            ax_bio.bar(x_pos_bio, df_bio["Score"], color=bio_colors, width=0.5)
            ax_bio.set_xticks(x_pos_bio)
            ax_bio.set_xticklabels(
                df_bio["Embedding"], rotation=45, ha="right", fontsize=9
            )
            ax_bio.set_title("Average Bio Score", fontsize=12, fontweight="bold")
            ax_bio.set_ylabel("Avg Bio", fontsize=10)
            # Adjust y-axis to better show differences in bio scores
            if bio_data:
                min_score = min([d["Score"] for d in bio_data])
                max_score = max([d["Score"] for d in bio_data])
                y_range = max_score - min_score
                y_min = max(0, min_score - 0.1 * y_range)
                y_max = min(1, max_score + 0.1 * y_range)
                ax_bio.set_ylim([y_min, y_max])
            else:
                ax_bio.set_ylim([0, 1])
            ax_bio.grid(axis="y", alpha=0.3, linestyle="--")
            ax_bio.spines["top"].set_visible(False)
            ax_bio.spines["right"].set_visible(False)

    # Row 2: UMAP plots (using sorted order)
    for idx, emb_key in enumerate(sorted_embedding_keys):
        ax = fig.add_subplot(gs_mid[0, idx])

        # Create temp adata for UMAP
        adata_temp = adata.copy()
        adata_temp.obsm["X_embedding"] = adata_temp.obsm[emb_key]

        # Compute UMAP
        sc.pp.neighbors(adata_temp, use_rep="X_embedding", random_state=random_seed)
        sc.tl.umap(adata_temp, random_state=random_seed)

        # Get MoA classes and colors
        moa_classes = adata_temp.obs["target"].unique()
        n_moa = len(moa_classes)
        palette = dict(zip(moa_classes, plt.cm.tab10(np.linspace(0, 1, n_moa))))

        # Plot
        for moa in moa_classes:
            mask = adata_temp.obs["target"] == moa
            umap_coords = adata_temp.obsm["X_umap"][mask]
            ax.scatter(
                umap_coords[:, 0],
                umap_coords[:, 1],
                c=[palette[moa]],
                label=moa
                if idx == n_embeddings - 1
                else "",  # Legend only on last plot
                alpha=0.6,
                s=8,
                edgecolors="none",
            )

        title = emb_key.replace("X_bmfm_", "").replace("_", " ").upper()
        ax.set_title(f"{title}\nUMAP", fontsize=11, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=9)
        ax.set_ylabel("UMAP 2", fontsize=9)
        ax.set_aspect("equal", "box")  # Make UMAP square
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if idx == n_embeddings - 1:  # Add legend to last plot
            ax.legend(
                bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7, markerscale=1.5
            )

    # Row 3: Confusion Matrices (using sorted order)
    for idx, emb_key in enumerate(sorted_embedding_keys):
        ax = fig.add_subplot(gs_bot[0, idx])

        results = all_results[emb_key]
        y_true = np.array(results["y_true"])
        y_pred = np.array(results["y_pred"])

        # Get unique classes
        classes = np.unique(y_true)
        cm = confusion_matrix(y_true, y_pred, labels=classes)

        # Normalize
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Get class names
        class_names = label_encoder.inverse_transform(classes)

        # Plot heatmap
        sns.heatmap(
            cm_norm,
            annot=False,
            cmap="Blues",
            ax=ax,
            cbar_kws={"label": "Proportion"},
            square=True,
            xticklabels=class_names,
            yticklabels=class_names,
            vmin=0,
            vmax=1,
        )

        title = emb_key.replace("X_bmfm_", "").replace("_", " ").upper()
        ax.set_title(f"{title}\nConfusion Matrix", fontsize=11, fontweight="bold")
        ax.set_ylabel("True MoA", fontsize=9)
        ax.set_xlabel("Predicted MoA", fontsize=9)
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.tick_params(axis="y", rotation=0, labelsize=7)

    plt.savefig(output_dir / "publication_panel.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "publication_panel.png", dpi=300, bbox_inches="tight")
    print(f"Saved publication panel to {output_dir / 'publication_panel.pdf'}")
    plt.close()


def plot_results(
    all_results: dict[str, dict],
    embedding_names: list[str],
    output_dir: Path,
    label_encoder: LabelEncoder = None,
):
    """
    Create Nature Methods-style plots for results.

    Parameters
    ----------
    all_results : Dict[str, Dict]
        Results for each embedding
    embedding_names : List[str]
        Names of embeddings
    output_dir : Path
        Output directory
    label_encoder : LabelEncoder, optional
        Label encoder to convert numeric labels back to MoA names
    """
    # Create figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    # Prepare data for plotting
    metrics_data = []
    for emb_name in embedding_names:
        per_repeat = calculate_per_repeat_metrics(all_results[emb_name])
        for _, row in per_repeat.iterrows():
            metrics_data.append(
                {
                    "Embedding": emb_name.replace("X_bmfm_", "")
                    .replace("X_raw_", "RAW ")
                    .replace("_", " ")
                    .upper(),
                    "Weighted F1": row["weighted_f1"],
                    "Weighted Recall": row["weighted_recall"],
                    "ROC AUC": row["roc_auc_ovr"],
                    "MCC": row["mcc"],
                }
            )

    df_plot = pd.DataFrame(metrics_data)

    # Plot 1: Weighted F1 Score
    ax1 = axes[0]
    sns.barplot(
        data=df_plot,
        x="Embedding",
        y="Weighted F1",
        ax=ax1,
        capsize=0.1,
        errwidth=1.5,
        errorbar="sd",
    )
    ax1.set_ylabel("Weighted F1-Score", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Embedding", fontsize=11, fontweight="bold")
    ax1.set_ylim([0, 1])
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Plot 2: Weighted Recall Score
    ax2 = axes[1]
    sns.barplot(
        data=df_plot,
        x="Embedding",
        y="Weighted Recall",
        ax=ax2,
        capsize=0.1,
        errwidth=1.5,
        errorbar="sd",
    )
    ax2.set_ylabel("Weighted Recall", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Embedding", fontsize=11, fontweight="bold")
    ax2.set_ylim([0, 1])
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Plot 3: ROC AUC
    ax3 = axes[2]
    sns.barplot(
        data=df_plot,
        x="Embedding",
        y="ROC AUC",
        ax=ax3,
        capsize=0.1,
        errwidth=1.5,
        errorbar="sd",
    )
    ax3.set_ylabel("ROC AUC (OvR)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Embedding", fontsize=11, fontweight="bold")
    ax3.set_ylim([0, 1])
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(axis="y", alpha=0.3, linestyle="--")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # Plot 4: MCC
    ax4 = axes[3]
    sns.barplot(
        data=df_plot,
        x="Embedding",
        y="MCC",
        ax=ax4,
        capsize=0.1,
        errwidth=1.5,
        errorbar="sd",
    )
    ax4.set_ylabel("Matthews Correlation Coefficient", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Embedding", fontsize=11, fontweight="bold")
    ax4.set_ylim([-1, 1])
    ax4.tick_params(axis="x", rotation=45)
    ax4.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax4.grid(axis="y", alpha=0.3, linestyle="--")
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "moa_comparison_metrics.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "moa_comparison_metrics.png", dpi=300, bbox_inches="tight")
    print(f"Saved metrics plot to {output_dir / 'moa_comparison_metrics.pdf'}")
    plt.close()

    # Create confusion matrices
    n_embeddings = len(embedding_names)
    fig, axes = plt.subplots(1, n_embeddings, figsize=(5 * n_embeddings, 4))
    if n_embeddings == 1:
        axes = [axes]

    for idx, emb_name in enumerate(embedding_names):
        results = all_results[emb_name]
        y_true = np.array(results["y_true"])
        y_pred = np.array(results["y_pred"])

        # Get unique classes
        classes = np.unique(y_true)
        cm = confusion_matrix(y_true, y_pred, labels=classes)

        # Normalize
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Get class names if label encoder provided
        if label_encoder is not None:
            class_names = label_encoder.inverse_transform(classes)
        else:
            class_names = classes

        # Plot
        sns.heatmap(
            cm_norm,
            annot=False,
            fmt=".2f",
            cmap="Blues",
            ax=axes[idx],
            cbar_kws={"label": "Proportion"},
            square=True,
            xticklabels=class_names,
            yticklabels=class_names,
        )
        axes[idx].set_title(
            emb_name.replace("X_bmfm_", "").replace("_", " ").upper(),
            fontsize=12,
            fontweight="bold",
        )
        axes[idx].set_ylabel("True MoA", fontsize=10)
        axes[idx].set_xlabel("Predicted MoA", fontsize=10)
        axes[idx].tick_params(axis="x", rotation=90, labelsize=8)
        axes[idx].tick_params(axis="y", rotation=0, labelsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrices.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "confusion_matrices.png", dpi=300, bbox_inches="tight")
    print(f"Saved confusion matrices to {output_dir / 'confusion_matrices.pdf'}")
    plt.close()


def save_results_to_csv(
    all_results: dict[str, dict],
    all_metrics: dict[str, dict],
    embedding_names: list[str],
    output_dir: Path,
    label_encoder: LabelEncoder = None,
):
    """
    Save detailed results to CSV files.

    Parameters
    ----------
    all_results : Dict[str, Dict]
        Results for each embedding
    all_metrics : Dict[str, Dict]
        Metrics for each embedding
    embedding_names : List[str]
        Names of embeddings
    output_dir : Path
        Output directory
    label_encoder : LabelEncoder, optional
        Label encoder to convert numeric labels back to MoA names
    """
    # Save overall metrics
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.index.name = "embedding"
    metrics_df.to_csv(output_dir / "overall_metrics.csv")
    print(f"Saved overall metrics to {output_dir / 'overall_metrics.csv'}")

    # Save per-repeat metrics
    per_repeat_list = []
    for emb_name in embedding_names:
        per_repeat = calculate_per_repeat_metrics(all_results[emb_name])
        per_repeat["embedding"] = emb_name
        per_repeat_list.append(per_repeat)

    per_repeat_df = pd.concat(per_repeat_list, ignore_index=True)
    per_repeat_df.to_csv(output_dir / "per_repeat_metrics.csv", index=False)
    print(f"Saved per-repeat metrics to {output_dir / 'per_repeat_metrics.csv'}")

    # Save detailed predictions
    for emb_name in embedding_names:
        results = all_results[emb_name]

        # Convert numeric labels to MoA names if encoder provided
        if label_encoder is not None:
            y_true_names = label_encoder.inverse_transform(results["y_true"])
            y_pred_names = label_encoder.inverse_transform(results["y_pred"])
        else:
            y_true_names = results["y_true"]
            y_pred_names = results["y_pred"]

        pred_df = pd.DataFrame(
            {
                "y_true": y_true_names,
                "y_pred": y_pred_names,
                "test_product": results["test_products"],
                "fold": results["fold"],
                "repeat": results["repeat"],
            }
        )

        filename = (
            f"predictions_{emb_name.replace('X_bmfm_', '').replace('X_raw_', '')}.csv"
        )
        pred_df.to_csv(output_dir / filename, index=False)
        print(f"Saved predictions to {output_dir / filename}")


def plot_avg_bio_scores(
    adata: sc.AnnData,
    embedding_keys: list[str],
    output_dir: Path,
) -> dict[str, dict[str, float]]:
    """
    Calculate and plot average bio scores for each embedding using scib.metrics.

    Uses scib.metrics.metrics() to calculate biological conservation scores:
    - NMI (cluster/label)
    - ARI (cluster/label)
    - ASW (label) - Average Silhouette Width
    - avg_bio - average of the above three metrics

    Parameters
    ----------
    adata : AnnData
        AnnData object with embeddings in obsm
    embedding_keys : List[str]
        List of embedding keys to evaluate
    output_dir : Path
        Directory to save plots
    """
    print(f"\n{'=' * 60}")
    print("Calculating average bio scores using scib.metrics...")
    print(f"{'=' * 60}")

    # Calculate bio scores for each embedding
    bio_scores = {}

    for emb_key in embedding_keys:
        print(f"Processing {emb_key}...")

        if emb_key not in adata.obsm:
            print(f"  Warning: {emb_key} not found in adata.obsm")
            continue

        m = scib_bio_metrics(adata.obs, np.asarray(adata.obsm[emb_key]))
        nmi_score, ari_score, asw_score, avg_bio = (
            m["NMI"],
            m["ARI"],
            m["ASW"],
            m["avg_bio"],
        )

        bio_scores[emb_key] = {
            "NMI_cluster/label": nmi_score,
            "ARI_cluster/label": ari_score,
            "ASW_label": asw_score,
            "avg_bio": avg_bio,
        }

        # Format output strings properly
        nmi_str = f"{nmi_score:.4f}" if not np.isnan(nmi_score) else "N/A"
        ari_str = f"{ari_score:.4f}" if not np.isnan(ari_score) else "N/A"
        asw_str = f"{asw_score:.4f}" if not np.isnan(asw_score) else "N/A"
        print(f"  NMI (cluster/label): {nmi_str}")
        print(f"  ARI (cluster/label): {ari_str}")
        print(f"  ASW (label): {asw_str}")
        print(f"  Avg Bio: {avg_bio:.4f}")

    if not bio_scores:
        print("No bio scores calculated. Skipping plot generation.")
        return {}

    # Create bar plot for avg_bio matching publication theme
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data for plotting
    emb_names = list(bio_scores.keys())
    avg_bio_values = [bio_scores[k]["avg_bio"] for k in emb_names]

    # Create bar plot with colors matching other plots
    x_pos = np.arange(len(emb_names))
    colors = plt.cm.tab10(np.linspace(0, 1, len(emb_names)))
    bars = ax.bar(
        x_pos, avg_bio_values, color=colors, width=0.6, edgecolor="black", linewidth=1.2
    )

    # Customize plot to match publication theme
    ax.set_xlabel("Embedding Type", fontsize=12, fontweight="bold")
    ax.set_ylabel("Average Bio Score", fontsize=12, fontweight="bold")
    ax.set_title("Average Bio Conservation Scores", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [name.replace("X_bmfm_", "").replace("_", " ").upper() for name in emb_names],
        rotation=45,
        ha="right",
        fontsize=10,
    )
    ax.set_ylim([0, 1.0])
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, avg_bio_values)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()

    # Save plot
    output_path_png = output_dir / "avg_bio_scores.png"
    output_path_pdf = output_dir / "avg_bio_scores.pdf"
    plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_path_pdf, bbox_inches="tight")
    plt.close()

    print("\nAverage bio scores plot saved to:")
    print(f"  {output_path_png}")
    print(f"  {output_path_pdf}")

    # Also create a detailed comparison plot with all metrics matching theme
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    metrics = ["NMI_cluster/label", "ARI_cluster/label", "ASW_label", "avg_bio"]
    titles = [
        "NMI (cluster/label)",
        "ARI (cluster/label)",
        "ASW (label)",
        "Average Bio Score",
    ]

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        values = [bio_scores[k][metric] for k in emb_names]

        bars = ax.bar(
            x_pos, values, color=colors, width=0.6, edgecolor="black", linewidth=1.2
        )
        ax.set_xlabel("Embedding", fontsize=10, fontweight="bold")
        ax.set_ylabel("Score", fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [
                name.replace("X_bmfm_", "").replace("_", " ").upper()
                for name in emb_names
            ],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        ax.set_ylim([0, 1.0])
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.02,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    plt.tight_layout()

    # Save detailed plot
    output_path_detailed_png = output_dir / "bio_scores_detailed.png"
    output_path_detailed_pdf = output_dir / "bio_scores_detailed.pdf"
    plt.savefig(output_path_detailed_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_path_detailed_pdf, bbox_inches="tight")
    plt.close()

    print("\nDetailed bio scores plot saved to:")
    print(f"  {output_path_detailed_png}")
    print(f"  {output_path_detailed_pdf}")

    # Return bio_scores for use in publication panel
    return bio_scores


def main():
    """Main execution function."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get embedding keys from command line arguments
    embedding_keys = args.embedding_keys
    print(f"\nEmbedding keys to compare: {embedding_keys}")

    # Load data with filtering parameters
    adata, embeddings = load_and_prepare_data(
        args.input,
        embedding_keys,
        cell_type=args.cell_type,
        skip_vehicle_filter=args.skip_vehicle_filter,
        dose=args.dose,
        viability_threshold=args.viability_threshold,
        skip_viability_filter=args.skip_viability_filter,
        skip_others_filter=args.skip_others_filter,
        min_drugs_per_moa=args.min_drugs_per_moa,
    )

    # Plot data distribution
    print(f"\n{'=' * 60}")
    print("Generating data distribution plots...")
    print(f"{'=' * 60}")
    plot_data_distribution(adata, output_dir)

    # Prepare labels
    le = LabelEncoder()
    y = le.fit_transform(adata.obs["target"].values)
    products = adata.obs["product_name"].values

    print(
        f"\nStarting {args.n_repeats}-repeat stratified drug group cross-validation..."
    )
    print(f"Total unique products: {len(np.unique(products))}")
    print("Each fold will contain at least one drug from each MoA class")

    # Run CV for each embedding
    all_results = {}
    all_metrics = {}

    # Optionally add raw gene expression as baseline
    if args.include_baseline:
        print(f"\n{'=' * 60}")
        print("Processing baseline: Raw Gene Expression")
        print(f"{'=' * 60}")
        print(f"Using adata.X with shape: {adata.X.shape}")

        # Convert to dense if sparse
        X_baseline = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

        results_baseline = stratified_drug_group_cv(
            X_baseline,
            y,
            products,
            n_folds=5,
            n_repeats=args.n_repeats,
            random_seed=args.random_seed,
            max_iter=args.max_iter,
        )

        metrics_baseline = calculate_metrics(results_baseline)
        all_results["X_raw_expression"] = results_baseline
        all_metrics["X_raw_expression"] = metrics_baseline

        print("\nResults for Raw Gene Expression:")
        print(f"  Weighted F1-Score: {metrics_baseline['weighted_f1']:.4f}")
        print(f"  Macro F1-Score: {metrics_baseline['macro_f1']:.4f}")
        print(f"  ROC AUC (OvR): {metrics_baseline['roc_auc_ovr']:.4f}")
        print(f"  MCC: {metrics_baseline['mcc']:.4f}")

    # Run CV for embeddings
    for emb_name, X in embeddings.items():
        print(f"\n{'=' * 60}")
        print(f"Processing embedding: {emb_name}")
        print(f"{'=' * 60}")

        # Run cross-validation
        results = stratified_drug_group_cv(
            X,
            y,
            products,
            n_folds=5,
            n_repeats=args.n_repeats,
            random_seed=args.random_seed,
            max_iter=args.max_iter,
        )

        # Calculate metrics
        metrics = calculate_metrics(results)

        all_results[emb_name] = results
        all_metrics[emb_name] = metrics

        print(f"\nResults for {emb_name}:")
        print(f"  Weighted F1-Score: {metrics['weighted_f1']:.4f}")
        print(f"  Weighted Recall: {metrics['weighted_recall']:.4f}")
        print(f"  Macro F1-Score: {metrics['macro_f1']:.4f}")
        print(f"  Macro Recall: {metrics['macro_recall']:.4f}")
        print(f"  ROC AUC (OvR): {metrics['roc_auc_ovr']:.4f}")
        print(f"  MCC: {metrics['mcc']:.4f}")

    # Plot results (include baseline if requested)
    print(f"\n{'=' * 60}")
    print("Generating plots...")
    print(f"{'=' * 60}")
    all_embedding_keys = (
        ["X_raw_expression"] if args.include_baseline else []
    ) + embedding_keys

    # Get number of classes
    n_classes = len(np.unique(y))

    # Plot UMAP embeddings (only BMFM, not raw expression)
    print("\nGenerating UMAP visualizations...")
    plot_umap_embeddings(adata, embedding_keys, output_dir, args.random_seed)

    # Plot average bio scores and get the scores for publication panel
    bio_scores = plot_avg_bio_scores(adata, embedding_keys, output_dir)

    # Plot ROC and PR curves
    plot_roc_pr_curves(all_results, all_embedding_keys, output_dir, n_classes)

    # Plot other results
    plot_results(all_results, all_embedding_keys, output_dir, label_encoder=le)

    # Create publication panel (BMFM embeddings with optional baseline)
    # Use the embedding keys provided (already in desired order)
    print("\nGenerating publication panel figure...")
    baseline_key = (
        "X_raw_expression"
        if (args.include_baseline and "X_raw_expression" in all_results)
        else None
    )
    plot_publication_panel(
        adata,
        all_results,
        embedding_keys,
        le,
        n_classes,
        output_dir,
        args.random_seed,
        baseline_key=baseline_key,
        bio_scores=bio_scores,
    )

    # Save results
    print(f"\n{'=' * 60}")
    print("Saving results to CSV...")
    print(f"{'=' * 60}")
    save_results_to_csv(
        all_results, all_metrics, all_embedding_keys, output_dir, label_encoder=le
    )

    print(f"\n{'=' * 60}")
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
