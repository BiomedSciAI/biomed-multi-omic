"""Plots of the per-gene skills computed in `bmfm_targets.evaluation.gene_learnability`."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from bmfm_targets.evaluation.gene_learnability import quantification_skill

WIN_COLOR, LOSS_COLOR = "#2ca25f", "#de2d26"
DETECTION_COLOR, QUANTIFICATION_COLOR = "#3b6fb0", "#3a9c54"


def plot_error_vs_expression(skills: pd.DataFrame, ax=None):
    """Quantification error against expression, with the band `q_skill` is scaled by."""
    ax = ax or plt.gca()
    band = quantification_skill(skills["log_gt"], skills["log_mae"])
    worst = skills["q_skill"] <= skills["q_skill"].quantile(0.10)
    color = np.select([skills["q_win"], worst], [WIN_COLOR, LOSS_COLOR], "#c8ccd0")
    ax.scatter(skills["log_gt"], np.exp(skills["log_mae"]), s=3, c=color)
    order = np.argsort(skills["log_gt"].to_numpy())
    x = skills["log_gt"].to_numpy()[order]
    for column, style in (("trend", "-"), ("lo", "--"), ("hi", "--")):
        ax.plot(x, np.exp(band[column].to_numpy()[order]), style, color="black")
    ax.set(yscale="log", xlabel="log mean expression", ylabel="MAE on expressed cells")
    return ax


def plot_detection_vs_quantification(skills: pd.DataFrame, ax=None):
    """Detection AUC against quantification skill; returns the axis and Spearman rho."""
    ax = ax or plt.gca()
    rho = spearmanr(skills["det_auc"], skills["q_skill"])[0]
    points = ax.scatter(
        skills["det_auc"], skills["q_skill"], c=skills["log_gt"], s=4, alpha=0.45
    )
    ax.figure.colorbar(points, ax=ax, label="log mean expression")
    ax.set_ylim(np.percentile(skills["q_skill"], [1, 99]))
    ax.set(xlabel="detection ROC-AUC", ylabel="quantification skill")
    ax.set_title(f"Spearman rho = {rho:.2f}")
    return ax, rho


def family_medians(skills: pd.DataFrame, families: pd.Series) -> pd.DataFrame:
    """Median `det_dev` and `q_skill` per family, sorted by `q_skill`, with gene counts."""
    grouped = skills.groupby(families.reindex(skills.index))
    table = grouped[["det_dev", "q_skill"]].median()
    table["n"] = grouped.size()
    return table.sort_values("q_skill")


def plot_family_medians(table: pd.DataFrame, ax=None):
    """Family medians from `family_medians` on a shared row per family."""
    ax = ax or plt.gca()
    twin = ax.twiny()
    rows = np.arange(len(table))
    ax.scatter(table["q_skill"], rows - 0.15, color=QUANTIFICATION_COLOR)
    twin.scatter(table["det_dev"], rows + 0.15, color=DETECTION_COLOR)
    for axis, column in ((ax, "q_skill"), (twin, "det_dev")):
        limit = table[column].abs().max() * 1.1
        axis.set_xlim(-limit, limit)
    ax.axvline(0, color="#444444", ls="--")
    ax.set_yticks(rows, [f"{f} (n={n})" for f, n in table["n"].items()])
    ax.set_xlabel("median quantification skill", color=QUANTIFICATION_COLOR)
    twin.set_xlabel("median detection AUC - overall median", color=DETECTION_COLOR)
    return ax


def family_trajectories(
    per_checkpoint: dict[float, pd.DataFrame], families: pd.Series, column: str
) -> pd.DataFrame:
    """
    Family medians of one per-gene column at each checkpoint.

    Args:
    ----
        per_checkpoint: training step or epoch -> per-gene table indexed by gene
        families: family of each gene, indexed by gene
        column: per-gene column to summarise

    Returns:
    -------
        Checkpoints x families table, ordered by step.
    """
    medians = {
        step: table[column].groupby(families.reindex(table.index)).median()
        for step, table in per_checkpoint.items()
    }
    return pd.DataFrame(medians).T.sort_index()


def rank_trajectories(
    trajectories: pd.DataFrame, higher_is_better: bool
) -> pd.DataFrame:
    """Rank of each family at each checkpoint (1 = best)."""
    return trajectories.rank(axis=1, ascending=not higher_is_better)


def plot_family_trajectories(trajectories: pd.DataFrame, highlight: list[str], ax=None):
    """Family trajectories over training, with some families highlighted."""
    ax = ax or plt.gca()
    rest = trajectories.columns.difference(highlight)
    ax.plot(trajectories.index, trajectories[rest], color="#dcdcdc", lw=0.7)
    for family in highlight:
        ax.plot(trajectories.index, trajectories[family], lw=2.2, label=family)
    ax.legend(frameon=False)
    return ax


def plot_model_agreement(first: pd.Series, second: pd.Series, ax=None):
    """One per-gene metric under two models; returns the axis and Spearman rho."""
    ax = ax or plt.gca()
    both = pd.concat([first, second], axis=1, join="inner").dropna()
    rho = spearmanr(both.iloc[:, 0], both.iloc[:, 1])[0]
    ax.scatter(both.iloc[:, 0], both.iloc[:, 1], s=4, alpha=0.25)
    low, high = np.percentile(both.to_numpy(), [0.5, 99.5])
    ax.plot([low, high], [low, high], "k--", lw=0.8)
    ax.set(xlabel=first.name, ylabel=second.name)
    ax.set_title(f"Spearman rho = {rho:.2f} (n = {len(both):,})")
    return ax, rho
