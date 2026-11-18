"""
Figure 6 of the BMFM-RNA paper (Sci-Plex 3 MoA) from the outputs of run_auc.py, run_avg_bio.py, run_umap.py.

The WCED MULTITASK embedding is the reconstruction [CLS] token (position 0, obsm key X_bmfm_wced_cls_0).
Inputs: <data>/obs.csv, <results>/auc_per_repeat.csv, roc_curves.npz, avg_bio.csv, umap_coords.npz
Outputs: <results>/Figure6.pdf / .png, <results>/fig6_summary.csv
"""
import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

matplotlib.use("Agg")

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--data", default="data")
ap.add_argument("--results", default="results")
args = ap.parse_args()
R = Path(args.results)
obs = pd.read_csv(Path(args.data) / "obs.csv", index_col=0)

# key -> (label, hidden size, colour)
MODELS = {
    "X_316M_geneformer": ("316M Geneformer", 1152, "#22a35a"),
    "X_transcriptformer": ("TranscriptFormer", 2048, "#f0a020"),
    "X_bmfm_wced_cls_0": ("WCED MULTITASK", 384, "#5aa9e0"),
    "X_38M_geneformer": ("38M Geneformer", 512, "#e07b20"),
    "X_scGPT": ("scGPT", 512, "#8e44ad"),
    "X_bmfm_mlm": ("MLM MULTITASK", 384, "#2e6fb0"),
}
KEYS = list(MODELS)
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

auc = pd.read_csv(R / "auc_per_repeat.csv")
pooled = auc[auc.repeat == "pooled"].set_index("embedding").micro_auc
rep = auc[auc.repeat != "pooled"].groupby("embedding").micro_auc.agg(["mean", "std"])
roc = np.load(R / "roc_curves.npz")
bio = pd.read_csv(R / "avg_bio.csv")
bio_full = bio[bio.replicate == "full"].set_index("embedding").avg_bio
boot = bio[bio.replicate != "full"].groupby("embedding").avg_bio.agg(["mean", "std"])
U = np.load(R / "umap_coords.npz")

fig = plt.figure(figsize=(20, 14.5))
gs = GridSpec(2, 1, height_ratios=[1.15, 1], hspace=0.28, figure=fig)
top = gs[0].subgridspec(
    2, 3, width_ratios=[1.1, 1.1, 1.0], height_ratios=[1, 1], wspace=0.28, hspace=0.45
)


def tag(ax, s):
    ax.text(
        -0.12,
        1.06,
        s,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="bottom",
    )


# A: ROC curves (micro-average, pooled over 5 repeats), legend gives pooled AUC and mean +/- sd over repeats
ax = fig.add_subplot(top[:, 0])
tag(ax, "A")
for k in sorted(KEYS, key=lambda k: -pooled[k]):
    lab, _, c = MODELS[k]
    ax.plot(
        roc[f"{k}__fpr"],
        roc[f"{k}__tpr"],
        color=c,
        lw=2,
        label=f"{lab} (AUC = {pooled[k]:.3f}; {rep.loc[k, 'mean']:.3f} ± {rep.loc[k, 'std']:.3f})",
    )
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
ax.set(
    xlim=(0, 1),
    ylim=(0, 1.02),
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
)
ax.set_title("ROC Curve (Micro-Average)", fontweight="bold")
ax.legend(loc="lower right", fontsize=8.5, frameon=True)
ax.grid(alpha=0.3, ls="--")

# B: hidden size vs AUC with Pareto front; error bars = sd over 5 repeats
ax = fig.add_subplot(top[0, 1])
tag(ax, "B")
pts = sorted((MODELS[k][1], pooled[k], k) for k in KEYS)
front, best = [], -1
for h, a, k in sorted(pts, key=lambda t: (t[0], -t[1])):
    if a > best:
        front.append((h, a))
        best = a
fx, fy = zip(*front)
ax.step(
    list(fx) + [2300],
    list(fy) + [fy[-1]],
    where="post",
    color="#f08070",
    lw=3,
    alpha=0.8,
    label="Pareto front",
    zorder=1,
)
for h, a, k in pts:
    lab, _, c = MODELS[k]
    ax.errorbar(
        h,
        a,
        yerr=rep.loc[k, "std"],
        fmt="o",
        ms=9,
        color=c,
        mec="k",
        ecolor=c,
        capsize=3,
        zorder=3,
    )
    off = {
        "X_bmfm_wced_cls_0": (0, 0, "center"),
        "X_38M_geneformer": (8, -14, "left"),
        "X_bmfm_mlm": (0, 0, "center"),
        "X_scGPT": (8, 6, "left"),
        "X_316M_geneformer": (8, 4, "left"),
        "X_transcriptformer": (0, -22, "center"),
    }[k]
    if k in (
        "X_bmfm_wced_cls_0",
        "X_bmfm_mlm",
    ):  # stacked at hidden size 384: label in data coords above/below the error bar
        y = (
            a + rep.loc[k, "std"] + 0.0035
            if k == "X_bmfm_wced_cls_0"
            else a - rep.loc[k, "std"] - 0.012
        )
        ax.text(
            h + 45,
            y,
            lab,
            ha="center",
            va="bottom" if k == "X_bmfm_wced_cls_0" else "top",
            fontsize=8,
            fontweight="bold",
        )
    else:
        ax.annotate(
            lab,
            (h, a),
            xytext=off[:2],
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            ha=off[2],
        )
ax.set(
    xlabel="Hidden Dimension Size",
    ylabel="AUC (Micro-Average)",
    xlim=(200, 2300),
    ylim=(0.78, 0.895),
)
ax.set_title("Hidden Dimension Size vs AUC with Pareto Front", fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3, ls="--")

# C: cells per MoA class
ax = fig.add_subplot(top[1, 1])
tag(ax, "C")
vc = obs.target.value_counts()
ax.bar(range(len(vc)), vc.values, color="#5b9bd5", edgecolor="k", width=0.6)
ax.set_xticks(range(len(vc)))
ax.set_xticklabels(vc.index, rotation=25, ha="right", fontsize=8)
ax.set(ylabel="Number of Cells", xlabel="MoA Class")
ax.set_title(f"Cells per MoA Class (n={len(obs):,} cells)", fontweight="bold")
ax.text(
    0.97,
    0.95,
    f"Mean: {vc.mean():.0f}\nMedian: {vc.median():.0f}\nMin: {vc.min()}\nMax: {vc.max()}",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=8,
    bbox={"boxstyle": "round", "fc": "#f5e9d0", "ec": "0.6"},
)
ax.grid(alpha=0.3, ls="--", axis="y")

# D: Avg Bio, bar = full data, error bar = sd over 10 paired 90% subsamples
ax = fig.add_subplot(top[:, 2])
tag(ax, "D")
order = sorted(KEYS, key=lambda k: -bio_full[k])
x = np.arange(len(order))
ax.bar(
    x,
    [bio_full[k] for k in order],
    yerr=[boot.loc[k, "std"] for k in order],
    capsize=4,
    color=[MODELS[k][2] for k in order],
    edgecolor="k",
    width=0.6,
)
for i, k in enumerate(order):
    ax.text(
        i,
        bio_full[k] + boot.loc[k, "std"] + 0.006,
        f"{bio_full[k]:.3f}",
        ha="center",
        fontsize=9,
        fontweight="bold",
    )
ax.set_xticks(x)
ax.set_xticklabels([MODELS[k][0] for k in order], rotation=40, ha="right")
ax.set(ylabel="Average Bio Score", ylim=(0, max(bio_full) * 1.25))
ax.set_title("Average Bio Conservation Scores", fontweight="bold")
ax.grid(alpha=0.3, ls="--", axis="y")

# E: UMAPs coloured by MoA
bot = gs[1].subgridspec(1, 6, wspace=0.25)
classes = obs.target.value_counts().index.tolist()
pal = dict(
    zip(
        classes,
        [
            "#9b59b6",
            "#2ecc71",
            "#2c5d8a",
            "#e74c3c",
            "#f1c40f",
            "#5dade2",
            "#e67e22",
            "#1abc9c",
            "#7f8c8d",
        ],
    )
)
for i, k in enumerate(
    [
        "X_bmfm_wced_cls_0",
        "X_38M_geneformer",
        "X_316M_geneformer",
        "X_transcriptformer",
        "X_scGPT",
        "X_bmfm_mlm",
    ]
):
    ax = fig.add_subplot(bot[0, i])
    if i == 0:
        tag(ax, "E")
    for m in classes[::-1]:
        s = (obs.target == m).values
        ax.scatter(
            *U[k][s].T,
            c=pal[m],
            s=4,
            alpha=0.6,
            edgecolors="none",
            label=m,
            rasterized=True,
        )
    ax.set_title(MODELS[k][0], fontweight="bold")
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2" if i == 0 else "")
    ax.grid(alpha=0.2, ls="--")
h = [plt.Line2D([], [], marker="o", ls="", color=pal[m], ms=7) for m in classes]
fig.legend(
    h,
    classes,
    loc="lower center",
    ncol=len(classes),
    fontsize=10,
    frameon=False,
    bbox_to_anchor=(0.5, 0.01),
)

for ext in ("pdf", "png"):
    fig.savefig(
        R / f"Figure6.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight"
    )

summary = pd.DataFrame(
    {
        "model": [MODELS[k][0] for k in KEYS],
        "hidden": [MODELS[k][1] for k in KEYS],
        "auc_pooled": [pooled[k] for k in KEYS],
        "auc_mean": [rep.loc[k, "mean"] for k in KEYS],
        "auc_sd": [rep.loc[k, "std"] for k in KEYS],
        "avg_bio_full": [bio_full[k] for k in KEYS],
        "avg_bio_boot_mean": [boot.loc[k, "mean"] for k in KEYS],
        "avg_bio_boot_sd": [boot.loc[k, "std"] for k in KEYS],
    }
)
summary.to_csv(R / "fig6_summary.csv", index=False)
print(summary.round(4).to_string(index=False))
