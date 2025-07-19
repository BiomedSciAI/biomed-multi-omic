import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
from anndata import AnnData


def plot_logits(
    cell_name: str,
    label: str,
    predictions: dict,
    adata: AnnData | None = None,
    top_n_logits: int = 10,
):
    import numpy as np

    logits = predictions["probabilities"]
    keep_labels = [l for l in logits.columns if l.startswith(label)]
    logits = logits[keep_labels]
    logits = logits[logits.index == cell_name]
    logits = logits.T.reset_index()
    logits = logits.rename(columns={"index": label, cell_name: f"{label}_logits"})
    logits = logits.sort_values(by=f"{label}_logits", ascending=False).head(
        top_n_logits
    )
    color = "blue"

    if adata:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        adata = adata.copy()
        adata.obs["highlight"] = adata.obs_names == cell_name
        idx = np.where(adata.obs_names == cell_name)[0][0]
        other_indices = np.setdiff1d(np.arange(adata.n_obs), idx)
        new_order = np.concatenate([other_indices, [idx]])
        adata = adata[new_order, :]
        sc.pl.umap(
            adata,
            color="highlight",
            palette=["lightgray", color],
            title=cell_name,
            sort_order="highlight",
            show=False,
            ax=axes[0],
        )

        logits = sns.barplot(
            data=logits, x=f"{label}_logits", y=label, color=color, ax=axes[1]
        )
    else:
        sns.barplot(data=logits, x=f"{label}_logits", y=label, color=color)

    plt.tight_layout()
    plt.show()


def plot_embeddings(adata, labels: list | str, title: str | None = None) -> None:
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    title = (f"UMAP - {labels}",)

    sc.tl.umap(adata, n_components=2)
    sc.pl.embedding(
        adata,
        basis="umap",
        title=title,
        color=labels,
        show=False,
    )

    plt.show()


def plot_cell_type_counts(adata: AnnData, key: str, ax=None, show: bool = False):
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    cell_counts = adata.obs[key].value_counts().reset_index()
    bar = sns.barplot(
        cell_counts,
        x=key,
        y="count",
        hue=key,
        order=cell_counts[key],
        ax=ax,
    )
    ax.set_title(f"N Counts - {key}")
    ax.tick_params(axis="x", labelrotation=90)

    if show:
        plt.show()
    else:
        return bar
