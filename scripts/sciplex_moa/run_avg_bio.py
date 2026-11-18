"""
Avg Bio (mean of NMI, ARI, ASW_label from scib.metrics) against MoA labels (Fig 6D).

Same scib.metrics call as moa_predict.plot_avg_bio_scores: one dummy batch, NMI/ARI on the
Leiden clustering at the NMI-optimal resolution. Computed on all cells, then on --n-boot
random 90% subsamples (rng seed 0; the same subsample for every embedding) for the error bars.

NMI/ARI depend on the Leiden implementation: use the versions in requirements.txt to
reproduce the paper's values.

Writes <results>/avg_bio.csv.
"""
import argparse
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scib.metrics as scm

warnings.filterwarnings("ignore")


def avg_bio(obs, X):
    a = ad.AnnData(X=np.zeros((len(obs), 1), dtype=np.float32), obs=obs.copy())
    a.obs["target"] = a.obs["target"].astype("category")
    a.obs["batch"] = "batch1"
    a.obsm["X_emb"] = X
    r = (
        scm.metrics(
            a,
            adata_int=a,
            batch_key="batch",
            label_key="target",
            embed="X_emb",
            isolated_labels_asw_=False,
            silhouette_=True,
            hvg_score_=False,
            pcr_=False,
            isolated_labels_f1_=False,
            trajectory_=False,
            nmi_=True,
            ari_=True,
            cell_cycle_=False,
            kBET_=False,
            ilisi_=False,
            clisi_=False,
            graph_conn_=False,
        )
        .iloc[:, 0]
        .to_dict()
    )
    nmi, ari, asw = (
        r.get("NMI_cluster/label"),
        r.get("ARI_cluster/label"),
        r.get("ASW_label"),
    )
    return {"NMI": nmi, "ARI": ari, "ASW": asw, "avg_bio": np.nanmean([nmi, ari, asw])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data")
    p.add_argument("--results", default="results")
    p.add_argument(
        "--keys", nargs="*", help="embedding keys (default: all in obsm.npz)"
    )
    p.add_argument("--n-boot", type=int, default=10)
    a = p.parse_args()
    data, out = Path(a.data), Path(a.results)
    out.mkdir(parents=True, exist_ok=True)
    obs = pd.read_csv(data / "obs.csv", index_col=0)
    Z = np.load(data / "obsm.npz")
    keys = a.keys or Z.files

    rows = []
    for k in keys:
        rows.append(dict(embedding=k, replicate="full", **avg_bio(obs, Z[k])))
        print(rows[-1], flush=True)
    rng = np.random.default_rng(0)
    for b in range(a.n_boot):
        idx = np.sort(rng.choice(len(obs), int(0.9 * len(obs)), replace=False))
        for k in keys:
            rows.append(
                dict(embedding=k, replicate=b, **avg_bio(obs.iloc[idx], Z[k][idx]))
            )
        print("subsample", b, "done", flush=True)
        pd.DataFrame(rows).to_csv(out / "avg_bio.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "avg_bio.csv", index=False)


if __name__ == "__main__":
    main()
