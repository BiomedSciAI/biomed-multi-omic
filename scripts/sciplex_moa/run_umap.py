"""
UMAP of each embedding for Fig 6E (same neighbors/umap calls as moa_predict.plot_umap_embeddings).

Writes <results>/umap_coords.npz.
"""
import argparse
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data")
    p.add_argument("--results", default="results")
    p.add_argument(
        "--keys", nargs="*", help="embedding keys (default: all in obsm.npz)"
    )
    a = p.parse_args()
    data, out = Path(a.data), Path(a.results)
    out.mkdir(parents=True, exist_ok=True)
    obs = pd.read_csv(data / "obs.csv", index_col=0)
    Z = np.load(data / "obsm.npz")
    coords = {}
    for k in a.keys or Z.files:
        ad_ = ad.AnnData(X=np.zeros((len(obs), 1), dtype=np.float32), obs=obs.copy())
        ad_.obsm["X_embedding"] = Z[k]
        sc.pp.neighbors(ad_, use_rep="X_embedding", random_state=42)
        sc.tl.umap(ad_, random_state=42)
        coords[k] = ad_.obsm["X_umap"]
        print(k, flush=True)
    np.savez_compressed(out / "umap_coords.npz", **coords)


if __name__ == "__main__":
    main()
