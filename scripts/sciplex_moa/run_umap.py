"""
UMAP of each embedding for Fig 6E (scanpy neighbors + umap, default parameters).

Writes <results>/umap_coords.npz.
"""
import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data")
    p.add_argument("--results", default="results")
    p.add_argument(
        "--keys", nargs="*", help="embedding keys (default: all in obsm.npz)"
    )
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    data, out = Path(a.data), Path(a.results)
    out.mkdir(parents=True, exist_ok=True)
    obs = pd.read_csv(data / "obs.csv", index_col=0)
    Z = np.load(data / "obsm.npz")
    coords = {}
    for k in a.keys or Z.files:
        adata = ad.AnnData(X=np.zeros((len(obs), 1), dtype=np.float32), obs=obs.copy())
        adata.obsm["X_embedding"] = Z[k]
        sc.pp.neighbors(adata, use_rep="X_embedding", random_state=a.seed)
        sc.tl.umap(adata, random_state=a.seed)
        coords[k] = adata.obsm["X_umap"]
        print(k, flush=True)
    np.savez_compressed(out / "umap_coords.npz", **coords)


if __name__ == "__main__":
    main()
