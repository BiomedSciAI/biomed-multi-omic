"""
Filter the Sci-Plex 3 A549 AnnData to the MoA task set and export it for the other scripts.

Applies moa_predict.load_and_prepare_data with its defaults (no vehicle, dose 10,000 nM,
viability < 0.85, no "Others" class, >= 2 drugs per MoA). On the paper's data this gives
4,667 cells, 37 drugs, 9 MoA classes.

Writes <out>/obs.csv and <out>/obsm.npz (one array per embedding key in adata.obsm).
"""
import argparse
from pathlib import Path

import moa_predict as M
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        required=True,
        help="h5ad with obs target/product_name/dose/viability and embeddings in obsm",
    )
    p.add_argument("--out", default="data", help="output directory")
    p.add_argument(
        "--keys", nargs="*", help="obsm keys to export (default: all except X_umap)"
    )
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    adata, _ = M.load_and_prepare_data(a.input, embedding_keys=[])
    keys = a.keys or [k for k in adata.obsm if k != "X_umap"]
    print(
        f"{adata.n_obs} cells, {adata.obs.product_name.nunique()} drugs, {adata.obs.target.nunique()} MoA classes"
    )
    adata.obs.to_csv(out / "obs.csv")
    np.savez_compressed(
        out / "obsm.npz", **{k: np.asarray(adata.obsm[k]) for k in keys}
    )


if __name__ == "__main__":
    main()
