"""
Avg Bio (mean of NMI, ARI, ASW_label from scib.metrics) against MoA labels (Fig 6D).

moa_lib.scib_bio_metrics: one dummy batch, NMI/ARI on the Leiden clustering at the
NMI-optimal resolution. Computed on all cells (replicate "full"), then on
--n-subsamples random 90% subsamples drawn without replacement (the same subsamples for
every embedding) for the error bars.

NMI/ARI depend on the Leiden implementation: use the versions in requirements.txt to
reproduce the paper's values.

Writes <results>/avg_bio.csv.
"""
import argparse
from pathlib import Path

import moa_lib as M
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data")
    p.add_argument("--results", default="results")
    p.add_argument(
        "--keys", nargs="*", help="embedding keys (default: all in obsm.npz)"
    )
    p.add_argument("--n-subsamples", type=int, default=10)
    p.add_argument("--fraction", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0, help="seed for drawing subsamples")
    a = p.parse_args()
    data, out = Path(a.data), Path(a.results)
    out.mkdir(parents=True, exist_ok=True)
    obs = pd.read_csv(data / "obs.csv", index_col=0)
    Z = np.load(data / "obsm.npz")
    keys = a.keys or Z.files

    rows = []

    def save():
        pd.DataFrame(rows).to_csv(out / "avg_bio.csv", index=False)

    for k in keys:
        rows.append(
            dict(embedding=k, replicate="full", **M.scib_bio_metrics(obs, Z[k]))
        )
        print(rows[-1], flush=True)
    save()
    rng = np.random.default_rng(a.seed)
    for b in range(a.n_subsamples):
        idx = np.sort(rng.choice(len(obs), int(a.fraction * len(obs)), replace=False))
        for k in keys:
            rows.append(
                dict(
                    embedding=k,
                    replicate=b,
                    **M.scib_bio_metrics(obs.iloc[idx], Z[k][idx]),
                )
            )
        print("subsample", b, "done", flush=True)
        save()


if __name__ == "__main__":
    main()
