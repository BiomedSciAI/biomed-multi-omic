"""
MoA classification AUC with stratified drug-holdout cross-validation (Fig 6A-B).

Logistic regression on each frozen embedding, moa_predict.stratified_drug_group_cv with
5 repeats (folds are reduced automatically to 2 so that every fold holds out at least one
drug per MoA class), seed 42. Micro-averaged ROC AUC on the held-out predictions, pooled
over all repeats and per repeat.

Writes <results>/auc_per_repeat.csv and <results>/roc_curves.npz.
"""
import argparse
from pathlib import Path

import moa_predict as M
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, label_binarize


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
    keys = a.keys or Z.files
    le = LabelEncoder()
    y = le.fit_transform(obs["target"].values)
    prod = obs["product_name"].values
    K = len(le.classes_)

    rows, curves = [], {}
    for k in keys:
        r = M.stratified_drug_group_cv(
            Z[k], y, prod, n_folds=5, n_repeats=5, random_seed=42, max_iter=1000
        )
        yt, yp, rep = (
            np.array(r["y_true"]),
            np.vstack(r["y_prob"]),
            np.array(r["repeat"]),
        )

        def micro(m):
            b = label_binarize(yt[m], classes=range(K))
            f, t, _ = roc_curve(b.ravel(), yp[m].ravel())
            return auc(f, t), f, t

        a_, f, t = micro(np.ones(len(yt), bool))
        curves[k] = (f, t)
        rows.append(
            {
                "embedding": k,
                "repeat": "pooled",
                "micro_auc": a_,
                "macro_auc": roc_auc_score(yt, yp, multi_class="ovr", average="macro"),
            }
        )
        for q in np.unique(rep):
            m = rep == q
            rows.append(
                {
                    "embedding": k,
                    "repeat": int(q),
                    "micro_auc": micro(m)[0],
                    "macro_auc": roc_auc_score(
                        yt[m], yp[m], multi_class="ovr", average="macro"
                    ),
                }
            )
        print(k, round(a_, 4), flush=True)

    pd.DataFrame(rows).to_csv(out / "auc_per_repeat.csv", index=False)
    np.savez_compressed(
        out / "roc_curves.npz",
        **{f"{k}__fpr": v[0] for k, v in curves.items()},
        **{f"{k}__tpr": v[1] for k, v in curves.items()},
    )


if __name__ == "__main__":
    main()
