"""
Compare models over a set of benchmark blocks.

Every function takes a ``scores`` table with one row per model and one column per block
(for example a metric on one tissue), where higher is better.
"""

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


def rank_models(scores: pd.DataFrame) -> pd.DataFrame:
    """Rank models within each block (1 = best, ties share the average rank)."""
    _check_complete(scores)
    ranks = scores.rank(axis=0, ascending=False, method="average")
    return ranks.loc[ranks.mean(axis=1).sort_values(kind="stable").index]


def holm_adjust(pvalues) -> np.ndarray:
    """Holm step-down adjusted p values, in the input order."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p, kind="stable")
    stepped = p[order] * (len(p) - np.arange(len(p)))
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(np.maximum.accumulate(stepped), 1.0)
    return adjusted


def paired_rank_tests(ranks: pd.DataFrame, references: list[str]) -> pd.DataFrame:
    """
    Compare each reference model with every other model by paired ranks.

    A two-sided Wilcoxon signed-rank test is run over the blocks, dropping ties. The p
    values are Holm-adjusted across the comparisons of each reference model. Blocks that
    share data (several metrics on one tissue) are not independent, so the p values are
    then approximate.

    Returns
    -------
        One row per (reference, other) pair: the number of blocks in which the reference
        ranks better, worse or tied, the mean ranks, ``p`` and ``p_holm``.
    """
    _check_complete(ranks)
    rows = []
    for ref in references:
        for other in ranks.index.drop(ref):
            diff = (ranks.loc[other] - ranks.loc[ref]).to_numpy()
            rows.append(
                {
                    "reference": ref,
                    "other": other,
                    "better": int((diff > 0).sum()),
                    "worse": int((diff < 0).sum()),
                    "tied": int((diff == 0).sum()),
                    "mean_rank_reference": ranks.loc[ref].mean(),
                    "mean_rank_other": ranks.loc[other].mean(),
                    "p": wilcoxon(diff).pvalue if diff.any() else 1.0,
                }
            )
    result = pd.DataFrame(rows)
    result["p_holm"] = result.groupby("reference")["p"].transform(holm_adjust)
    return result


def friedman_test(scores: pd.DataFrame) -> tuple[float, float]:
    """Friedman test of equal performance over blocks: (statistic, p value)."""
    _check_complete(scores)
    result = friedmanchisquare(*scores.to_numpy())
    return float(result.statistic), float(result.pvalue)


def mean_zscore(scores: pd.DataFrame) -> pd.Series:
    """Mean over blocks of each model's z-score against the field (sample std)."""
    _check_complete(scores)
    return ((scores - scores.mean()) / scores.std(ddof=1)).mean(axis=1)


def pareto_front(cost: pd.Series, score: pd.Series) -> list[str]:
    """Models that no other model beats on score at equal or lower cost, by cost."""
    table = pd.DataFrame({"cost": cost, "score": score.reindex(cost.index)})
    _check_complete(table)
    table = table.sort_values(["cost", "score"], ascending=[True, False])
    previous_best = table["score"].cummax().shift(fill_value=-np.inf)
    return table.index[table["score"] > previous_best].tolist()


def _check_complete(table: pd.DataFrame) -> None:
    if table.index.has_duplicates:
        raise ValueError("Model names must be unique")
    if table.isna().any().any():
        incomplete = table.index[table.isna().any(axis=1)].tolist()
        raise ValueError(f"Missing values for {incomplete}")
