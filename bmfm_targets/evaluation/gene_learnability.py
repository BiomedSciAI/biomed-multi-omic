"""
Gene-level learnability metrics for expression-prediction models.

These functions turn the per-gene table written by
`bmfm_targets.training.metrics.batch_prediction_metrics.get_gene_level_expression_error`
(columns `gene_err_nz_by_logits`, `gene_err_nz_null`, `gene_avg_nz_expressions`,
`gene_nz_freq`, `is_zero_roc_auc`) into the per-gene skills and the family
significance gate used for the gene-learnability figures in the BMFM-RNA paper.

Two skills, on different scales:

- detection: raw `is_zero_roc_auc`. AUC does not depend on prevalence, so it is
  not detrended. `det_dev` is the AUC minus the median over reliable genes.
- quantification: `q_skill`, the log MAE on expressed cells relative to genes of
  similar expression (see `quantification_skill`).

`mae_over_null` is the null-referenced quantification metric. `q_skill` is not
null-referenced; a high `q_skill` does not mean the gene beats its null.
"""

from collections.abc import Sequence
from importlib import resources

import numpy as np
import pandas as pd

GENE_ANNOTATIONS_FILE = "gene_families.csv"

MHC_FAMILIES = (
    "MHC Class I (Classical)",
    "MHC Class II",
    "MHC Class I (Non-Classical)",
    "MHC / Antigen Presentation (Other)",
)
MHC_AGGREGATE = "MHC / Antigen Presentation"


def load_gene_annotations(
    aggregate_mhc: bool = False, normalize_case: bool = True
) -> pd.DataFrame:
    """
    Load the packaged two-axis gene annotation table.

    Each of the 19,027 genes has one functional `family` (34 families, assigned in
    the priority order of Supplementary Table S1) and one `expr_regime`
    (`state_dissociation`, `state_cell_cycle`, `constitutive`,
    `identity_associated`, `unassigned`). The table is rebuilt from public sources
    by `bmfm_targets.evaluation.gene_annotations.build`.

    Args:
    ----
        aggregate_mhc: merge the four MHC sub-families into a single
            "MHC / Antigen Presentation" family (23 genes). The sub-families
            have 3-13 genes each, below the family gate's default size floor.
        normalize_case: upper-case gene symbols (`C1orf74` -> `C1ORF74`) for
            joining to upper-cased per-gene tables.

    Returns:
    -------
        DataFrame with columns `gene`, `family`, `expr_regime`, sorted by gene.

    """
    path = resources.files("bmfm_targets.evaluation.resources").joinpath(
        GENE_ANNOTATIONS_FILE
    )
    with resources.as_file(path) as p:
        table = pd.read_csv(p, keep_default_na=False)
    if normalize_case:
        table["gene"] = table["gene"].str.upper()
    if aggregate_mhc:
        table["family"] = table["family"].replace(
            dict.fromkeys(MHC_FAMILIES, MHC_AGGREGATE)
        )
    return table


def mae_over_null(
    per_gene: pd.DataFrame,
    err_col: str = "gene_err_nz_by_logits",
    null_col: str = "gene_err_nz_null",
) -> pd.Series:
    """
    Per-gene MAE on expressed cells divided by the MAE of the per-gene mean null.

    The null predicts each gene's mean non-zero expression for every expressed
    cell. Values below 1 beat the null.

    `err_col` defaults to the ungated regression output. For WCED models the gated
    prediction (`gene_err_nz`) is zeroed wherever the is_zero head fires, which
    inflates the error on expressed cells; use the ungated column when comparing
    quantification skill.
    """
    missing = [c for c in (err_col, null_col) if c not in per_gene.columns]
    if missing:
        raise KeyError(f"per-gene table is missing columns {missing}")
    return (per_gene[err_col] / per_gene[null_col]).rename("mae_over_null")


def quantification_skill(
    log_expr,
    log_mae,
    n_bins: int = 40,
    frac_q: float = 0.4,
    q: Sequence[float] = (0.10, 0.50, 0.90),
) -> pd.DataFrame:
    """
    Quantification skill: log MAE relative to genes at the same expression level.

    Log MAE rises steeply with expression (Spearman rho ~0.88 on the WCED
    checkpoint), so raw MAE mostly ranks genes by expression. This removes that
    trend non-parametrically:

    1. Sort genes by `log_expr` and split them into `n_bins` equal-count bins.
    2. In each bin, take the `q[0]`, `q[1]`, `q[2]` quantiles of `log_mae`.
    3. Smooth each quantile curve against the bin mean expression with LOWESS
       (`frac=frac_q`, 3 robustness iterations) and interpolate it back to every
       gene, holding the end values constant outside the bin range.
    4. Residual = log MAE - smoothed median. Scale negative residuals by the
       lower half-band (median - low) and positive ones by the upper half-band
       (high - median).

    `q_skill` is the negated scaled residual, so higher is better: 0 means typical
    for the gene's expression level, +1 means the error sits on the low quantile
    curve, -1 on the high one.

    `q_skill` is expression-detrended, not null-referenced. A gene with high
    `q_skill` has low error compared to genes of similar expression, which says
    nothing about whether it beats its own per-gene null. Use `mae_over_null` for
    that.

    Args:
    ----
        log_expr: per-gene log mean expression on expressed cells.
        log_mae: per-gene log MAE on expressed cells.
        n_bins: number of equal-count expression bins.
        frac_q: LOWESS span, as a fraction of the bins.
        q: (low, median, high) quantiles that define the band.

    Returns:
    -------
        DataFrame indexed like `log_expr` (if it is a Series) with columns
        `q_skill`, `trend` (smoothed median), `lo`, `hi` (band edges), `win`
        (log MAE at or below the low curve), `los` (at or above the high curve).

    """
    import statsmodels.api as sm

    index = log_expr.index if isinstance(log_expr, pd.Series) else None
    x = np.asarray(log_expr, dtype=float)
    y = np.asarray(log_mae, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("log_expr and log_mae must be 1-D and the same length")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("log_expr and log_mae must be finite; drop NaN/inf first")
    if len(x) < 2 * n_bins:
        raise ValueError(f"need at least {2 * n_bins} genes for {n_bins} bins")
    q_lo, q_md, q_hi = q

    order = np.argsort(x)
    xs, ys = x[order], y[order]
    bins = np.array_split(np.arange(len(xs)), n_bins)
    bin_x = np.array([xs[b].mean() for b in bins])

    def smoothed(quantile: float) -> np.ndarray:
        bin_q = np.array([np.percentile(ys[b], quantile * 100) for b in bins])
        curve = sm.nonparametric.lowess(
            bin_q, bin_x, frac=frac_q, it=3, return_sorted=False
        )
        return np.interp(x, bin_x, curve)

    lo, md, hi = smoothed(q_lo), smoothed(q_md), smoothed(q_hi)
    half_lo = np.abs(md - lo).clip(1e-8)
    half_hi = np.abs(hi - md).clip(1e-8)
    resid = y - md
    z = np.where(resid < 0, resid / half_lo, resid / half_hi)
    return pd.DataFrame(
        {
            "q_skill": -z,
            "trend": md,
            "lo": lo,
            "hi": hi,
            "win": y <= lo,
            "los": y >= hi,
        },
        index=index,
    )


def gene_skill_table(
    per_gene: pd.DataFrame,
    min_nz_freq: int = 50,
    err_col: str = "gene_err_nz_by_logits",
    null_col: str = "gene_err_nz_null",
    expr_col: str = "gene_avg_nz_expressions",
    nz_freq_col: str = "gene_nz_freq",
    auc_col: str = "is_zero_roc_auc",
) -> pd.DataFrame:
    """
    Per-gene detection and quantification skills for reliable genes.

    Keeps genes expressed in at least `min_nz_freq` evaluation cells that have a
    detection AUC, then adds:

    - `det_auc`: the raw is_zero ROC AUC; `det_dev`: AUC minus its median.
    - `det_win` / `det_los`: AUC at or above the 90th / at or below the 10th
      percentile.
    - `mae_over_null`: see `mae_over_null`.
    - `log_gt`, `log_mae`: natural logs of expression and error, clipped below
      at 1e-4 and 1e-6.
    - `q_skill`, `q_win`, `q_los`: see `quantification_skill`.

    Column defaults match the output of `get_gene_level_expression_error`.
    """
    table = per_gene[per_gene[nz_freq_col] >= min_nz_freq].copy()
    table = table.dropna(subset=[auc_col, err_col, null_col, expr_col])

    auc = table[auc_col].astype(float)
    det_lo, det_hi = np.percentile(auc, [10, 90])
    table["det_auc"] = auc
    table["det_dev"] = auc - auc.median()
    table["det_win"] = auc >= det_hi
    table["det_los"] = auc <= det_lo

    table["mae_over_null"] = mae_over_null(table, err_col, null_col)
    table["log_gt"] = np.log(table[expr_col].astype(float).clip(lower=1e-4))
    table["log_mae"] = np.log(table[err_col].astype(float).clip(lower=1e-6))
    skill = quantification_skill(table["log_gt"], table["log_mae"])
    table["q_skill"] = skill["q_skill"]
    table["q_win"] = skill["win"]
    table["q_los"] = skill["los"]
    return table


def bh_fdr(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values), in input order."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    out = np.empty(n)
    out[order] = ranked
    return out


def _ks_vs_pool(pool_sorted: np.ndarray, sample_sorted: np.ndarray) -> np.ndarray:
    """
    Two-sample KS statistic of each sorted sample (last axis) against the pool.

    The sample CDF only jumps at sample points, so the supremum of
    |F_sample - F_pool| is reached at a sample point: either just after it
    (F_sample = i/n vs the right-continuous F_pool) or just before it
    (F_sample = (i-1)/n vs the left limit of F_pool). Both sides are needed.
    """
    m = len(pool_sorted)
    n = sample_sorted.shape[-1]
    upper = np.arange(1, n + 1) / n
    lower = np.arange(0, n) / n
    pool_right = np.searchsorted(pool_sorted, sample_sorted, side="right") / m
    pool_left = np.searchsorted(pool_sorted, sample_sorted, side="left") / m
    d_plus = np.max(upper - pool_right, axis=-1)
    d_minus = np.max(pool_left - lower, axis=-1)
    return np.maximum(d_plus, d_minus)


def _family_null(
    pool_sorted: np.ndarray,
    fam_vals: np.ndarray,
    rng: np.random.Generator,
    n_perm: int,
    chunk: int,
) -> tuple[float, float, float, float]:
    """Observed median and KS of one family, with Monte-Carlo p-values."""
    m = len(pool_sorted)
    n = len(fam_vals)
    obs_med = float(np.median(fam_vals))
    obs_ks = float(_ks_vs_pool(pool_sorted, np.sort(fam_vals)))

    ge_med = le_med = ge_ks = 0
    done = 0
    while done < n_perm:
        c = min(chunk, n_perm - done)
        draws = np.sort(pool_sorted[rng.integers(0, m, size=(c, n))], axis=1)
        null_med = np.median(draws, axis=1)
        ge_med += int(np.sum(null_med >= obs_med - 1e-12))
        le_med += int(np.sum(null_med <= obs_med + 1e-12))
        null_ks = _ks_vs_pool(pool_sorted, draws)
        ge_ks += int(np.sum(null_ks >= obs_ks - 1e-12))
        done += c

    p_med = min(1.0, 2 * min((1 + le_med) / (n_perm + 1), (1 + ge_med) / (n_perm + 1)))
    p_ks = (1 + ge_ks) / (n_perm + 1)
    return obs_med, p_med, obs_ks, p_ks


def family_permutation_gate(
    values: pd.Series,
    families: pd.Series,
    n_perm: int = 20000,
    q_thresh: float = 0.05,
    ks_thresh: float = 0.10,
    min_size: int = 10,
    seed: int = 0,
    chunk: int = 1000,
) -> pd.DataFrame:
    """
    Test whether each gene family's skill differs from a random gene set of the same size.

    The pool is every gene with a non-NaN value. For each family with at least
    `min_size` genes, draw `n_perm` random size-n gene sets from the pool, with
    replacement, and compare:

    - the family median to the null medians (two-sided p, +1 corrected);
    - the two-sample KS statistic, family vs pool, to the null KS values
      (one-sided p, +1 corrected).

    Drawing with replacement differs from a permutation only for families that
    are a large fraction of the pool; there it widens the null slightly, so the
    test is conservative for the largest families.

    p-values are BH-adjusted across families. A family passes the gate when
    `q_median < q_thresh` and `ks >= ks_thresh`. The KS floor exists because
    with large n the median test flags families whose distribution matches the
    genome: the 4,334-gene "Unclassified" family has delta median -0.004 AUC and
    KS 0.03, but q_median ~0.003.

    In the paper, a family is "distinct" if it passes the gate on at least one
    WCED skill (detection `det_dev` or quantification `q_skill`).

    Args:
    ----
        values: per-gene skill, indexed by gene.
        families: per-gene family label, indexed by gene. Genes missing from
            either series are dropped.
        n_perm: Monte-Carlo draws per family.
        q_thresh: BH q threshold for the median test.
        ks_thresh: minimum KS effect size.
        min_size: smallest family tested.
        seed: seed for the random draws.
        chunk: draws per vectorised block; bounds memory at chunk * n floats.

    Returns:
    -------
        DataFrame with one row per tested family, sorted by family name:
        `family`, `n`, `pool_median`, `obs_median`, `delta_median`, `p_median`,
        `q_median`, `ks`, `p_ks`, `q_ks`, `gate`.

    """
    joined = pd.concat({"value": values, "family": families}, axis=1, join="inner")
    joined = joined.dropna(subset=["value", "family"])
    pool_sorted = np.sort(joined["value"].to_numpy(dtype=float))
    pool_median = float(np.median(pool_sorted))
    counts = joined["family"].value_counts()
    tested = sorted(counts[counts >= min_size].index)

    rng = np.random.default_rng(seed)
    rows = []
    for fam in tested:
        fam_vals = joined.loc[joined["family"] == fam, "value"].to_numpy(dtype=float)
        obs_med, p_med, ks, p_ks = _family_null(
            pool_sorted, fam_vals, rng, n_perm, chunk
        )
        rows.append(
            {
                "family": fam,
                "n": len(fam_vals),
                "pool_median": pool_median,
                "obs_median": obs_med,
                "delta_median": obs_med - pool_median,
                "p_median": p_med,
                "ks": ks,
                "p_ks": p_ks,
            }
        )
    columns = [
        "family",
        "n",
        "pool_median",
        "obs_median",
        "delta_median",
        "p_median",
        "q_median",
        "ks",
        "p_ks",
        "q_ks",
        "gate",
    ]
    out = pd.DataFrame(
        rows, columns=[c for c in columns if c not in ("q_median", "q_ks", "gate")]
    )
    out["q_median"] = bh_fdr(out["p_median"])
    out["q_ks"] = bh_fdr(out["p_ks"])
    out["gate"] = (out["q_median"] < q_thresh) & (out["ks"] >= ks_thresh)
    return out[columns]
