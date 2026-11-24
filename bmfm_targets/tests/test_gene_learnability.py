import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

from bmfm_targets.evaluation import gene_learnability as gl
from bmfm_targets.training.metrics.batch_prediction_metrics import (
    get_gene_level_expression_error,
)

# --- annotations: oracle is Supplementary Table S1 and the Axis 2 section ---

SUPP_TABLE_S1 = {
    "Mitochondrial Genome": 13,
    "Ribosome (Cytosolic)": 91,
    "Ribosome (Mitochondrial)": 77,
    "MHC Class I (Classical)": 4,
    "MHC Class II": 13,
    "MHC Class I (Non-Classical)": 3,
    "MHC / Antigen Presentation (Other)": 3,
    "TF: Zinc-Coordinating": 673,
    "TF: Helix-Turn-Helix": 362,
    "TF: Basic Domains": 139,
    "TF: Other (Structural)": 116,
    "TF: Unclassified": 281,
    "Cilia & Flagella": 510,
    "Cytoskeleton & Motor": 1082,
    "Histone": 75,
    "GTPase / Trafficking": 271,
    "Kinase": 632,
    "Phosphatase": 229,
    "Synapse": 868,
    "Cell Adhesion": 539,
    "Cytokine / Growth Factor": 284,
    "Olfactory Receptor": 403,
    "GPCR (Non-Olfactory)": 300,
    "Ion Channel": 287,
    "Transporter (SLC/ABC)": 714,
    "Surface Marker (CD)": 58,
    "Cell Cycle": 167,
    "DNA Repair": 395,
    "Ubiquitin Ligase": 350,
    "Peptidase": 488,
    "Extracellular Matrix": 203,
    "RNA Binding": 1103,
    "Enzyme: Hydrolase": 560,
    "Enzyme: Transferase": 756,
    "Enzyme: Oxidoreductase": 498,
    "Enzyme: Other (Lyase/Ligase/Iso)": 226,
    "Unclassified": 6254,
}

SUPP_REGIMES = {
    "state_dissociation": 154,
    "state_cell_cycle": 210,
    "constitutive": 3338,
    "identity_associated": 3373,
    "unassigned": 11952,
}


def test_annotation_family_counts_match_supp_table_s1():
    ann = gl.load_gene_annotations()
    assert len(ann) == 19027
    assert ann["gene"].is_unique
    assert ann["family"].value_counts().to_dict() == SUPP_TABLE_S1
    assert ann["expr_regime"].value_counts().to_dict() == SUPP_REGIMES


def test_annotation_spot_checks():
    ann = gl.load_gene_annotations().set_index("gene")
    assert ann.loc["MT-CO1", "family"] == "Mitochondrial Genome"
    assert ann.loc["RPL13A", "family"] == "Ribosome (Cytosolic)"
    assert ann.loc["HLA-DRA", "family"] == "MHC Class II"
    assert ann.loc["FOS", "expr_regime"] == "state_dissociation"
    assert ann.loc["MKI67", "expr_regime"] == "state_cell_cycle"


def test_annotation_mhc_aggregation():
    ann = gl.load_gene_annotations(aggregate_mhc=True)
    counts = ann["family"].value_counts()
    assert counts[gl.MHC_AGGREGATE] == 4 + 13 + 3 + 3
    assert not set(gl.MHC_FAMILIES) & set(counts.index)
    assert counts.sum() == 19027


# --- mae_over_null: hand-computed case through the library's per-gene table ---


def test_mae_over_null_hand_computed():
    # Gene A, expressed cells [1, 3]: nonzero-mean null = 2, null MAE = 1.
    #   ungated regression output exact -> MAE 0 -> ratio 0.
    #   the gate zeroes the second cell, so the gated MAE is (0 + 3) / 2 = 1.5.
    # Gene B, expressed cells [2, 6]: null = 4, null MAE = 2.
    #   regression output [4, 4] -> MAE 2 -> ratio 1 (ties the null).
    exp_preds = pd.DataFrame(
        {
            "input_genes": ["A", "A", "A", "B", "B", "B"],
            "input_expressions": [1, 3, 0, 2, 6, 0],
            "label_expressions": [1.0, 3.0, 0.0, 2.0, 6.0, 0.0],
            "predicted_expressions": [1.0, 0.0, 0.0, 4.0, 4.0, 0.0],
            "logits_expressions_mse": [1.0, 3.0, 0.5, 4.0, 4.0, 0.5],
        }
    )
    per_gene = get_gene_level_expression_error(exp_preds)
    ratio = gl.mae_over_null(per_gene)
    assert ratio["A"] == pytest.approx(0.0)
    assert ratio["B"] == pytest.approx(1.0)
    gated = gl.mae_over_null(per_gene, err_col="gene_err_nz")
    assert gated["A"] == pytest.approx(1.5)


def test_mae_over_null_missing_column_raises():
    with pytest.raises(KeyError, match="gene_err_nz_by_logits"):
        gl.mae_over_null(pd.DataFrame({"gene_err_nz_null": [1.0]}))


# --- quantification_skill ---


def _trend(x):
    # smooth and nearly monotone, like log MAE vs log expression
    return 0.8 * x + 0.05 * x**2


@pytest.fixture(scope="module")
def skill_background():
    rng = np.random.default_rng(0)
    x = rng.uniform(-3, 3, 4000)
    y = _trend(x) + rng.normal(0, 0.3, x.size)
    return x, y


def test_skill_is_zero_on_the_trend(skill_background):
    x, y = skill_background
    probe_x = np.linspace(-2.5, 2.5, 21)
    xs = np.concatenate([x, probe_x])
    ys = np.concatenate([y, _trend(probe_x)])
    skill = gl.quantification_skill(xs, ys)["q_skill"].to_numpy()
    assert np.abs(skill[-len(probe_x) :]).max() < 0.1


def test_skill_band_calibration(skill_background):
    # Homoscedastic noise: about 10% of genes fall below the low curve and 10%
    # above the high curve, and q_skill = +-1 sits at the 10/90% noise quantiles.
    x, y = skill_background
    skill = gl.quantification_skill(x, y)
    assert skill["win"].mean() == pytest.approx(0.10, abs=0.02)
    assert skill["los"].mean() == pytest.approx(0.10, abs=0.02)
    half_band = 0.3 * stats.norm.ppf(0.9)
    assert np.median(skill["hi"] - skill["trend"]) == pytest.approx(half_band, rel=0.1)


def test_skill_ranks_half_error_gene_in_top_decile(skill_background):
    x, y = skill_background
    xs = np.append(x, 0.5)
    ys = np.append(y, _trend(0.5) + np.log(0.5))
    skill = gl.quantification_skill(xs, ys)["q_skill"].to_numpy()
    assert (skill[:-1] < skill[-1]).mean() > 0.9


def test_skill_ignores_expression_level(skill_background):
    # The same residual at low and high expression gets the same skill.
    x, y = skill_background
    xs = np.append(x, [-2.0, 2.0])
    ys = np.append(y, [_trend(-2.0) - 0.4, _trend(2.0) - 0.4])
    skill = gl.quantification_skill(xs, ys)["q_skill"].to_numpy()
    assert skill[-2] == pytest.approx(skill[-1], abs=0.25)
    assert skill[-2] > 0.8


def test_skill_does_not_depend_on_gene_order(skill_background):
    x, y = skill_background
    genes = pd.Index([f"g{i}" for i in range(len(x))])
    ref = gl.quantification_skill(pd.Series(x, genes), pd.Series(y, genes))
    perm = np.random.default_rng(1).permutation(len(x))
    shuffled = gl.quantification_skill(
        pd.Series(x[perm], genes[perm]), pd.Series(y[perm], genes[perm])
    )
    pd.testing.assert_frame_equal(shuffled.loc[genes], ref)


def test_skill_rejects_bad_input():
    with pytest.raises(ValueError, match="finite"):
        gl.quantification_skill(np.r_[np.arange(100.0), np.nan], np.arange(101.0))
    with pytest.raises(ValueError, match="at least 80"):
        gl.quantification_skill(np.arange(50.0), np.arange(50.0))


# --- gene_skill_table: end to end from a prediction frame ---


def _prediction_frame(n_genes=300, n_cells=120, seed=0):
    rng = np.random.default_rng(seed)
    genes = np.repeat([f"g{i}" for i in range(n_genes)], n_cells)
    mean = np.repeat(rng.uniform(0.5, 4.0, n_genes), n_cells)
    nz_prob = np.repeat(rng.uniform(0.3, 0.9, n_genes), n_cells)
    expressed = rng.random(genes.size) < nz_prob
    label = np.where(expressed, mean * rng.lognormal(0, 0.3, genes.size), 0.0)
    regression = label + rng.normal(0, 0.2, genes.size) * mean
    is_zero_score = (~expressed) + rng.normal(0, 0.8, genes.size)
    gated = np.where(is_zero_score > 0.5, 0.0, regression)
    frame = pd.DataFrame(
        {
            "input_genes": genes,
            "input_expressions": label,
            "label_expressions": label,
            "predicted_expressions": gated,
            "logits_expressions_mse": regression,
            "logits_expressions_is_zero": is_zero_score,
        }
    )
    # gene g0 gets a perfect ungated regression output
    frame.loc[frame.input_genes == "g0", "logits_expressions_mse"] = frame.loc[
        frame.input_genes == "g0", "label_expressions"
    ]
    # gene g1 is expressed in too few cells to be reliable
    g1 = np.flatnonzero(frame.input_genes == "g1")
    frame.loc[g1[10:], ["label_expressions", "input_expressions"]] = 0.0
    return frame


def test_gene_skill_table_end_to_end():
    per_gene = get_gene_level_expression_error(_prediction_frame())
    table = gl.gene_skill_table(per_gene, min_nz_freq=20)
    assert "g1" not in table.index
    assert len(table) == 299
    # g0's ungated regression output is exact; its gated output is not.
    assert table.loc["g0", "mae_over_null"] == 0
    assert per_gene.loc["g0", "gene_err_nz"] > 0
    assert table["q_skill"].idxmax() == "g0"
    # regression noise sd 0.2 * mean is below the lognormal(0, 0.3) spread the
    # per-gene mean null has to absorb, so genes beat the null.
    assert (table["mae_over_null"] < 1).mean() > 0.9
    assert table["det_dev"].median() == pytest.approx(0.0)
    assert table["det_win"].mean() == pytest.approx(0.1, abs=0.01)
    assert table["q_skill"].notna().all()


# --- bh_fdr and family_permutation_gate ---


def test_bh_fdr_matches_statsmodels():
    p = np.random.default_rng(0).uniform(size=200) ** 2
    expected = multipletests(p, method="fdr_bh")[1]
    np.testing.assert_allclose(gl.bh_fdr(p), expected)
    assert gl.bh_fdr([]).size == 0


def _random_families(rng, n_genes, sizes):
    genes = pd.Index([f"g{i}" for i in range(n_genes)])
    labels = np.full(n_genes, "background", dtype=object)
    picked = rng.choice(n_genes, size=sum(sizes), replace=False)
    start = 0
    for k, size in enumerate(sizes):
        labels[picked[start : start + size]] = f"fam{k:02d}"
        start += size
    return genes, pd.Series(labels, index=genes)


def test_gate_ks_matches_scipy():
    rng = np.random.default_rng(0)
    genes, families = _random_families(rng, 2000, [15, 60, 200])
    values = pd.Series(rng.normal(size=2000), index=genes)
    out = gl.family_permutation_gate(values, families, n_perm=200).set_index("family")
    for fam in ("fam00", "fam01", "fam02"):
        ref = stats.ks_2samp(values[families == fam], values).statistic
        assert out.loc[fam, "ks"] == pytest.approx(ref)
        assert out.loc[fam, "obs_median"] == pytest.approx(
            values[families == fam].median()
        )


def test_gate_false_positive_rate_at_nominal_level():
    # Families are random gene sets, so the median-test p-values are uniform.
    rng = np.random.default_rng(1)
    p_values = []
    for rep in range(3):
        sizes = rng.integers(10, 150, size=50).tolist()
        genes, families = _random_families(rng, 12000, sizes)
        values = pd.Series(rng.normal(size=12000), index=genes)
        out = gl.family_permutation_gate(
            values, families[families != "background"], n_perm=2000, seed=rep
        )
        assert len(out) == 50
        assert not out["gate"].any()
        p_values.extend(out["p_median"])
    p_values = np.asarray(p_values)
    assert (p_values < 0.05).mean() == pytest.approx(0.05, abs=0.04)
    assert stats.kstest(p_values, "uniform").pvalue > 0.01


def test_gate_flags_shifted_family_and_respects_size():
    rng = np.random.default_rng(2)
    genes, families = _random_families(rng, 5000, [40, 40, 5])
    values = pd.Series(rng.normal(size=5000), index=genes)
    values[families == "fam00"] += 1.0  # 1 SD shift
    values[families == "fam02"] += 1.0  # same shift, below min_size
    out = gl.family_permutation_gate(values, families, n_perm=2000).set_index("family")
    assert "fam02" not in out.index
    assert out.loc["fam00", "gate"]
    assert out.loc["fam00", "delta_median"] == pytest.approx(1.0, abs=0.35)
    assert not out.loc["fam01", "gate"]


def test_gate_ks_floor_blocks_large_family_with_tiny_shift():
    # A large family with a small location shift is significant by the median
    # test but fails the KS effect-size floor (the "Unclassified" case).
    rng = np.random.default_rng(3)
    genes, families = _random_families(rng, 20000, [6000])
    values = pd.Series(rng.normal(size=20000), index=genes)
    values[families == "fam00"] += 0.08
    out = gl.family_permutation_gate(values, families, n_perm=2000).set_index("family")
    assert out.loc["fam00", "q_median"] < 0.05
    assert out.loc["fam00", "ks"] < 0.10
    assert not out.loc["fam00", "gate"]
