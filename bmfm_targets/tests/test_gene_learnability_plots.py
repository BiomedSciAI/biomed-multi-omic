import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from bmfm_targets.evaluation import gene_learnability_plots as glp
from bmfm_targets.evaluation.gene_learnability import gene_skill_table
from bmfm_targets.evaluation.predictions import constant_prediction_macro_f1


@pytest.fixture()
def skills():
    rng = np.random.default_rng(0)
    n = 400
    expr = np.exp(rng.normal(size=n))
    err = expr * np.exp(rng.normal(0, 0.2, n))
    table = pd.DataFrame(
        {
            "gene_avg_nz_expressions": expr,
            "gene_err_nz_by_logits": err,
            "gene_err_nz_null": err * 2,
            "gene_nz_freq": 100,
            "is_zero_roc_auc": rng.uniform(0.6, 0.9, n),
        },
        index=[f"g{i}" for i in range(n)],
    )
    return gene_skill_table(table)


@pytest.fixture(autouse=True)
def _close_figures():
    plt.figure()
    yield
    plt.close("all")


def test_error_band_lies_on_the_axes(skills):
    ax = glp.plot_error_vs_expression(skills)
    trend, low, high = (line.get_ydata() for line in ax.lines)
    assert (low < trend).all()
    assert (trend < high).all()
    assert len(ax.collections[0].get_offsets()) == len(skills)


def test_detection_vs_quantification_rho(skills):
    skills["det_auc"] = -skills["q_skill"] ** 3
    assert glp.plot_detection_vs_quantification(skills)[1] == pytest.approx(-1)


def test_family_intervals_exclude_zero_only_when_significant():
    gate = pd.DataFrame(
        {
            "family": ["edge", "strong", "null"],
            "n": [10, 20, 30],
            "delta_median": [0.5, -1.0, 0.1],
            "p_median": [0.05, 0.001, 0.8],
        }
    )
    table = glp.family_intervals(gate)
    assert table.loc["edge", "low"] == pytest.approx(0)
    assert table.loc["edge", "high"] == pytest.approx(1)
    assert table.loc["strong", "high"] < 0
    assert table.loc["null", "low"] < 0 < table.loc["null", "high"]
    ax = glp.plot_family_intervals(table, table, families=["edge", "strong"])
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == ["strong (n=20)", "edge (n=10)"]


def test_trajectories_and_ranks_known_case():
    families = pd.Series({"g1": "A", "g2": "A", "g3": "B"})
    tables = {
        2: pd.DataFrame({"err": [1.0, 3.0, 5.0]}, index=["g1", "g2", "g3"]),
        1: pd.DataFrame({"err": [9.0, 9.0, 4.0]}, index=["g1", "g2", "g3"]),
    }
    traj = glp.family_trajectories(tables, families, "err")
    assert traj.index.tolist() == [1, 2]
    assert traj["A"].tolist() == [9, 2]
    assert traj["B"].tolist() == [4, 5]
    assert glp.rank_trajectories(traj, False)["A"].tolist() == [2, 1]
    assert glp.rank_trajectories(traj, True)["A"].tolist() == [1, 2]
    ax = glp.plot_family_trajectories(traj, ["A"])
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["A"]


def test_model_agreement_uses_shared_genes():
    a = pd.Series({"g1": 1.0, "g2": 2.0, "g3": 3.0, "only": 9.0}, name="a")
    b = pd.Series({"g1": 10.0, "g2": 20.0, "g3": 30.0}, name="b")
    ax, rho = glp.plot_model_agreement(a, b)
    assert rho == pytest.approx(1)
    assert len(ax.collections[0].get_offsets()) == 3


def test_constant_prediction_f1_matches_sklearn():
    labels = np.random.default_rng(1).choice(list("abcd"), 200, p=[0.5, 0.3, 0.1, 0.1])
    expected = f1_score(labels, np.full(200, "a"), average="macro")
    assert constant_prediction_macro_f1(labels) == pytest.approx(expected)
