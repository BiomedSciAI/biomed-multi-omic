import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

from bmfm_targets.evaluation import model_comparison as mc


@pytest.fixture()
def scores():
    rng = np.random.default_rng(0)
    table = pd.DataFrame(rng.normal(size=(5, 20)), index=list("abcde"))
    table.loc["a"] += 2
    return table


def test_rank_models_known_case_with_tie():
    table = pd.DataFrame({"x": [0.9, 0.5, 0.5]}, index=["m1", "m2", "m3"])
    assert mc.rank_models(table)["x"].to_dict() == {"m1": 1.0, "m2": 2.5, "m3": 2.5}


def test_holm_matches_statsmodels():
    p = np.random.default_rng(1).uniform(size=30) ** 3
    np.testing.assert_allclose(mc.holm_adjust(p), multipletests(p, method="holm")[1])


def test_paired_rank_tests_match_scipy(scores):
    ranks = mc.rank_models(scores)
    result = mc.paired_rank_tests(ranks, ["a", "b"]).set_index(["reference", "other"])
    row = result.loc[("a", "c")]
    assert row.better == int((scores.loc["a"] > scores.loc["c"]).sum())
    assert row.better + row.worse + row.tied == scores.shape[1]
    assert row.p == stats.wilcoxon(ranks.loc["a"], ranks.loc["c"]).pvalue
    expected = multipletests(result.loc["a"].p, method="holm")[1]
    np.testing.assert_allclose(result.loc["a"].p_holm, expected)


def test_identical_models_are_not_different():
    table = pd.DataFrame(np.ones((3, 6)), index=list("abc"))
    assert (mc.paired_rank_tests(mc.rank_models(table), ["a"]).p_holm == 1).all()


def test_friedman_matches_scipy(scores):
    expected = stats.friedmanchisquare(*scores.to_numpy())
    assert mc.friedman_test(scores) == pytest.approx(tuple(expected))


def test_mean_zscore_matches_scipy(scores):
    expected = stats.zscore(scores.to_numpy(), axis=0, ddof=1).mean(axis=1)
    np.testing.assert_allclose(mc.mean_zscore(scores), expected)


def test_pareto_front_known_case():
    cost = pd.Series({"small": 1, "dominated": 2, "big": 3, "same_cost": 3})
    score = pd.Series({"small": 0.5, "dominated": 0.4, "big": 0.9, "same_cost": 0.8})
    assert mc.pareto_front(cost, score) == ["small", "big"]


def test_missing_score_is_an_error(scores):
    scores.iloc[1, 3] = np.nan
    with pytest.raises(ValueError, match="'b'"):
        mc.rank_models(scores)
