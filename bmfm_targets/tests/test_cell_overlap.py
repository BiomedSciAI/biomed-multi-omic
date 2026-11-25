import pandas as pd
import pytest

from bmfm_targets.datasets.cellxgene import cell_overlap


def test_summarize_overlap_known_case():
    obs = {
        "half": pd.DataFrame({"observation_joinid": ["a", "b"]}),
        "none": pd.DataFrame({"observation_joinid": ["x", "y", "z"]}),
    }
    table = cell_overlap.summarize_overlap(obs, {"b", "c"})
    assert table["n_overlap"].to_dict() == {"half": 1, "none": 0}
    assert table["frac_overlap"].to_dict() == {"half": 0.5, "none": 0.0}


def test_missing_ids_are_errors():
    with pytest.raises(KeyError):
        cell_overlap.summarize_overlap({"d": pd.DataFrame({"x": [1]})}, set())
    with pytest.raises(ValueError):
        cell_overlap.summarize_overlap(
            {"d": pd.DataFrame({"observation_joinid": ["a", None]})}, {"a"}
        )


def test_split_query_uses_split_and_assay_filter(tmp_path, monkeypatch):
    split_file = tmp_path / "split.csv"
    pd.DataFrame(
        {"dataset_id": ["d1", "d2", "d3"], "split": ["train", "train", "dev"]}
    ).to_csv(split_file)
    seen = {}

    def fake_obs(**kwargs):
        seen.update(kwargs)
        return pd.DataFrame({"observation_joinid": ["a", "b", "a"]})

    monkeypatch.setattr(cell_overlap, "get_obs_as_pandas", fake_obs)
    ids = cell_overlap.get_split_observation_joinids(split_file, "2025-01-30")
    assert ids == {"a", "b"}
    assert seen["value_filter"].endswith("dataset_id in ['d1', 'd2']")
    assert "10x 3' v3" in seen["value_filter"]
    with pytest.raises(ValueError, match="d3"):
        cell_overlap.get_split_observation_joinids(
            split_file, "2025-01-30", dataset_ids=["d3"]
        )
