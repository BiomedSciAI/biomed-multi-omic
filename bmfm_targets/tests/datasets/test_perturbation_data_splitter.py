import pandas as pd

from bmfm_targets.datasets.datasets_utils import get_perturbation_split_column


def test_single_only_split_perturbations(single_perts_df):
    tol = 0.05
    split_weights = {"train": 0.8, "dev": 0.1, "test": 0.1}
    perturbation_column_name = "perturbation"
    splits = get_perturbation_split_column(
        metadata_df=single_perts_df,
        split_weights=split_weights,
        perturbation_column_name=perturbation_column_name,
        stratification_type="single_only",
    )
    splits.name = "split_column"
    single_perts_df_with_splits = pd.concat([single_perts_df, splits], axis=1)
    perturbations_by_split = (
        single_perts_df_with_splits.groupby("split_column")["perturbation"]
        .unique()
        .to_dict()
    )
    assert len(perturbations_by_split["train"]) == 9
    assert "Control" in perturbations_by_split["train"]
    assert len(perturbations_by_split["dev"]) == 1
    assert len(perturbations_by_split["test"]) == 1


def test_single_split_perturbations(single_and_combo_perts_df):
    split_weights = {"train": 0.8, "dev": 0.1, "test": 0.1}
    perturbation_column_name = "perturbation"
    splits = get_perturbation_split_column(
        metadata_df=single_and_combo_perts_df,
        split_weights=split_weights,
        perturbation_column_name=perturbation_column_name,
        stratification_type="single",
    )

    metadata_df = single_and_combo_perts_df.assign(split=splits)
    train_perts = metadata_df[metadata_df["split"] == "train"]["perturbation"]
    dev_perts = metadata_df[metadata_df["split"] == "dev"]["perturbation"]
    test_perts = metadata_df[metadata_df["split"] == "test"]["perturbation"]
    assert any("_" in pert for pert in train_perts)
    assert any("_" not in pert for pert in train_perts)
    assert len(dev_perts) > 0
    assert all("_" not in pert for pert in dev_perts)
    assert len(test_perts) > 0
    assert all("_" not in pert for pert in test_perts)


def test_combo_split_perturbations_seen0(single_and_combo_perts_df):
    split_weights = {"train": 0.5, "dev": 0.25, "test": 0.25}
    perturbation_column_name = "perturbation"
    splits = get_perturbation_split_column(
        metadata_df=single_and_combo_perts_df,
        split_weights=split_weights,
        perturbation_column_name=perturbation_column_name,
        stratification_type="combo_seen0",
    )

    metadata_df = single_and_combo_perts_df.assign(split=splits)
    train_perts = metadata_df[metadata_df["split"] == "train"]["perturbation"]
    dev_perts = metadata_df[metadata_df["split"] == "dev"]["perturbation"]
    test_perts = metadata_df[metadata_df["split"] == "test"]["perturbation"]

    assert len(train_perts) > 0
    assert len(dev_perts) > 0
    assert len(test_perts) > 0

    for dev_pert in dev_perts:
        if "_" in dev_pert:
            pert_list = dev_pert.split("_")
            assert all(pert not in train_perts for pert in pert_list) == True

    for test_pert in test_perts:
        if "_" in test_pert:
            pert_list = test_pert.split("_")
            assert all(pert not in train_perts for pert in pert_list) == True


def test_combo_split_perturbations_seen1(single_and_combo_perts_df):
    split_weights = {"train": 0.8, "dev": 0.1, "test": 0.1}
    perturbation_column_name = "perturbation"
    splits = get_perturbation_split_column(
        metadata_df=single_and_combo_perts_df,
        split_weights=split_weights,
        perturbation_column_name=perturbation_column_name,
        stratification_type="combo_seen1",
    )
    metadata_df = single_and_combo_perts_df.assign(split=splits)
    train_perts = metadata_df[metadata_df["split"] == "train"]["perturbation"]
    dev_perts = metadata_df[metadata_df["split"] == "dev"]["perturbation"]
    test_perts = metadata_df[metadata_df["split"] == "test"]["perturbation"]

    assert len(train_perts) > 0
    assert any("_" in pert for pert in train_perts)
    assert any("_" not in pert for pert in train_perts)
    assert len(dev_perts) > 0
    assert len(test_perts) > 0
    assert len(set(train_perts).intersection(set(test_perts))) == 0
    assert len(set(train_perts).intersection(set(dev_perts))) == 0
    train_perts = list(train_perts)
    for dev_pert in dev_perts:
        if "_" in dev_pert:
            pert_list = dev_pert.split("_")
            assert any(pert in train_perts for pert in pert_list) == True

    for test_pert in test_perts:
        if "_" in test_pert:
            pert_list = test_pert.split("_")
            assert any(pert in train_perts for pert in pert_list) == True


def test_combo_split_perturbations_seen2(single_and_combo_perts_df):
    split_weights = {"train": 0.8, "dev": 0.1, "test": 0.1}
    perturbation_column_name = "perturbation"
    splits = get_perturbation_split_column(
        metadata_df=single_and_combo_perts_df,
        split_weights=split_weights,
        perturbation_column_name=perturbation_column_name,
        stratification_type="combo_seen2",
    )
    metadata_df = single_and_combo_perts_df.assign(split=splits)
    train_perts = metadata_df[metadata_df["split"] == "train"]["perturbation"]
    dev_perts = metadata_df[metadata_df["split"] == "dev"]["perturbation"]
    test_perts = metadata_df[metadata_df["split"] == "test"]["perturbation"]

    assert len(train_perts) > 0
    assert any("_" in pert for pert in train_perts)
    assert any("_" not in pert for pert in train_perts)
    assert len(dev_perts) > 0
    assert len(test_perts) > 0

    for dev_pert in dev_perts:
        if "_" in dev_pert:
            for pert in dev_pert.split("_"):
                assert pert in list(train_perts)

    for test_pert in test_perts:
        if "_" in test_pert:
            for pert in test_pert.split("_"):
                assert pert in list(train_perts)


def test_simulation_split_perturbations(single_and_combo_perts_df):
    split_weights = {"train": 0.8, "dev": 0.1, "test": 0.1}
    perturbation_column_name = "perturbation"
    splits, groups = get_perturbation_split_column(
        metadata_df=single_and_combo_perts_df,
        split_weights=split_weights,
        perturbation_column_name=perturbation_column_name,
        stratification_type="simulation",
    )
    # append splits to the metadata
    metadata_df = single_and_combo_perts_df.assign(split=splits)
    metadata_df = metadata_df.assign(group=groups)
    train_perts = metadata_df[metadata_df["split"] == "train"]["perturbation"]
    dev_perts = metadata_df[metadata_df["split"] == "dev"]["perturbation"]
    test_perts = metadata_df[metadata_df["split"] == "test"]["perturbation"]

    assert len(train_perts) > 0
    assert len(dev_perts) > 0
    assert len(test_perts) > 0
    assert len(set(train_perts).intersection(set(test_perts))) == 0
    assert len(set(train_perts).intersection(set(dev_perts))) == 0
    assert metadata_df["group"].nunique() >= 4
