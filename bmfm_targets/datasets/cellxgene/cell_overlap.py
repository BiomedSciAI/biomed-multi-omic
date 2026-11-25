"""Find evaluation cells that are in the CELLxGENE data a model was pretrained on."""

from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

from bmfm_targets.datasets.cellxgene.cellxgene_soma_utils import (
    get_obs_as_pandas,
    get_value_filter,
)
from bmfm_targets.datasets.cellxgene.cellxgene_splits import load_split_dataset_ids


def get_split_observation_joinids(
    split_file: str | Path,
    census_version: str,
    split: str = "train",
    dataset_ids: Iterable[str] | None = None,
    filter_criteria: str | None = "scTab",
    uri: str | None = None,
    experiment: str = "homo_sapiens",
) -> set[str]:
    """
    `observation_joinid` of the cells eligible for a pretraining split.

    Applies the filters that build the litdata index of the split: the datasets of the
    split file and the assay criteria.

    Args:
    ----
        split_file: csv with `dataset_id` and `split` columns
        census_version: CELLxGENE census version
        split: split name
        dataset_ids: restrict to these datasets, all of which must be in the split
        filter_criteria: criteria passed to `get_value_filter`
        uri: census uri; None uses the public census
        experiment: census experiment
    """
    split_ids = set(load_split_dataset_ids(split_file, split=split))
    selected = split_ids if dataset_ids is None else set(dataset_ids)
    if outside := selected - split_ids:
        raise ValueError(f"Datasets not in split '{split}': {sorted(outside)}")
    obs = get_obs_as_pandas(
        uri=uri,
        experiment=experiment,
        value_filter=(
            f"{get_value_filter(filter_criteria)} and dataset_id in {sorted(selected)}"
        ),
        census_version=census_version,
    )
    return set(obs["observation_joinid"])


def summarize_overlap(
    evaluation_obs: Mapping[str, pd.DataFrame], pretraining_joinids: set[str]
) -> pd.DataFrame:
    """
    Count the cells of each evaluation dataset that are in the pretraining set.

    Cells are matched by `observation_joinid`, which CELLxGENE keeps across exports.

    Returns
    -------
        One row per dataset: `n_cells`, `n_overlap`, `frac_overlap`.
    """
    rows = {}
    for name, obs in evaluation_obs.items():
        if "observation_joinid" not in obs:
            raise KeyError(f"{name} has no observation_joinid column")
        joinids = obs["observation_joinid"]
        if joinids.isna().any():
            raise ValueError(f"{name} has cells without an observation_joinid")
        overlap = int(joinids.isin(pretraining_joinids).sum())
        rows[name] = {
            "n_cells": len(obs),
            "n_overlap": overlap,
            "frac_overlap": overlap / len(obs) if len(obs) else float("nan"),
        }
    return pd.DataFrame.from_dict(rows, orient="index")
