from logging import getLogger
from pathlib import Path

import cellxgene_census as cc
import cellxgene_census.experimental.ml as census_ml
import tiledbsoma as soma

from bmfm_targets.datasets.data_conversion.litdata_indexing import build_index

logger = getLogger()

from .cellxgene_splits import load_split_dataset_ids


def open_soma(uri, census_version="2023-12-15"):
    census_db = cc.open_soma(uri=uri, census_version=census_version)
    return census_db


def get_obs_as_pandas(
    uri,
    experiment: str,
    value_filter: str,
    census_version: str,
    n_records: int | None = None,
):
    census_db = open_soma(uri, census_version)
    experiment = census_db["census_data"][experiment]

    coords = None

    obs_query = soma.AxisQuery(value_filter=value_filter)
    query = experiment.axis_query(
        measurement_name="RNA",
        obs_query=obs_query,
    )
    coords = query.obs_joinids().to_numpy()
    if n_records:
        coords = coords[:n_records]

    filtered_obs = (
        experiment.obs.read(
            coords=(coords,),
        )
        .concat()
        .to_pandas()
    )
    return filtered_obs


def build_range_index(
    uri,
    index_dir,
    label_columns: list[str],
    n_records: int,
    experiment: str = "homo_sapiens",
    value_filter: str = "is_primary_data==True",
    census_version: str = "2023-12-15",
    chunk_size=5000,
):
    """
    Building index for debugging proposes.
    The function creates a single split from value_filter and cut # of items to n_records.
    """
    filtered_obs = get_obs_as_pandas(
        uri, experiment, value_filter, census_version, n_records
    )
    id_df = filtered_obs["soma_joinid"]
    index = id_df[:n_records].values.tolist()
    label_dict = get_label_dicts(
        uri, experiment, value_filter, census_version, label_columns
    )
    build_index(index_dir, index, label_dict, chunk_size=chunk_size)


def get_label_dicts(
    uri: str | Path | None,
    experiment: str,
    value_filter: str,
    census_version: str,
    label_columns: list[str],
):
    """
    Build label dict that used in SOMA datasets. NexusDB saves label dict in the index file.
    Note. `value_filter` could be different from filters that generate splits. However,
    split filter sets should be subsets of the filter set generated by get_label_dicts.
    """
    census_db = cc.open_soma(uri=uri, census_version=census_version)
    soma_experiment = census_db["census_data"][experiment]
    experiment_datapipe = census_ml.ExperimentDataPipe(
        soma_experiment,
        measurement_name="RNA",
        X_name="normalized",
        obs_query=soma.AxisQuery(value_filter=value_filter),
        obs_column_names=label_columns,
        batch_size=64,
        shuffle=False,
        return_sparse_X=False,  # sparse breaks multiprocessing
        soma_chunk_size=20_000,
    )
    label_encoders = {
        l: experiment_datapipe.obs_encoders[l].classes_ for l in label_columns
    }
    label_dict = {
        label_name: {l: i for i, l in enumerate(labels)}
        for label_name, labels in label_encoders.items()
    }
    return label_dict


def build_index_from_dataset_id(
    uri: str | Path | None = None,
    index_dir: str | Path = "cellxgene_nexus_index",
    census_version: str = "2023-12-15",
    experiment: str = "homo_sapiens",
    label_columns: list[str] = ["cell_type", "tissue"],
    label_dict_value_filter: str = "is_primary_data == True",
    chunk_size=5000,
):
    """
    Building index based on dataset_id splits from celltypes_split.csv.
    Function creates index in index_dir from the SOMA dataset at uri and save label_dict
    for columns in label_columns list. label_dict_value_filter is used to filter records
    when label_dict is built.

    Args:
    ----
    uri (str  |  Path, optional): Path to soma database. If `None`, will access the hosted
                version on AWS, which may be slow but can run from . Defaults to the downloaded copy
                on CCC at "/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/pretrain/cellxgene/soma-2023-12-15".
    """
    index_dir = Path(index_dir)
    split_file = Path(__file__).parent / "celltypes_split.csv"
    for split in ["train", "dev"]:
        split_index_dir = str(index_dir / split)
        split_dataset_ids = load_split_dataset_ids(split_file, split)
        value_filter = "is_primary_data==True"
        if split_dataset_ids:
            value_filter += f" and dataset_id in {split_dataset_ids}"
            filtered_obs = get_obs_as_pandas(
                uri, experiment, value_filter, census_version
            )
            id_df = filtered_obs["soma_joinid"]
            index = id_df.values.tolist()
            label_dict = get_label_dicts(
                uri,
                experiment,
                label_dict_value_filter,
                census_version,
                label_columns,
            )
            build_index(split_index_dir, index, label_dict, chunk_size=chunk_size)


def get_split_value_filter(split, custom_split_file=None):
    if custom_split_file is not None:
        split_file = custom_split_file
    else:
        split_file = Path(__file__).parent / "celltypes_split.csv"
    if split is not None:
        split_dataset_ids = load_split_dataset_ids(split_file, split)
        if split_dataset_ids:
            return f" and dataset_id in {split_dataset_ids}"
        else:
            logger.warning(
                f"Split {split} not found in file. Full dataset will be used."
            )
    return ""
