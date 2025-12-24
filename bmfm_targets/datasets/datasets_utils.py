import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse as ss
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def get_split_column(
    metadata_df: pd.DataFrame,
    split_weights: dict[str, float],
    stratifying_label: str | None = None,
    random_state: int = 42,
) -> pd.Series:
    """
    Create a column with splits for metadata_df with optional balancing.

    Args:
    ----
        metadata_df (pd.DataFrame): metadata file with element ids and labels
        split_weights (dict): dictionary of weights
        stratifying_label (str, None): label to use for balancing. Must be present in metadata_df
        random_state (int, optional): Random state for shuffling the data. Defaults to 42. Defaults to 42.

    Returns:
    -------
        dict[str, pd.Series]: Dictionary containing the splits as keys and corresponding dataframes as values.

    """
    if stratifying_label is None:
        return get_random_split(metadata_df, split_weights, random_state)

    return get_stratified_split(
        metadata_df, split_weights, stratifying_label, random_state
    )


def get_random_split(
    metadata_df: pd.DataFrame, split_weights: dict[str, float], random_state: int
) -> pd.Series:
    choices = ["train", "dev", "test"]
    p = [split_weights[i] for i in choices]
    rng = np.random.default_rng(random_state)
    splits_array = rng.choice(choices, size=len(metadata_df), p=p)
    return metadata_df.assign(split=splits_array).pop("split")


def get_stratified_split(
    metadata_df: pd.DataFrame,
    split_weights: dict[str, float],
    stratifying_label: str,
    random_state: int,
) -> pd.Series:
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=split_weights["test"], random_state=random_state
    )
    train_dev_idx, test_idx = next(
        sss.split(X=metadata_df, y=metadata_df[stratifying_label])
    )
    train_dev_df = metadata_df.iloc[train_dev_idx]
    train_dev_label = metadata_df.iloc[train_dev_idx][stratifying_label]
    dev_frac = split_weights["dev"] / (split_weights["train"] + split_weights["dev"])
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=dev_frac, random_state=random_state
    )

    train_idx, dev_idx = next(sss.split(X=train_dev_df, y=train_dev_label))

    metadata_df["split"] = "test"
    split_column = metadata_df.pop("split")
    split_column.loc[train_dev_df.iloc[train_idx].index] = "train"
    split_column.loc[train_dev_df.iloc[dev_idx].index] = "dev"

    return split_column


def load_extra_metadata_mapping(dataset_name, mapping_key, mapping_value):
    """
    Load metadata_extra_mapping.csv from the given dataset metadata folder,
    and return the values of a requested key and value columns as a dictionary.
    """
    extra_metadata_mapping_file_path = (
        Path(__file__).parent / dataset_name / "metadata_extra_mapping.csv"
    )
    if not os.path.exists(extra_metadata_mapping_file_path):
        raise FileNotFoundError(str(extra_metadata_mapping_file_path) + "is not found")
    else:
        mapping_df = pd.read_csv(extra_metadata_mapping_file_path, index_col=False)
        extra_metadata_mapping = dict(
            zip(
                mapping_df[mapping_key],
                mapping_df[mapping_value],
            )
        )
        return extra_metadata_mapping


def create_celltype_labels_json_file(adata, output_path, label_column_name):
    obs_tab = adata.obs
    unique_values = sorted(obs_tab[label_column_name].unique())
    value_to_number = {value: idx for idx, value in enumerate(unique_values)}

    with open(output_path, "w") as json_file:
        json.dump(value_to_number, json_file, indent=4)


def load_and_update_all_labels_json(adata, output_path, label_column_name):
    if output_path is not None and Path(output_path).exists():
        with open(output_path) as json_file:
            all_labels_dict = json.load(json_file)
            for item in all_labels_dict:
                if label_column_name == item["label_name"]:
                    return item["label_values"]
    else:
        all_labels_dict = []

    obs_tab = adata.obs if hasattr(adata, "obs") else adata
    unique_values = sorted(obs_tab[label_column_name].dropna().unique())
    value_to_number = {str(value): idx for idx, value in enumerate(unique_values)}

    new_label_dict = {"label_name": label_column_name, "label_values": value_to_number}
    all_labels_dict.append(new_label_dict)
    if output_path is not None:
        with open(output_path, "w") as json_file:
            json.dump(all_labels_dict, json_file, indent=4)

    return value_to_number


def obs_key_wise_subsampling(adata, obs_key, N):
    """
    Subsample each class to same cell numbers (N). Classes are given by obs_key pointing to categorical in adata.obs.

    From https://github.com/scverse/scanpy/issues/987
    """
    counts = adata.obs[obs_key].value_counts()
    # subsample indices per group defined by obs_key
    indices = [
        np.random.choice(
            adata.obs_names[adata.obs[obs_key] == group], size=N, replace=False
        )
        for group in counts.index
        if len(adata.obs_names[adata.obs[obs_key] == group]) >= N
    ]
    selection = np.hstack(np.array(indices))
    return adata[selection].copy()


def random_subsampling(adata, n_samples, shuffle):
    """
    Randomly subsample a dataset to n samples.

    Args:
    ----
    adata (AnnData) : the dataset
    n_samples(int|None) : number of samples to return
    shuffle (bool): wether to shuffle while sampling, if False will return first n samples
                    otherwise a random sub sample of size n.
    """
    if not shuffle:
        data = adata[: min(len(adata), n_samples)]
    else:
        samples = np.random.choice(
            len(adata), size=min(len(adata), n_samples), replace=False
        )
        data = adata[samples]
    data = data.copy()
    return data


def _single_only_split_perturbations(unique_perturbations, test_size=0.1):
    """
    Splits a given single perturbation list perturbation list into train and test with no shared perturbations.

    Returns
    -------
        list: Train perturbations.
        list: Test perturbations.
    """
    single_perturbations = [pert for pert in unique_perturbations if "_" not in pert]
    test_count = int(len(single_perturbations) * test_size)
    if test_count == 0:
        logger.warning("Test size is too small. Setting test size to 1")
        test_count = 1
    single_test_perturbations = random.sample(single_perturbations, test_count)
    single_train_perturbations = list(
        set(single_perturbations) - set(single_test_perturbations)
    )
    return single_train_perturbations, single_test_perturbations


def _single_split_perturbations(unique_perturbations, test_size=0.1):
    """
    Splits a given single perturbation list into train and test with no shared perturbations.
    Add combo perturbations that do not contain any test genes to the train set.

    Returns
    -------
        list: Train perturbations.
        list: Test perturbations.
    """
    (
        single_train_perturbations,
        single_test_perturbations,
    ) = _single_only_split_perturbations(unique_perturbations, test_size)
    combo_perturbations = [pert for pert in unique_perturbations if "_" in pert]
    train_combo_perturbations = [
        p
        for p in combo_perturbations
        if not any(g in single_test_perturbations for g in p.split("_"))
    ]

    train_perturbations = single_train_perturbations + train_combo_perturbations
    return train_perturbations, single_test_perturbations


def _combo_split_perturbations(unique_perturbations, split_weights, seen=0):
    """
    Splits single and combo perturbations into train, dev and test sets.

    if seen = 2, model has seen each of the two genes in the combination individually experimentally perturbed in the training data
    if seen = 1, model has seen at least one of the two genes in the combination individually experimentally perturbed in the training data
    if seen = 0, model has not seen any of the two genes in the combination individually experimentally perturbed in the training data

    Returns
    -------
        list: Train perturbations.
        list: Test perturbations.

    """
    (
        single_train_perturbations,
        single_test_perturbations,
    ) = _single_only_split_perturbations(unique_perturbations, split_weights["test"])

    (
        single_train_perturbations,
        single_dev_perturbations,
    ) = _single_only_split_perturbations(
        single_train_perturbations, split_weights["dev"]
    )
    combo_perturbations = list({pert for pert in unique_perturbations if "_" in pert})
    combo_train_perturbations = []
    combo_dev_perturbations = []
    combo_test_perturbations = []

    if seen == 0:
        combo_perturbations_for_dev_test = [
            t
            for t in combo_perturbations
            if len([p for p in t.split("_") if p in single_train_perturbations]) == 0
        ]
    elif seen == 1:
        combo_perturbations_for_dev_test = [
            t
            for t in combo_perturbations
            if len([p for p in t.split("_") if p in single_train_perturbations]) == 1
        ]

    else:
        combo_perturbations_for_dev_test = [
            t
            for t in combo_perturbations
            if len([p for p in t.split("_") if p in single_train_perturbations]) == 2
        ]
    if len(combo_perturbations_for_dev_test) != 0:
        test_count = int(len(combo_perturbations_for_dev_test) * split_weights["test"])
        if test_count == 0:
            test_count = 1
        combo_test_perturbations = random.sample(
            combo_perturbations_for_dev_test, test_count
        )
    combo_perturbations_for_dev = list(
        set(combo_perturbations_for_dev_test) - set(combo_test_perturbations)
    )

    if len(combo_perturbations_for_dev) != 0:
        dev_count = int(len(combo_perturbations_for_dev) * split_weights["dev"])
        if dev_count == 0:
            dev_count = 1
        combo_dev_perturbations = random.sample(combo_perturbations_for_dev, dev_count)

    combo_train_perturbations = [
        p
        for p in combo_perturbations
        if (p not in combo_test_perturbations) and (p not in combo_dev_perturbations)
    ]
    """
    combo_train_perturbations = [
        t
        for t in combo_train_perturbations
        if len([p for p in t.split("_") if p in single_test_perturbations]) == 0
    ]
    """
    train_perturbations = single_train_perturbations + combo_train_perturbations
    dev_perturbations = single_dev_perturbations + combo_dev_perturbations
    test_perturbations = single_test_perturbations + combo_test_perturbations
    return train_perturbations, dev_perturbations, test_perturbations


def _simulation_split_perturbations(
    unique_perturbations, split_weights, combo_seen2_train_frac=0.85
):
    """
    Splits the data into train, dev, test sets based on the split type.

    Returns
    -------
        list: Train perturbations.
        list: Test perturbations.
        dict: Dictionary of subgroups.
    """
    (
        single_train_perturbations,
        single_test_perturbations,
    ) = _single_only_split_perturbations(unique_perturbations, split_weights["test"])

    (
        single_train_perturbations,
        single_dev_perturbations,
    ) = _single_only_split_perturbations(
        single_train_perturbations, split_weights["dev"]
    )

    combo_perturbations = list({pert for pert in unique_perturbations if "_" in pert})

    combo_perturbations_seen0 = [
        t
        for t in combo_perturbations
        if len([p for p in t.split("_") if p in single_train_perturbations]) == 0
    ]

    combo_perturbations_seen1 = [
        t
        for t in combo_perturbations
        if len([p for p in t.split("_") if p in single_train_perturbations]) == 1
    ]
    combo_perturbations_seen2 = [
        t
        for t in combo_perturbations
        if len([p for p in t.split("_") if p in single_train_perturbations]) == 2
    ]

    combo_train_perturbations_seen2 = random.sample(
        combo_perturbations_seen2,
        int(len(combo_perturbations_seen2) * combo_seen2_train_frac),
    )

    combo_dev_test_perturbations_seen2 = list(
        set(combo_perturbations_seen2) - set(combo_train_perturbations_seen2)
    )

    train_perturbations = single_train_perturbations + combo_train_perturbations_seen2

    # split combo_perturbations_seen0 into dev and test
    combo_perturbations_seen0_test = random.sample(
        combo_perturbations_seen0,
        int(len(combo_perturbations_seen0) * 0.5),
    )

    combo_perturbations_seen0_dev = list(
        set(combo_perturbations_seen0) - set(combo_perturbations_seen0_test)
    )

    # split combo_perturbations_seen1 into dev and test
    combo_perturbations_seen1_test = random.sample(
        combo_perturbations_seen1,
        int(len(combo_perturbations_seen1) * 0.5),
    )
    combo_perturbations_seen1_dev = list(
        set(combo_perturbations_seen1) - set(combo_perturbations_seen1_test)
    )

    # split combo_dev_test_perturbations_seen2 into dev and test
    combo_perturbations_seen2_test = random.sample(
        combo_dev_test_perturbations_seen2,
        int(len(combo_dev_test_perturbations_seen2) * 0.5),
    )
    combo_perturbations_seen2_dev = list(
        set(combo_dev_test_perturbations_seen2) - set(combo_perturbations_seen2_test)
    )

    dev_perturbations = (
        single_dev_perturbations
        + combo_perturbations_seen0_dev
        + combo_perturbations_seen1_dev
        + combo_perturbations_seen2_dev
    )

    test_perturbations = (
        single_test_perturbations
        + combo_perturbations_seen0_test
        + combo_perturbations_seen1_test
        + combo_perturbations_seen2_test
    )

    return (
        train_perturbations,
        dev_perturbations,
        test_perturbations,
        {
            "combo_seen0_dev": combo_perturbations_seen0_dev,
            "combo_seen1_dev": combo_perturbations_seen1_dev,
            "combo_seen2_dev": combo_perturbations_seen2_dev,
            "unseen_single_dev": single_dev_perturbations,
        },
        {
            "combo_seen0_test": combo_perturbations_seen0_test,
            "combo_seen1_test": combo_perturbations_seen1_test,
            "combo_seen2_test": combo_perturbations_seen2_test,
            "unseen_single_test": single_test_perturbations,
        },
    )


def get_perturbation_split_column(
    metadata_df: pd.DataFrame,
    split_weights: dict[str, float],
    perturbation_column_name: str,
    random_state: int = 42,
    stratification_type: str = "single_only",
    combo_seen2_train_frac=0.85,
) -> pd.Series | list[pd.Series]:
    """
    Splits the data into train, dev, test sets based on the stratifying label.

    Returns
    -------
        pd.Series | list[pd.Series]: The split column for the data. If simulation split, returns a list of split and group columns.
    """
    unique_perturbations = list(set(metadata_df[perturbation_column_name].values))
    unique_perturbations.remove("Control")

    if stratification_type == "single_only":
        (
            train_perturbations,
            test_perturbations,
        ) = _single_only_split_perturbations(
            unique_perturbations, split_weights["test"]
        )
        (
            train_perturbations,
            dev_perturbations,
        ) = _single_only_split_perturbations(train_perturbations, split_weights["dev"])

    elif stratification_type == "single":
        train_perturbations, test_perturbations = _single_split_perturbations(
            unique_perturbations, split_weights["test"]
        )
        train_perturbations, dev_perturbations = _single_split_perturbations(
            train_perturbations, split_weights["dev"]
        )

    elif stratification_type == "combo_seen0":
        (
            train_perturbations,
            dev_perturbations,
            test_perturbations,
        ) = _combo_split_perturbations(unique_perturbations, split_weights, seen=0)

    elif stratification_type == "combo_seen1":
        (
            train_perturbations,
            dev_perturbations,
            test_perturbations,
        ) = _combo_split_perturbations(unique_perturbations, split_weights, seen=1)

    elif stratification_type == "combo_seen2":
        (
            train_perturbations,
            dev_perturbations,
            test_perturbations,
        ) = _combo_split_perturbations(unique_perturbations, split_weights, seen=2)

    elif stratification_type == "simulation":
        (
            train_perturbations,
            dev_perturbations,
            test_perturbations,
            dev_subgroups,
            test_subgroups,
        ) = _simulation_split_perturbations(
            unique_perturbations, split_weights, combo_seen2_train_frac
        )

        metadata_df["group"] = None
        group_column = metadata_df.pop("group")

        for subgroup, perturbations in test_subgroups.items():
            group_column.loc[
                metadata_df[
                    metadata_df[perturbation_column_name].isin(perturbations)
                ].index
            ] = subgroup
        for subgroup, perturbations in dev_subgroups.items():
            group_column.loc[
                metadata_df[
                    metadata_df[perturbation_column_name].isin(perturbations)
                ].index
            ] = subgroup

    metadata_df["split"] = ""
    split_column = metadata_df.pop("split")

    map_dict = {x: "train" for x in train_perturbations}
    map_dict.update({x: "dev" for x in dev_perturbations})
    map_dict.update({x: "test" for x in test_perturbations})
    map_dict.update({"Control": "train"})
    split_column = metadata_df[perturbation_column_name].map(map_dict)
    if stratification_type == "simulation":
        return [split_column, group_column]
    else:
        return split_column


def add_gene_ranking(
    ad,
    perturbation_column_name="perturbation",
    control_name="Control",
    n_genes=50,
    key_added="rank_genes_groups_cov_all",
):
    sc.tl.rank_genes_groups(
        ad,
        groupby=perturbation_column_name,
        reference=control_name,
        rankby_abs=True,
        n_genes=n_genes,
        use_raw=False,
    )
    de_genes = pd.DataFrame(ad.uns["rank_genes_groups"]["names"])
    gene_dict = {group: de_genes[group].tolist() for group in de_genes}

    ad.uns[key_added] = gene_dict


def add_non_zero_non_dropout_de_gene_ranking(
    adata, perturbation_column_name="perturbation", control_label="Control"
):
    """
    Based on GEARS
    See original implementation at
    https://github.com/snap-stanford/GEARS/blob/6575a8f4e780814badbcff75a9dfd53b91e47218/gears/data_utils.py#L72
    MIT License.
    """
    # calculate mean expression for each perturbation
    unique_perturbations = adata.obs[perturbation_column_name].unique()
    perturbations2index = {}
    for i in unique_perturbations:
        perturbations2index[i] = np.where(adata.obs[perturbation_column_name] == i)[0]

    perturbation2mean_expression = {}
    for i, j in perturbations2index.items():
        perturbation2mean_expression[i] = np.mean(adata.X[j], axis=0)
    pert_list = np.array(list(perturbation2mean_expression.keys()))
    mean_expression = np.array(list(perturbation2mean_expression.values())).reshape(
        len(adata.obs[perturbation_column_name].unique()), adata.X.toarray().shape[1]
    )
    ctrl = mean_expression[np.where(pert_list == control_label)[0]]

    gene_id2idx = dict(zip(adata.var.index.values, range(len(adata.var))))
    gene_idx2id = dict(zip(range(len(adata.var)), adata.var.index.values))

    non_zeros_gene_idx = {}
    top_non_dropout_de_20 = {}
    top_non_zero_de_20 = {}
    non_dropout_gene_idx = {}

    for pert in adata.uns["rank_genes_groups_cov_all"].keys():
        X = np.mean(adata[adata.obs[perturbation_column_name] == pert].X, axis=0)

        non_zero = np.where(np.array(X)[0] != 0)[0]
        zero = np.where(np.array(X)[0] == 0)[0]
        true_zeros = np.intersect1d(zero, np.where(np.array(ctrl)[0] == 0)[0])
        non_dropouts = np.concatenate((non_zero, true_zeros))

        top = adata.uns["rank_genes_groups_cov_all"][pert]
        gene_idx_top = [gene_id2idx[i] for i in top]

        non_dropout_20 = [i for i in gene_idx_top if i in non_dropouts][:20]
        non_dropout_20_gene_id = [gene_idx2id[i] for i in non_dropout_20]

        non_zero_20 = [i for i in gene_idx_top if i in non_zero][:20]
        non_zero_20_gene_id = [gene_idx2id[i] for i in non_zero_20]

        non_zeros_gene_idx[pert] = np.sort(non_zero).tolist()
        non_dropout_gene_idx[pert] = np.sort(non_dropouts).tolist()
        top_non_dropout_de_20[pert] = non_dropout_20_gene_id
        top_non_zero_de_20[pert] = non_zero_20_gene_id

    non_zero = np.where(np.array(X)[0] != 0)[0]
    zero = np.where(np.array(X)[0] == 0)[0]
    true_zeros = np.intersect1d(zero, np.where(np.array(ctrl)[0] == 0)[0])
    non_dropouts = np.concatenate((non_zero, true_zeros))

    adata.uns["top_non_dropout_de_20"] = top_non_dropout_de_20
    adata.uns["non_dropout_gene_idx"] = non_dropout_gene_idx
    adata.uns["non_zeros_gene_idx"] = non_zeros_gene_idx
    adata.uns["top_non_zero_de_20"] = top_non_zero_de_20

    return adata


def equal_samples_per_set_downsample(
    df: pd.DataFrame,
    groupby_columns: str | list[str],
    frac: float,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Downsample a DataFrame to exactly (frac * total samples) while fairly distributing
    samples across groups. Small groups keep all their samples; larger groups share
    the remaining slots equally.

    Guarantees:
    1. Total samples = int(len(df) * frac) (exact match)
    2. No group exceeds its original size
    3. Groups contribute as equally as possible

    Args:
    ----
        df: Input DataFrame to downsample
        groupby_columns: Column name(s) defining groups (e.g., ["category"])
        frac: Fraction of total samples to keep (e.g., 0.7 for 70%)
        random_state: Optional random seed for reproducibility

    Returns:
    -------
        Downsampled DataFrame with these properties:
        - Small groups: Keep all their samples (if needed to reach total)
        - Large groups: Equal samples after small groups are preserved

    Example:
    -------
        Input: {"A":50, "B":150, "C":100}, frac=0.5 so target_size=200
        Output: {"A":50, "B":75, "C":75} (exactly 200 samples)
                (A keeps all, B/C split remaining 150 equally)
    """
    total_size = int(len(df) * frac)
    logger.info(f"Downsampling {df.shape[0]} to {total_size}")
    grouped = df.groupby(groupby_columns, observed=True)
    group_sizes = grouped.size()
    groups = group_sizes.index.tolist()
    avail = group_sizes.values
    num_groups = len(avail)
    logger.info(
        f"Attempting to construct equal sample sizes for {num_groups} combination of the following columns: {groupby_columns}"
    )

    # Step 1: Initial assignment (capped at group size)
    base_quota = total_size // num_groups
    assignment = np.minimum(base_quota, avail)
    remaining_quota = total_size - assignment.sum()
    logger.info(
        f"Initial sample size of {base_quota} leaves {remaining_quota} remaining samples due to insufficient samples in some sets."
    )

    # Step 2: Fair redistribution with strict size enforcement
    while remaining_quota > 0:
        # Find groups that can take more samples
        remaining_capacity = avail - assignment
        eligible = remaining_capacity > 0
        if not eligible.any():
            break

        # Distribute 1 sample per eligible group
        extras = np.zeros(num_groups, dtype=int)
        extras[eligible] = 1
        extras = np.minimum(extras, remaining_capacity)
        assignment += extras
        remaining_quota -= extras.sum()

    samples = []
    for name, group in grouped:
        n = assignment[groups.index(name)]
        if n > 0:
            samples.append(group.sample(n=n, random_state=random_state))
    group_sizes, counts = np.unique(assignment, return_counts=True)
    final_df = pd.concat(samples).reset_index(drop=True)
    logger.info(f"Obtained {final_df.shape[0]} samples")
    logger.info(
        f"With {counts[-1]} downsampled to size {group_sizes[-1]} and {sum(counts[:-1])} smaller groups"
    )

    return final_df


def guess_if_raw(x: np.ndarray, threshold=4):
    """
    Guess if data from a sample is raw or lognormalized.

    Args:
    ----
        x (np.ndarray): the X.data component of an anndata object
        threshold (int, optional): threshold. Defaults to 4.

    Returns:
    -------
        bool: if heuristic flags data as raw

    """
    is_int = np.issubdtype(x.dtype, np.integer)

    x_sample = x[:100_000] if x.size > 100_000 else x

    max_val = x_sample.max()
    mean_val = x_sample.mean()
    median_val = np.median(x_sample)
    pct_ones = np.mean(x_sample == 1)
    pct_small_ints = np.mean(np.isin(x_sample, [1, 2, 3]))

    # Scoring heuristic
    raw_score = sum(
        [
            is_int,
            max_val > 50,
            pct_ones > 0.4,
            mean_val < 2.5,
            median_val <= 1,
            pct_small_ints > 0.6,
        ]
    )

    return raw_score >= threshold


def make_group_means(
    ad: sc.AnnData,
    perturbation_column_name: str,
    split_column_name: str,
    exp_before_mean: bool = False,
    *,
    train_split_label: str = "train",
    control_label: str = "Control",
) -> sc.AnnData:
    """
    Compute per-perturbation pseudobulk means and append a train-average row.

    Aggregates single-cell expression data by perturbation group, producing
    one pseudobulk per perturbation plus an "Average_Perturbation_Train" row
    representing the mean of all non-control training perturbations.

    Parameters
    ----------
    ad : sc.AnnData
        Input single-cell data. Must contain `perturbation_column_name` and
        `split_column_name` in `.obs`.
    perturbation_column_name : str
        Column in `.obs` defining perturbation groups (e.g., "target_gene").
    split_column_name : str
        Column in `.obs` defining dataset split (e.g., "train", "test").
    exp_before_mean : bool, default False
        If True, compute unbiased means for log1p-normalized data via
        log1p(mean(expm1(X))). Use when `.X` is log1p-transformed.
    train_split_label : str, default "train"
        Label identifying training split in `.obs[split_column_name]`.
    control_label : str, default "Control"
        Perturbation label to exclude from train average computation.

    Returns
    -------
    sc.AnnData
        Rows are per-perturbation pseudobulks plus final "Average_Perturbation_Train"
        row (if training data exists). `.var` and `.uns` copied from input.

    Examples
    --------
    >>> # Standard usage with raw counts
    >>> agg = make_group_means(adata, "target_gene", "split")

    >>> # With log1p-normalized data (unbiased means)
    >>> agg = make_group_means(adata, "target_gene", "split", exp_before_mean=True)
    """

    def _mean(X):
        """Compute mean with optional log-bias correction."""
        X = X.toarray() if ss.issparse(X) else np.asarray(X)
        return np.log1p(np.expm1(X).mean(0)) if exp_before_mean else X.mean(0)

    def _pseudobulk(adata, group_col):
        grouped = adata.obs.groupby(group_col, sort=False, observed=True)
        indices = grouped.indices  # dict: group_name -> np.ndarray of positions
        group_names = list(indices.keys())
        if not group_names:
            raise ValueError(f"No groups found in {group_col}")

        means = [_mean(adata.X[pos]) for pos in indices.values()]
        return np.vstack(means), group_names

    # Compute all per-perturbation pseudobulks
    X_means, group_names = _pseudobulk(ad, perturbation_column_name)

    agg = sc.AnnData(
        X=ss.csr_matrix(X_means),
        obs=pd.DataFrame(index=pd.Index(group_names, name=perturbation_column_name)),
        var=ad.var.copy(deep=True),
        uns=dict(ad.uns.items()),
    )

    # Identify train perturbations (excluding control)
    train_perts = ad.obs.loc[
        (ad.obs[split_column_name] == train_split_label)
        & (ad.obs[perturbation_column_name] != control_label),
        perturbation_column_name,
    ].unique()

    if len(train_perts) == 0:
        return agg

    # Average the pseudobulks for train perturbations
    idx = [i for i, name in enumerate(group_names) if name in train_perts]
    avg_vec = X_means[idx].mean(0, keepdims=True)

    avg_row = sc.AnnData(
        X=ss.csr_matrix(avg_vec),
        obs=pd.DataFrame(index=pd.Index(["Average_Perturbation_Train"])),
        var=agg.var.copy(deep=True),
    )

    return sc.concat([agg, avg_row], axis=0, join="outer", merge="first")
