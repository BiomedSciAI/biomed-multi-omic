from pathlib import Path

import anndata as ad
import numpy as np

from bmfm_targets.datasets.datasets_utils import make_group_means


def fileset2aggregate_data(
    input_paths: list[Path],
    splits: list[str],
    output_path: Path,
    target_gene_column: str,
    control_label: str,
    train_split_label: str,
):
    controls = ad.read_h5ad(input_paths[0])
    controls = controls[controls.obs[target_gene_column] == control_label]
    adatas = []
    for path in input_paths:
        adata = ad.read_h5ad(path)
        adata = adata[adata.obs[target_gene_column] != control_label].copy()
        adatas.append(adata)

    adatas.append(controls)
    splits = splits.copy()
    nan_label = "__NaN__"
    splits.append(nan_label)
    adata = ad.concat(adatas, axis=0, label="split", keys=splits)
    adata.obs["split"] = adata.obs["split"].replace(nan_label, np.nan)
    agg = build_aggregate_data(
        adata,
        target_gene_column,
        control_label,
        split_column="split",
        train_split_label=train_split_label,
    )
    agg.write_h5ad(output_path)


def build_aggregate_data(
    input_path: Path | ad.AnnData,
    target_gene_column: str,
    control_label: str,
    split_column: str,
    train_split_label: str,
):
    """
    Build aggregate perturbation data with train average.

    Now a thin wrapper around make_group_means. Returns all per-perturbation
    pseudobulks plus "Average_Perturbation_Train" row. Control label is
    normalized to "Control".

    Parameters
    ----------
    input_path : Path | ad.AnnData
        Path to h5ad file or AnnData object
    target_gene_column : str
        Column defining perturbations
    control_label : str
        Original control label (will be renamed to "Control")
    split_column : str
        Column defining train/dev/test splits
    train_split_label : str
        Label for training split

    Returns
    -------
    ad.AnnData
        Aggregated data with all perturbation pseudobulks + train average row
    """
    adata = (
        input_path if isinstance(input_path, ad.AnnData) else ad.read_h5ad(input_path)
    )

    result = make_group_means(
        adata,
        perturbation_column_name=target_gene_column,
        split_column_name=split_column,
        exp_before_mean=False,
        train_split_label=train_split_label,
        control_label=control_label,
    )

    # Normalize control label to "Control"
    result.obs_names = result.obs_names.where(
        result.obs_names != control_label, "Control"
    )

    return result
