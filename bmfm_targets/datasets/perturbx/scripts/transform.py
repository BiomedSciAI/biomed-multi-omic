#!/usr/bin/env python

from pathlib import Path

import anndata as ad
import click
import numpy as np
from omegaconf import OmegaConf

from bmfm_targets.datasets import PerturbationDatasetTransformer

TRANSFORM_PREFIX = "transformed_"


def get_transformed_filename(h5ad_path):
    in_path = Path(h5ad_path)
    return in_path.with_name(f"{TRANSFORM_PREFIX}{in_path.name}")


def transform_anndata(
    h5ad_path: str,
    config_path: str = "configs/transforms.yaml",
    reindex_var_names: bool = False,
    var_name_col: str = "gene_name",
    rename_control: bool = False,
    target_gene_column: str = "gene",
    control_label: str = "Control",
) -> str:
    """
    Load an AnnData .h5ad file, perform transforms specified by config.
    Save result to a new file with 'transformed_' prefix.

    Args:
    ----
    h5ad_path : str
        Path to input .h5ad file.
    config_path: str
        Path to .yaml file with 'tokenizer' and 'transforms_kwargs'
    reindex_var_names: bool
        Reindex the var_names in the adata before further processing
    var_name_col: str
        The new var to use for the gene names, if reindexing
    rename_control: bool
        Rename the perturb_column value from control_name to "Control"
    target_gene_column: str
        Name of the column with perturbation info in obs. Default: "gene"
    control_label: str
        Original name of the control cells. Default: "Control"

    Returns:
    -------
    str
        Path to the saved shuffled .h5ad file.
    """
    config = OmegaConf.load(config_path)
    transform_kwargs = config["transform_kwargs"]

    # Prevent circular import
    from bmfm_targets.tokenization import load_tokenizer

    tokenizer = load_tokenizer(config["tokenizer"]["identifier"])

    # PerturbationDatasetTransformer expects a list here?
    source_h5ad_file_names = [h5ad_path]

    # This is set in data_module in the main code
    perturbation_column_name = target_gene_column or config["data_module"].get(
        "perturbation_column_name", "gene"
    )

    perturbation_column_name = target_gene_column

    transformer = PerturbationDatasetTransformer(
        tokenizer=tokenizer,
        source_h5ad_file_names=source_h5ad_file_names,
        perturbation_column_name=perturbation_column_name,
        transforms=transform_kwargs.get("transforms", None),
        split_column_name=transform_kwargs.get("split_column_name", None),
        group_column_name=transform_kwargs.get("group_column_name", None),
        split_weights=transform_kwargs.get("split_weights", None),
        random_state=transform_kwargs.get("random_state", 42),
    )

    raw_data = ad.read_h5ad(h5ad_path)

    if reindex_var_names:
        raw_data = _set_index_names_from_var_column(adata=raw_data, column=var_name_col)

    if rename_control:
        raw_data = _rename_control(
            adata=raw_data,
            target_gene_column=target_gene_column,
            control_label=control_label,
        )

    # Check h5ad file for a few issues
    cleaned_data = transformer._clean_dataset(raw_data)

    # Process specified transforms
    processed_data = transformer.transforms(adata=cleaned_data)["adata"]
    transformer.add_de_genes_to_uns(processed_data)
    processed_data.uns.pop(
        "rank_genes_groups", None
    )  # This is huge and we dont use it later in the code

    out_path = get_transformed_filename(h5ad_path)
    processed_data.write_h5ad(out_path)

    return str(out_path)


def _set_index_names_from_var_column(
    adata: ad.AnnData,
    column: str = "gene_name",
    tiebreaker_column: str = "mean",
) -> ad.AnnData:
    """
    Replace adata.var_names with values from a specified column.

    This is required if the values in 'var_names' in the original adata
    do not match the types of values in 'target_gene_column' (e.g.
    dataset has var_names as 'gene_ids' but 'target_gene_column is a
    'gene_name').
    If duplicates exist, keep the row with the highest 'mean' value.
    Returns a new AnnData object (view copy).

    Args:
    ----
    adata : AnnData
        adata object from .h5ad file
    column: str
        Which var column to map into the var_names
    tiebreaker_column: str
        Method for solving 'ties' where a gene name maps to multiple
        gene ids

    Returns:
    -------
    Anndata
        New anndata object with reindexed var_names variable.
    """
    symbols = adata.var[column].astype(str)
    if symbols.is_unique:
        # If it is unique we can already use the column as var_names
        adata.var_names = symbols
        return adata

    # If not unique, sort a column as the tiebreaker
    var_name_name = adata.var_names.name
    var = adata.var.reset_index(var_name_name).copy()
    var["__idx__"] = np.arange(var.shape[0])
    var_sorted = var.sort_values(tiebreaker_column, ascending=False)
    keep_idx = var_sorted.drop_duplicates(subset=column, keep="first")["__idx__"]
    keep_idx = np.sort(keep_idx)

    adata = adata[:, keep_idx].copy()
    adata.var_names = adata.var[column].astype(str)

    if column in adata.var.columns:
        adata.var = adata.var.drop(columns=[column])

    return adata


def _rename_control(
    adata: ad.AnnData,
    target_gene_column: str = "gene",
    control_label: str = "Control",
) -> ad.AnnData:
    """
    Rename the control variable to "Control" in the
    perturbation_column_name column.

    Args:
    ----
    adata (ad.AnnData)
        AnnData object containing perturbation data
    perturbation_column_name: str
        Which column of obs the perturbation information is in
    control_name (str)
        Name of the control values in the current dataset

    Returns:
    -------
    Anndata
        New anndata object with renamed controls.
    """
    if control_label == "Control":
        return adata

    control_mask = adata.obs[target_gene_column] == control_label
    current_categories = adata.obs[target_gene_column].cat.categories
    new_categories = current_categories.tolist() + ["Control"]
    new_categories.remove(control_label)
    adata.obs[target_gene_column] = adata.obs[target_gene_column].cat.set_categories(
        new_categories
    )
    adata.obs.loc[control_mask, target_gene_column] = "Control"

    return adata


@click.command()
@click.option(
    "--adata-path",
    type=click.Path(exists=True),
    help="Path to the AnnData file (.h5ad format)",
)
@click.option(
    "--config-path",
    default="../configs/transforms.yaml",
    type=click.Path(exists=True),
    help="Path to the config file (.yaml format)",
)
@click.option(
    "--reindex-var-names",
    is_flag=True,
    help="Reindex the var_names to a new column, using 'var_names' column.",
)
@click.option(
    "--var-names", default="gene_name", help="Which column to use for adata.var_names."
)
@click.option(
    "--rename-control",
    is_flag=True,
    help="Rename perturb_column[control_name] to 'Control'.",
)
@click.option(
    "--target-gene-column",
    default="gene",
    help="Which column contains perturbation names.",
)
@click.option(
    "--control-label", default="Control", help="Current value of control column."
)
def main(
    adata_path,
    config_path,
    reindex_var_names,
    var_names,
    rename_control,
    target_gene_column,
    control_label,
):
    click.echo(f"AnnData path: {adata_path}")
    click.echo(f"Config path: {config_path}")
    click.echo(f"Reindex var_names: {reindex_var_names}")
    click.echo(f"var_names: {var_names}")
    click.echo(f"Rename controls: {rename_control}")
    click.echo(f"Target gene column: {target_gene_column}")
    click.echo(f"Control_label: {control_label}")

    transformed_path = transform_anndata(
        adata_path,
        config_path,
        reindex_var_names,
        var_names,
        rename_control,
        target_gene_column,
        control_label,
    )

    print(f"Transformed adata written to: {transformed_path}")


if __name__ == "__main__":
    main()
