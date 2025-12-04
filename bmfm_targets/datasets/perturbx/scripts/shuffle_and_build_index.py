#!/usr/bin/env python

from pathlib import Path

import anndata as ad
import click
import numpy as np

from bmfm_targets.datasets.perturbx import (
    build_perturbx_index,
    get_shuffled_filename,
    shuffle_anndata,
)


def make_split_folder_path(file_path: str, suffix: str = "dev") -> Path:
    p = Path(file_path)
    return str(p.parent / (p.stem + "_" + suffix))


@click.command()
@click.option(
    "--adata-path",
    type=click.Path(exists=True),
    help="Path to the AnnData file (.h5ad format)",
)
@click.option(
    "--target-gene-column",
    default="gene",
    help='Column name containing target gene information (default: "gene")',
)
@click.option(
    "--control-label",
    default="Control",
    help='Label used to identify control samples (default: "Control")',
)
@click.option(
    "--split-column",
    default="none",
    help='Column name for data splitting, could be None (default: "scgpt_split")',
)
@click.option(
    "--splits",
    multiple=True,
    required=True,
    help="List of index splits, Example: --splits train --splits dev.",
)
@click.option("--chunk-size", type=int, default=2000, help="Chunk size for index.")
@click.option(
    "--reuse_shuffled",
    is_flag=True,
    help="Reuse previously created shuffled h5ad",
)
def main(
    adata_path,
    target_gene_column,
    control_label,
    split_column,
    splits,
    chunk_size,
    reuse_shuffled,
):
    shuffle_and_build_index(
        adata_path,
        target_gene_column,
        control_label,
        split_column,
        splits,
        chunk_size,
        reuse_shuffled,
    )


def shuffle_and_build_index(
    adata_path,
    target_gene_column,
    control_label,
    split_column,
    splits,
    chunk_size,
    reuse_shuffled=False,
):
    """Main function for processing single-cell data with configurable parameters."""
    if split_column and split_column.lower() == "none":
        split_column = None

    if len(splits) > 1 and not split_column:
        raise ValueError(
            "Multiple options --splits are used only with split_column, specify split_column."
        )

    click.echo(f"AnnData path: {adata_path}")
    click.echo(f"Target gene column: {target_gene_column}")
    click.echo(f"Control label: {control_label}")
    click.echo(f"Split column: {split_column}")

    shuffled_filename = (
        get_shuffled_filename()
        if reuse_shuffled
        else shuffle_anndata(
            h5ad_path=adata_path,
            target_gene_column=target_gene_column,
            control_label=control_label,
            split_column=split_column,
        )
    )

    adata = ad.read_h5ad(shuffled_filename)
    control_mask = adata.obs[target_gene_column] == control_label
    control = np.where(control_mask)[0]

    if split_column:
        for split in splits:
            split_index = np.where(adata.obs[split_column] == split)[0]
            build_perturbx_index(
                output_dir=make_split_folder_path(shuffled_filename, suffix=split),
                target_gene_column=target_gene_column,
                control_label=control_label,
                control_index=control,
                split_index=split_index,
                chunk_size=chunk_size,
            )
    else:
        non_control = np.where(~control_mask)[0]
        build_perturbx_index(
            output_dir=make_split_folder_path(shuffled_filename, suffix=splits[0]),
            target_gene_column=target_gene_column,
            control_label=control_label,
            control_index=control,
            split_index=non_control,
            chunk_size=chunk_size,
        )


if __name__ == "__main__":
    main()
