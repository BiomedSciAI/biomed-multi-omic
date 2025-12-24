#!/usr/bin/env python

from pathlib import Path

import click

from bmfm_targets.datasets.perturbx.scripts.aggregate_data import fileset2aggregate_data


@click.command()
@click.option(
    "--input-paths",
    type=click.Path(exists=True),
    multiple=True,
    required=True,
    help="Path to the AnnData file (.h5ad format)",
)
@click.option(
    "--splits",
    multiple=True,
    required=True,
    help="List of index splits, Example: --splits train --splits dev.",
)
@click.option(
    "--output-path",
    type=click.Path(writable=True, dir_okay=False, resolve_path=True),
    help="Path to the AnnData for resulting aggregated data (.h5ad format)",
)
@click.option(
    "--target-gene-column",
    default="target_gene",
    help='Column name containing target gene information (default: "gene")',
)
@click.option(
    "--control-label",
    default="non-targeting",
    help='Label used to identify control samples (default: "non-targeting")',
)
@click.option(
    "--split-column",
    default="split",
    help='Column name for data splitting, could be None (default: "split")',
)
@click.option(
    "--train-split-label",
    default="train",
    help='Label for train data in split column(default: "train")',
)
def main(
    input_paths: list[Path],
    splits: list[str],
    output_path: Path,
    target_gene_column: str,
    control_label: str,
    split_column: str,
    train_split_label: str,
):
    fileset2aggregate_data(
        input_paths,
        splits,
        output_path,
        target_gene_column,
        control_label,
        train_split_label,
    )


if __name__ == "__main__":
    main()
