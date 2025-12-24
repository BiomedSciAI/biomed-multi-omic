#!/usr/bin/env python

from pathlib import Path

import click

from .aggregate_data import build_aggregate_data


@click.command()
@click.option(
    "--input-path",
    type=click.Path(exists=True),
    help="Path to the AnnData file (.h5ad format)",
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
    input_path: Path,
    output_path: Path,
    target_gene_column: str,
    control_label: str,
    split_column: str,
    train_split_label: str,
):
    agg = build_aggregate_data(
        input_path,
        target_gene_column,
        control_label,
        split_column,
        train_split_label,
    )
    agg.write_h5ad(output_path)


if __name__ == "__main__":
    main()
