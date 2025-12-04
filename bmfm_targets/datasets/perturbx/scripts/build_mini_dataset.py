#!/usr/bin/env python

"""The script that builds subset of the anndata with perturbations, selecting only few examples per target genes, see options."""

from pathlib import Path

import anndata as ad
import click
import numpy as np
import pandas as pd
import scipy.sparse as sp


def sparsify_genes(
    adata: ad.AnnData, drop_fraction: float = 0.5, seed: int = 0, copy: bool = True
):
    """
    Randomly drop nonzero entries in adata.X while ensuring each row keeps >= 1 nonzero.
    Returns a new AnnData.

    Args:
    ----
    adata : AnnData
        Input with X as a CSR/CSC sparse matrix or dense.
    drop_fraction : float
        Probability to drop each *non-forced* nonzero in a row. Clipped to [0, 1).
    seed : int
        RNG seed.
    copy : bool
        If True, return a new AnnData; otherwise modify adata.X in place.

    Returns:
    -------
    AnnData
        New object if copy=True, else returns the same adata with sparsified X.
    """
    if not (0 <= drop_fraction < 1):
        raise ValueError("drop_fraction must be in [0, 1).")

    rng = np.random.default_rng(seed)

    X = adata.X
    X_csr = X.copy() if copy else X

    indptr = X_csr.indptr
    indices = X_csr.indices
    data = X_csr.data

    n_rows = X_csr.shape[0]
    new_indptr = np.zeros_like(indptr)
    kept_indices = []
    kept_data = []

    nnz_total = data.size
    global_mask_ind = []

    for i in range(n_rows):
        start, end = indptr[i], indptr[i + 1]
        row_nnz = end - start
        if row_nnz == 0:
            new_indptr[i + 1] = new_indptr[i]
            continue

        # Force-keep exactly one original nonzero
        force_idx_pos = int(rng.integers(low=0, high=row_nnz))

        # For the rest, keep with probability (1 - drop_fraction)
        if row_nnz > 1:
            if drop_fraction <= 0.0:
                mask_keep_local = np.ones(row_nnz, dtype=bool)
                mask_keep_local[force_idx_pos] = True
            else:
                u = rng.random(row_nnz)
                mask_keep_local = u > drop_fraction
                mask_keep_local[force_idx_pos] = True

            mask_ind = np.where(mask_keep_local)[0] + start
            global_mask_ind.append(mask_ind)

        new_indptr[i + 1] = new_indptr[i] + len(mask_ind)

    global_mask_ind = np.hstack(global_mask_ind)
    kept_indices = np.copy(indices[global_mask_ind])
    kept_data = np.copy(data[global_mask_ind])
    local_kept = len(kept_data)

    X_new = sp.csr_matrix((kept_data, kept_indices, new_indptr), shape=X_csr.shape)
    X_new.eliminate_zeros()

    if copy:
        return ad.AnnData(
            X_new, obs=adata.obs.copy(), var=adata.var.copy(), uns=adata.uns.copy()
        )
    else:
        adata.X = X_new
        return adata


def subset_anndata_by_gene(adata, gene_column="gene", n_samples_per_gene=2):
    """
    Create a subset of AnnData object with exactly n_samples_per_gene examples per gene.

    Args:
    ----
    adata : anndata.AnnData
        Input AnnData object
    gene_column : str
        Column name in adata.obs that contains gene information
    n_samples_per_gene : int
        Number of samples to select per gene

    Returns:
    -------
    anndata.AnnData
        Subsetted AnnData object
    """
    # Check if gene column exists
    if gene_column not in adata.obs.columns:
        raise ValueError(f"Column '{gene_column}' not found in adata.obs")

    # Get unique genes
    unique_genes = adata.obs[gene_column].unique()
    print(f"Found {len(unique_genes)} unique genes")

    # Initialize list to store indices
    selected_indices = []

    # For each gene, select exactly n_samples_per_gene examples
    for gene in unique_genes:
        # Get indices for current gene
        gene_indices = adata.obs[adata.obs[gene_column] == gene].index

        # If there are fewer samples than requested, take all available
        if len(gene_indices) < n_samples_per_gene:
            print(
                f"Warning: Gene '{gene}' has only {len(gene_indices)} samples, taking all"
            )
            selected_indices.extend(gene_indices.tolist())
        else:
            # Randomly sample n_samples_per_gene indices
            sampled_indices = np.random.choice(
                gene_indices, size=n_samples_per_gene, replace=False
            )
            selected_indices.extend(sampled_indices.tolist())

    # Create subset
    adata_subset = adata[selected_indices, :].copy()

    print(f"Original data shape: {adata.shape}")
    print(f"Subset data shape: {adata_subset.shape}")
    print(f"Genes in subset: {len(adata_subset.obs[gene_column].unique())}")

    # Verify the subset has the correct number of samples per gene
    gene_counts = adata_subset.obs[gene_column].value_counts()
    print("Samples per gene in subset:")
    print(gene_counts.describe())

    return adata_subset


def subset_anndata_exactly_n_per_gene(
    adata, gene_column="gene", n_samples_per_gene=2, output_file="subset_exact.h5ad"
):
    """Create subset with EXACTLY n_samples_per_gene per gene, excluding genes with fewer samples."""
    # Get gene counts
    gene_counts = adata.obs[gene_column].value_counts()

    # Keep only genes that have at least n_samples_per_gene samples
    valid_genes = gene_counts[gene_counts >= n_samples_per_gene].index

    print(f"Genes with at least {n_samples_per_gene} samples: {len(valid_genes)}")
    print(f"Genes excluded: {len(gene_counts) - len(valid_genes)}")

    selected_indices = []

    for gene in valid_genes:
        gene_indices = adata.obs[adata.obs[gene_column] == gene].index
        # Randomly sample exactly n_samples_per_gene
        sampled_indices = np.random.choice(
            gene_indices, size=n_samples_per_gene, replace=False
        )
        selected_indices.extend(sampled_indices.tolist())

    adata_subset = adata[selected_indices, :].copy()

    print(f"Final subset shape: {adata_subset.shape}")
    print(
        f"All genes have exactly {n_samples_per_gene} samples: {all(adata_subset.obs[gene_column].value_counts() == n_samples_per_gene)}"
    )

    # Save to h5ad file
    adata_subset.write_h5ad(output_file)
    print(f"Subset saved to: {output_file}")

    return adata_subset


@click.command()
@click.option("--input-file", "-i", required=True, help="Input h5ad file path")
@click.option("--output-file", "-o", required=True, help="Output h5ad file path")
@click.option(
    "--gene-column",
    "-g",
    default="gene",
    help="Column name containing gene information (default: gene)",
)
@click.option(
    "--n-samples",
    "-n",
    default=2,
    type=int,
    help="Number of samples per gene (default: 2)",
)
@click.option(
    "--exact/--allow-fewer",
    default=False,
    help="Require exactly n samples per gene or allow fewer (default: allow-fewer)",
)
@click.option(
    "--drop-fraction",
    "-d",
    default=0.0,
    type=float,
    help="Fraction of entries in X matrix to drop",
)
@click.option("--drop-col", multiple=True, help="Drop column")
@click.option("--drop-var", is_flag=True, help="Enable verbose mode.")
@click.option("--drop-uns", is_flag=True, help="Enable verbose mode.")
def main(
    input_file,
    output_file,
    gene_column,
    n_samples,
    exact,
    drop_fraction: float,
    drop_col: list[str],
    drop_var: bool,
    drop_uns: bool,
):
    """
    Subset AnnData object with specified number of samples per gene.

    This tool creates a subset of an AnnData object by selecting a specific number
    of samples for each unique value in the gene column.
    """
    try:
        # Load the AnnData object
        click.echo(f"Loading data from: {input_file}")
        adata = ad.read_h5ad(input_file)
        click.echo(f"Original data shape: {adata.shape}")

        if exact:
            click.echo(
                f"Using exact mode: excluding genes with fewer than {n_samples} samples"
            )
            adata_subset = subset_anndata_exactly_n_per_gene(
                adata, gene_column=gene_column, n_samples_per_gene=n_samples
            )
        else:
            click.echo(
                f"Using flexible mode: allowing genes with fewer than {n_samples} samples"
            )
            adata_subset = subset_anndata_by_gene(
                adata, gene_column=gene_column, n_samples_per_gene=n_samples
            )

        if drop_fraction != 0.0:
            adata_subset = sparsify_genes(adata_subset, drop_fraction, copy=True)

        drop_col = list(drop_col)
        if drop_col:
            adata_subset.obs = adata_subset.obs.drop(columns=drop_col)

        if drop_var:
            adata_subset.var = pd.DataFrame(index=adata_subset.var.index)
        if drop_uns:
            adata_subset.uns.clear()

        # Save to h5ad file
        adata_subset.write_h5ad(output_file, compression="gzip")
        print(f"Subset saved to: {output_file}")

        size_bytes = Path(output_file).stat().st_size / (1024**2)
        click.echo(f"File size {size_bytes} MB.")
        click.echo("Subsetting completed successfully!")

    except FileNotFoundError:
        click.echo(f"Error: Input file '{input_file}' not found.", err=True)


if __name__ == "__main__":
    main()
