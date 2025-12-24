from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


def generate_test_adata(summary_filename: str, gene_list_filename):
    """
    Given a summary pertubation data csv file of format: target_gene, n_cells, median_umi_per_cell
    Creates a random h5ad file for all perturbations of size (total_cells, n_genes), where each row expression sum is ~median_umi_per_cell
    for that perturbation.
    """
    summary = pd.read_csv(summary_filename)
    genes = read_gene_list(gene_list_filename)
    n_genes = len(genes)

    all_dfs = []
    obs = []

    for _, row in summary.iterrows():
        target_gene = row["target_gene"]
        n_cells = int(row["n_cells"])
        median_umi = int(row["median_umi_per_cell"])
        print(f"{target_gene},{n_cells},{median_umi}")
        # random expressions for all cells and all genes
        expressions_df = pd.DataFrame(
            np.random.poisson(lam=1.0, size=(n_cells, n_genes)), columns=genes
        )
        row_sums = expressions_df.sum(axis=1).replace(
            0, 1
        )  # sum and avoid rare zero sum
        expressions_df = (expressions_df.T / row_sums).T * median_umi
        expressions_df = expressions_df.round().astype(int)

        all_dfs.append(expressions_df)
        obs.extend(
            {
                "target_gene": target_gene,
                "cell_id": f"{target_gene}_cell_{i+1}",
                "nUMI": expressions_df.iloc[i].sum(),
            }
            for i in range(n_cells)
        )

        full_df = pd.concat(all_dfs, ignore_index=True)

    obs = pd.DataFrame(obs).set_index("cell_id")
    var = pd.DataFrame(index=genes)
    X = sp.csr_matrix(full_df)
    adata = sc.AnnData(X=X, obs=obs, var=var)
    return adata


def read_gene_list(gene_list_filename):
    with open(gene_list_filename) as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes


def add_non_targeting(adata, train_h5ad: str):
    train_adata = sc.read_h5ad(train_h5ad)

    # subset original: only rows with target_gene == "non-targeting"
    adata_control = train_adata[train_adata.obs["target_gene"] == "non-targeting"]

    genes = adata.var_names
    genes_control = adata_control.var_names

    assert genes.equals(genes_control), "different gene lists! can't combine"
    adata_combined = sc.concat([adata, adata_control], axis=0, join="outer")

    return adata_combined


def generate_test_h5ad_file(summary_file, gene_list_file, train_file, output_filename):
    """
    generate a syntethetic h5ad file with distribution as in a given summary file.
    Given a summary pertubation data csv file of format: target_gene, n_cells, median_umi_per_cell and a gene list,
    creates a syntethetic h5ad file for all perturbations of size (total_cells, n_genes),
    where each cell expressions are random but sums up to the ~median_umi_per_cell for that perturbation.
    Add control cells from the given train file.

    """
    adata = generate_test_adata(summary_file, gene_list_file)

    adata_combined = add_non_targeting(adata, train_file)
    adata_combined.obs[
        "split"
    ] = "test"  # all file is the test set - a split with test set is required for this to be run as a test task with TestTaskConfig

    print("generating h5ad file")
    print(f"shape: {adata_combined.shape}")
    print("sample:")
    print(adata_combined.X[:10, :10])

    print(f"saving to {output_filename}")
    adata_combined.write_h5ad(output_filename)


def generate_summary_from_h5ad(h5ad_file, output_summary_file):
    pass


def validate_perturbations(adata, pert_summary):
    """- for each perturabtion - validate if the number of cells is as expected."""
    completed_adata = adata.copy()
    for _, row in pert_summary.iterrows():
        target_gene = row["target_gene"]
        expected_n_cells = int(row["n_cells"])
        median_umi = float(row["median_umi_per_cell"])

        # Check if target_gene exists in completed_adata
        if target_gene in completed_adata.obs["target_gene"].values:
            actual_n_cells = (completed_adata.obs["target_gene"] == target_gene).sum()
            if actual_n_cells != expected_n_cells:
                print(
                    f"Warning: perturbation {target_gene} has {actual_n_cells} cells, expected {expected_n_cells}"
                )
        else:
            print(f"Warning: missing perturbation: {target_gene} !!")

    return completed_adata


def complete_predictions_h5ad(
    predictions_file, train_file, completed_predictions_file, gene_col="target_gene"
):
    """Add control cells from train file and rename to 'non-targeting'."""
    predictions_adata = sc.read_h5ad(predictions_file)
    train_adata = sc.read_h5ad(train_file)
    control_adata = train_adata[train_adata.obs[gene_col] == "Control"].copy()
    control_adata.obs[gene_col] = "non-targeting"
    predictions_adata.obs = predictions_adata.obs.rename(
        columns={"target_gene": gene_col}
    )  # todo - fix predictions files that are saved with target_gene
    final_completion = predictions_adata.concatenate(
        control_adata, join="outer", batch_key=None
    )

    print(
        f"Completed adata file: Shape: {final_completion.shape}, n_pert={final_completion.obs[gene_col].nunique()}, n_control={final_completion[final_completion.obs[gene_col]=='non-targeting'].shape[0]}"
    )

    print("Saving completed file to: " + completed_predictions_file)
    final_completion.write_h5ad(completed_predictions_file)


def validate_predictions_h5ad(predictions_file, summary_file):
    """Validate perturbations number of cells againset a requirements summary file."""
    predictions_adata = sc.read_h5ad(predictions_file)
    pert_summary = pd.read_csv(summary_file)

    print(
        f"Predictions adata file: Shape: {predictions_adata.shape}, n_pert={predictions_adata.obs['target_gene'].nunique()}"
    )

    validate_perturbations(predictions_adata, pert_summary)


if __name__ == "__main__":
    # data_folder = Path(os.environ["BMFM_TARGETS_VCC_DATA"])
    data_folder = Path("/proj/bmfm/datasets/vcc/vcc_data")
    output_filename = (
        data_folder / "validation_synthetic_30oct" / "validation_synthetic.h5ad"
    )
    validation_summary_file = data_folder / "pert_counts_Validation.csv"
    gene_list_file = data_folder / "gene_names.csv"
    train_file = (
        data_folder.parent / "internal_split" / "train_and_test_09162025_processed.h5ad"
    )

    generate_test_h5ad_file(
        validation_summary_file, gene_list_file, train_file, output_filename
    )

    # VCC
    # predictions_file = "/dccstor/bmfm-targets1/users/sivanra/training_runs/perturbation/scbert_perturbation_vcc_internal_test/09162025b/predictions.h5ad"
    # validate_predictions_h5ad(
    #         predictions_file,
    #         validation_summary_file,
    #     )
    # completed_predictions_file = predictions_file.replace(".h5ad", ".complete.h5ad")
    # complete_predictions_h5ad(
    #     predictions_file,
    #     train_file,
    #     completed_predictions_file,
    # )

    # REPLOGLE
    # data_folder = Path(os.environ["BMFM_TARGETS_REPLOGLE_DATA"])
    # train_file = data_folder / "replogle_logn_scgpt_split_processed.h5ad"
    # predictions_file = "/dccstor/bmfm-targets1/users/sivanra/training_runs/perturbation/scbert_perturbation_replogle_logn/100525/random_last_predictions.h5ad"
    # completed_predictions_file = predictions_file.replace(".h5ad", ".complete.h5ad")
    # complete_predictions_h5ad(
    #     predictions_file, train_file, completed_predictions_file, gene_col="gene"
    # )
