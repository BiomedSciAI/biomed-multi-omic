import os
from pathlib import Path

import scanpy as sc


def create_cell_eval_test_file(
    filename,
    test_filename,
    split_col="internal_split",
    pert_col="target_gene",
    control_pert="Control",
):
    """
    read perturbation h5ad, extract the test split and the control cells and save to an output file
    This format of files is required for comparison of predicion results using cell-eval pacakage.
    """
    adata = sc.read_h5ad(filename)
    adata_for_test = adata[
        (adata.obs[pert_col] == control_pert) | (adata.obs[split_col] == "test")
    ].copy()
    adata_for_test.obs[pert_col] = adata_for_test.obs[pert_col].astype(str)
    adata_for_test.obs.loc[
        adata_for_test.obs[pert_col] == control_pert, pert_col
    ] = "non-targeting"
    adata_for_test.write_h5ad(test_filename)
    print(
        f"test adata file: Shape: {adata_for_test.shape}, n_pert={adata_for_test.obs[pert_col].nunique()}, n_control={adata_for_test[adata_for_test.obs[pert_col]=='non-targeting'].shape[0]}"
    )


def subsample_perturbation_data(adata, n_pert=5, n_genes=100):
    pert_genes = ["non-targeting"] + list(adata.obs["target_gene"].unique())[:n_pert]

    adata_sub = adata[adata.obs["target_gene"].isin(pert_genes)][:, :n_genes].copy()
    return adata_sub


def generate_subsample_from_h5ad(filename):
    """
    Save a small sample of perturbation and genes to create a test file from a given h5ad file. The new file is save with same path and
    suffix .small.h5ad.
    """
    adata_combined = sc.read_h5ad(filename)
    output_filename = filename.replace(".h5ad", ".small.h5ad")

    n_pert = 5
    n_genes = 8000
    adata_sub = subsample_perturbation_data(
        adata_combined, n_pert=n_pert, n_genes=n_genes, output_h5ad=output_filename
    )
    adata_sub.write(output_filename)
    print(
        f"Saved subset with {n_pert} pertrubations, {adata_sub.n_obs} cells and {adata_sub.n_vars} genes to {output_filename}"
    )


if __name__ == "__main__":
    # for vcc
    # data_folder = Path(os.environ["BMFM_TARGETS_VCC_DATA"])
    # train_file = data_folder / "internal_split" / "train_and_test_processed.h5ad"

    # train_means_file = data_folder / "internal_split" / "train_means.h5ad"
    # generate_and_save_train_means(train_file, split_column="internal_split", output_train_means_file=train_means_file)
    # generate_subsample_from_h5ad(train_file)

    # for replogle
    data_folder = Path(os.environ["BMFM_TARGETS_REPLOGLE_DATA"])
    # train_file = data_folder / "replogle_logn_scgpt_split_processed.h5ad"
    train_file = data_folder / "replogle_logn_scgpt_split_processed_small.h5ad"

    # test_file = data_folder / "replogle_logn_scgpt_split_processed_test_for_eval.h5ad"
    test_file = (
        data_folder / "replogle_logn_scgpt_split_processed_small_test_for_eval.h5ad"
    )

    # train_small =  data_folder / "replogle_logn_scgpt_split_processed_small.h5ad"
    # create_cell_eval_test_file(
    #     train_file,
    #     test_file,
    #     split_col="scgpt_split",
    #     pert_col="gene",
    #     control_pert="Control",
    # )
