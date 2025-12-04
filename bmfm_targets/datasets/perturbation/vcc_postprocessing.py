import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from bmfm_targets.datasets.datasets_utils import make_group_means


def override_noise_prediction_with_baseline(
    adata, baseline_vector, absolute=True, threshold=0.1
):
    """
    Replace values in AnnData with baseline values if the difference is lower than a ratio threshold.


    Parameters
    ----------
    adata (AnnData):  Annotated data matrix
    baseline_vector : Vector of baseline values per gene
    threshold (float):  Threshold for difference (default: 0.1)
    absolute (bool): if True (default) treat the th as absolute change th, otherwise as relative change th


    Returns
    -------
    AnnData -  Modified AnnData
    """
    epsilon = 0.01  # used in relative change calculation to avoid divide by zero, wonder if we need to test different values?
    baseline = np.array(baseline_vector)

    if len(baseline) != adata.n_vars:
        raise ValueError(
            f"Baseline vector length ({len(baseline)}) must match number of genes ({adata.n_vars})"
        )

    X = adata.X
    is_sparse = hasattr(X, "toarray")
    X_dense = X.toarray() if is_sparse else np.array(X, dtype=float)

    baseline_broadcast = baseline[np.newaxis, :]  # make it [1,n_genes]
    diff = X_dense - baseline_broadcast
    if absolute:
        mask = np.abs(diff) < threshold
    else:
        # Relative change: |value - baseline| / |baseline|
        # (1) Replace zero baselines with epsilon to so relative changes won't be exterimely high
        baseline_for_rel = baseline_broadcast.copy()
        zero_mask = baseline_for_rel == 0
        baseline_for_rel[zero_mask] = epsilon

        # (2) ignore negative changes from the epsilon we added( if baseline was zero and diff is smaller than epsilon then ignore it)
        diff[zero_mask & (diff < 0)] = 0

        # (3) compute Relative change = |value - baseline| / |baseline|
        relative_change = np.abs(diff) / baseline_for_rel
        mask = relative_change < threshold

    baseline_full = np.broadcast_to(baseline_broadcast, X_dense.shape)
    X_dense[mask] = baseline_full[mask]

    if is_sparse:
        from scipy.sparse import csr_matrix

        X_new = csr_matrix(X_dense)
    else:
        X_new = X_dense

    adata.X = X_new

    print(
        f"Replaced {mask.sum()} values (out of {mask.size} total) with baseline values"
    )
    return adata


def cap_outlier_predictions(adata, output_file, th=0.9):
    """Cap expression values above the given quantile threshold per gene. ignore zeros on quantile computation."""
    capped_adata = adata.copy()
    X = capped_adata.X.toarray()
    qdf = pd.DataFrame(X, columns=adata.var_names, index=adata.obs_names)

    # Compute per-gene quantiles (excluding zeros)
    quantile_series = qdf.apply(lambda x: x[x > 0].quantile(th)).fillna(0)

    # Vectorized capping
    qdf = qdf.where(qdf <= quantile_series, quantile_series, axis=1)

    # Assign capped values back to AnnData
    capped_adata.X = sp.csr_matrix(qdf.to_numpy())

    # Save to file
    capped_adata.write(output_file)

    print(f"Capped expressions saved to: {output_file}")
    return adata


if __name__ == "__main__":
    # This script runs the best post processing techniques on a latest results file.
    # for exploration of these techniques see vcc_postprocessing.ipynb

    train_file = (
        "/proj/bmfm/datasets/vcc/internal_split/train_and_test_09162025_processed.h5ad"
    )
    train_adata = sc.read_h5ad(train_file)

    # compute baseline vector
    print(f"train data shape: {train_adata.shape}")
    train_group_means = make_group_means(
        train_adata,
        perturbation_column_name="target_gene",
        split_column_name="internal_split_for_test",
    )
    baseline = train_group_means.to_df().loc["Average_Perturbation_Train"]
    print(f"baseline vector shape: {baseline.shape}")

    # post process restuls file:
    # internal validation - best results (DES=0.24, PDS=0.656)
    # res_file = "/proj/bmfm/users/thrumbel/vcc/expts/perturbx/all/perturbation/scbert_perturbx_all_logn/10092025/random_8192_predictions.h5ad"

    # external validation October 30
    res_file = "/proj/bmfm/users/thrumbel/vcc/expts/perturbx/all/perturbation/scbert_perturbx_all_logn/10092025/validation_synthetic_30oct/predictions.h5ad"
    res_adata = sc.read_h5ad(res_file)
    print(f"res data shape: {res_adata.shape}")
    output_file = res_file.replace(".h5ad", ".pp.h5ad")
    res_adata_pp = override_noise_prediction_with_baseline(
        res_adata, baseline, absolute=True, threshold=0.5
    )

    # remove noise from results
    res_adata_pp = res_adata_pp = cap_outlier_predictions(
        res_adata_pp, output_file, th=0.9
    )
    print(f"res data post-processed shape: {res_adata_pp.shape}")
    print(
        f'res data post-processed num perturbations: {res_adata_pp.obs["target_gene"].nunique()}'
    )
    print("res data post-processed preturbation value_counts")
    print(res_adata_pp.obs["target_gene"].value_counts())
    print(f"Saved post processed file to: {output_file}")

    # if running on internal_validation file, check results with cell-eval:
    # out_folder = output_file.replace(".h5ad", "_cell_eval")
    # de_real = "/proj/bmfm/users/sivanra/vcc_score/cell_eval_res_orig/real_de.csv"
    # os.mkdir(out_folder)
    # print()
    # print("To evaluate against internal validation ground-truth run:")
    # print()
    # print(
    #     f"cell-eval run -ap {output_file} -ar /proj/bmfm/datasets/vcc/internal_split/train_and_test_09162025_processed_test_only.h5ad --num-threads 64 --pert-col target_gene --profile vcc -o {out_folder} --de-real {de_real}"
    # )
