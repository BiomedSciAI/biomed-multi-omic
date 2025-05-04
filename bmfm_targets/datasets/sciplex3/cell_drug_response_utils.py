import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score

from bmfm_targets.training.metrics.plots import make_predictions_gt_density_plot


def calculate_baseline_control_results(control_adata, target_adata, limit_genes=None):
    """
    Calculate a baseline for r2 metric, by prediction the mean expression of the genes in the control set.
    Returns R2 per cell_type, drug, dosage combination.

    Assumes the `cov_drug_dose_name` cell_type,drug,dosage combination is given in the AnnData obs.

    Args:
    ----
    control_adata (AnnData): Control cells RNA expression
    target_adata (AnnData): the true RNAExpression to evaluate
    limit_genes (None|int): number of genes used (None to use all)

    Returns:
    -------
        pd.DataFrame: r2 per drug/dosa/cell_line combination

    """
    control = pd.DataFrame(control_adata.X.toarray(), columns=control_adata.var_names)
    control["cell_type"] = control_adata.obs["cell_type"].values

    baseline_prediction_mean = control.groupby(["cell_type"], observed=True).mean().T

    r2_by_key = {}

    for key in target_adata.obs["cov_drug_dose_name"].unique():
        target_sub = target_adata[
            target_adata.obs["cov_drug_dose_name"] == key
        ].X.toarray()
        target_sub_mean = target_sub.mean(0)

        cell_type = key.split("_")[0]
        baseline_prediction_sub = baseline_prediction_mean[cell_type]
        if limit_genes:
            baseline_prediction_sub = baseline_prediction_sub[:limit_genes]
            target_sub_mean = target_sub_mean[:limit_genes]

        r2_by_key[key] = r2_score(target_sub_mean, baseline_prediction_sub)

    return pd.DataFrame.from_dict(r2_by_key, orient="index", columns=["r2"])


def calculate_baseline_all_zero_results(target_adata, limit_genes=None):
    """
    Calculate a baseline for r2 metric per drug/dose/cell_line group, by predicting zero for all genes.

    Args:
    ----
    target_adata (AnnData): RNAExpression data to evalaute baseline for.
    limit_genes (None|int): number of genes used (None to use all)

    Returns:
    -------
        pd.DataFrame: r2 per drug/dosa/cell_line combination

    """
    zero_prediction_mean = np.zeros(target_adata.shape[1])

    r2_by_key = {}

    for key in target_adata.obs["cov_drug_dose_name"].unique():
        target_sub = target_adata[
            target_adata.obs["cov_drug_dose_name"] == key
        ].X.toarray()
        target_sub_mean = target_sub.mean(0)
        zero_prediction_mean = np.zeros(len(target_sub_mean))
        if limit_genes:
            zero_prediction_mean = zero_prediction_mean[:limit_genes]
            target_sub_mean = target_sub_mean[:limit_genes]

        r2_by_key[key] = r2_score(target_sub_mean, zero_prediction_mean)

    return pd.DataFrame.from_dict(r2_by_key, orient="index", columns=["r2"])


def assemble_model_results(model_results, clamp_negative=False):
    """
    Unify a set of dictionaries holding r2 results by cell_type, drug, dosage combination into a single dataframe.
    The dataframe is later used as input to the plot, in a similar way to what's done in BioLord paper.

    Args:
    ----
    model_results (dict(str:dict(str, float))): dictionary of results per model name. For each model_name the value is a dictionary of gropu names and their aggregated r2 score
    clamp_negative(bool): wether to clamp netagive r2 values to zero.

    Returns:
    -------
        pd.DataFrame: a dataframe with all models results with columns "cell_line", "drug", "dose", "combination", "r2"
    """
    results_dfs = []
    for model_name, df in model_results.items():
        df["model"] = model_name
        results_dfs.append(df)

    unified_results = pd.concat(results_dfs)

    if clamp_negative:
        unified_results["r2"] = unified_results["r2"].apply(lambda x: max(x, 0))

    unified_results["cell_line"] = (
        pd.Series(unified_results.index.values).apply(lambda x: x.split("_")[0]).values
    )
    unified_results["drug"] = (
        pd.Series(unified_results.index.values).apply(lambda x: x.split("_")[1]).values
    )
    unified_results["dose"] = (
        pd.Series(unified_results.index.values).apply(lambda x: x.split("_")[2]).values
    )
    unified_results["dose"] = unified_results["dose"].astype(float)
    unified_results["dose"] *= 10  # units should be micro molar

    unified_results["combination"] = unified_results.index.values
    unified_results = unified_results.reset_index()
    return unified_results


def plot_cell_drug_response_model_results(model_results, filename=None):
    """
    Box plot cell drug reponse model results, consisting of r2 metric value per group of <cell_line, drug, dosage>.
    Each such group r2 is a dot in the box plot, and the boxes are aggregate per dosage. So we can see the
    overall performance of the model in each dosage group.
    The input model results may consist of several model results, and so this plot evnable compare the model performamce.
    The plot is similar to the main plot in ChemCPA and BioLord paper.

    :model_results (pd.DataFrame) - results per model name. columns "cell_line", "drug", "dose", "combination", "r2"
    """
    fig, axs = plt.subplots(1, 1, figsize=(8, 4))

    # hue_order = ["biolord", "chemCPA-pre", "chemCPA", "baseline"]
    # pallete = ["#339AB8", "#9AB833", "#b89433", "#b85133"]
    pallete = sns.color_palette("Set2")
    pallete = pallete[-model_results["model"].nunique() :]
    sns.boxplot(
        data=model_results,
        x="dose",
        y="r2",
        hue="model",
        # hue_order=hue_order,
        palette=pallete,
        widths=0.15,
        ax=axs,
    )

    axs.set_ylabel("$E[r^2]$", fontsize=12)
    axs.set_xlabel(r"dosage in $\mu$M", fontsize=12)
    axs.set_axisbelow(True)
    axs.grid(".", color="darkgrey", axis="y")

    axs.legend(
        title="",
        fontsize=8,
        title_fontsize=14,
        loc="center left",
        frameon=True,
        bbox_to_anchor=(1, 0.5),
    )
    plt.subplots_adjust(right=0.75)
    plt.title("Average R2 scores of drug-celltype-dosage groups")

    plt.tight_layout()
    if filename:
        plt.savefig(filename, format="png", dpi=300)
    return fig


def plot_cell_drug_response_model_results_scatter(
    model_results, model_a, model_b, model_name_column, metric_column
):
    """
    Scatter plot of two cell drug reponse model results, consisting of metric values for some subsets of the data (eg r2 per group of <cell_line, drug, dosage>).
    For each subset the two models metric value are represented bty a dot in the scatter plot.

    :model_results (pd.DataFrame) - metric results per model name.
    :model_a(str): first model name
    :model_b(str): second model_name
    :model_name_column: model_name column in model_results df
    :metric_column: metric column in model_results df
    """
    results = pd.concat(
        [
            model_results[model_results[model_name_column] == model_a][metric_column],
            model_results[model_results[model_name_column] == model_b][metric_column],
        ],
        axis=1,
    )
    results.columns = [model_a, model_b]
    fig = make_predictions_gt_density_plot(
        predictions_df=results,
        predicted_label=model_a,
        gt_label=model_b,
        kind="scatter",
    )
    return fig


def calc_control_mean_expression():
    """
    Calculate mean expression of control cells in each cell_line to be used as input during training.
    Saves to the Sciplex folder.
    """
    import scanpy as sc

    source_h5ad_file_name = (
        Path(os.environ["BMFM_TARGETS_SCIPLEX3_DATA"]) / "sciplex3_biolord.h5ad"
    )
    adata = sc.read(source_h5ad_file_name)
    baseline_train_adata = adata[
        adata.obs["split_ood"] != "ood"
    ]  # just in case, altough control cells are not part of the test set.
    control_adata = baseline_train_adata[
        baseline_train_adata.obs["product_name"] == "control", :
    ]

    control = pd.DataFrame(control_adata.X.toarray(), columns=control_adata.var_names)
    control["cell_type"] = control_adata.obs["cell_type"].values

    baseline_prediction_mean = control.groupby(["cell_type"]).mean()
    baseline_prediction_mean.to_csv(
        Path(os.environ["BMFM_TARGETS_SCIPLEX3_DATA"]) / "sciplex_biolord.control.csv"
    )


def create_mean_expression_h5ad(source_h5ad_file_path, output_file_path):
    """
    Calculate mean expression of cells in each cell_line x durg x dosage group, to be used as a alternative input file
    more suitable for the r2 aggregated metric
    Saves as h5ad file to the Sciplex folder.
    Intened to run onces in offline, to create an input file to experiments on the mean data.

    :source_h5ad_file_path(str) - gene expression input data file, expecting a 'cov_drug_dose_name' observation.
    :output_file_path(str) - path to save the output avaraged data h5ad.
    """
    import scanpy as sc

    adata = sc.read(source_h5ad_file_path)
    drug_name_to_smiles_mapping = (
        adata.obs.groupby("product_name")["SMILES"].first().to_dict()
    )

    aggregated_adata = {}

    for split in ["test", "ood", "train"]:
        split_adata = adata[adata.obs["split_rep_ood"] == split]

        groups = split_adata.obs["cov_drug_dose_name"].unique().tolist()

        # Calculate means for each group, do it using a loop to maintain the order
        # group_means = np.zeros((len(groups), split_adata.n_vars))
        group_means = []
        for group in groups:
            group_means.append(
                split_adata.X[split_adata.obs["cov_drug_dose_name"] == group].mean(
                    axis=0
                )
            )

        grouped_adata = sc.AnnData(
            X=np.array(np.vstack(group_means)),
            var=split_adata.var,
        )
        grouped_adata.obs["cov_drug_dose_name"] = groups
        grouped_adata.obs["cell_type"] = (
            grouped_adata.obs["cov_drug_dose_name"].str.split("_").str[0]
        )
        grouped_adata.obs["product_name"] = (
            grouped_adata.obs["cov_drug_dose_name"].str.split("_").str[1]
        )
        grouped_adata.obs["dose"] = (
            grouped_adata.obs["cov_drug_dose_name"].str.split("_").str[2]
        )
        grouped_adata.obs["SMILES"] = grouped_adata.obs["product_name"].map(
            drug_name_to_smiles_mapping
        )

        # convert dose to mM to match original "dose" column
        grouped_adata.obs["dose"] = pd.to_numeric(grouped_adata.obs["dose"]) * 10000

        aggregated_adata[split] = grouped_adata

    combined_adata = sc.concat(
        [aggregated_adata["train"], aggregated_adata["test"], aggregated_adata["ood"]],
        join="outer",
        label="split_rep_ood",
        keys=["train", "test", "ood"],
    )

    combined_adata.write(output_file_path)

    # sc.pp.highly_variable_genes(combined_adata)
    # top_genes = adata.var_names[combined_adata.var['highly_variable']][:20]
    # sc.pl.heatmap(combined_adata, var_names=top_genes, groupby='cov_drug_dose_name', cmap='viridis', show=True, save="heatmap.png")


def cell_drug_response_aggregated_r2_scores(
    model_results, limit_genes, output_filename
):
    """
    Compute median and mean per dosage of group (<cell_line, drug, dosage>) r2 scores for a set of models
    :model_results (pd.DataFrame) -  models results with columns "cell_line", "drug", "dose", "combination", "r2"
    :output_filename (str | None) - optional, file name to write results to.
    """
    res = pd.DataFrame()
    grouped = model_results.groupby(["model", "dose"])
    res = (
        grouped["r2"]
        .agg(
            median="median",
            mean="mean",
            mean_clamped=lambda x: x.apply(lambda val: max(val, 0)).mean(),
        )
        .reset_index()
    )
    res["genes"] = res["model"].apply(
        lambda model: 500 if "500" in model else 2000
    )  # todo: better to pass the real gene limit

    #  Add BioLord results
    biolord = {
        "model": "BioLord (reproducability notebook)",
        "genes": 2000,
        "dose": 10,
        "median": 0.85,
        "mean": None,
        "mean_clamped": 0.76,
    }
    res = pd.concat([res, pd.DataFrame([biolord])], ignore_index=True)

    res = res[["model", "genes", "dose", "median", "mean", "mean_clamped"]]
    res = res.sort_values(["genes", "dose", "model"])
    res = res.round(decimals=4)
    if output_filename:
        res.to_csv(output_filename, index=False)
    return res


if __name__ == "__main__":
    source_h5ad_file_path = (
        Path(os.environ["BMFM_TARGETS_SCIPLEX3_DATA"])
        / "sciplex3_biolord_ibm_rep_validation.h5ad"
    )
    output_file_path = (
        Path(os.environ["BMFM_TARGETS_SCIPLEX3_DATA"])
        / "test_sciplex3_biolord_ibm_rep_validation_mean.h5ad"
    )
    create_mean_expression_h5ad(source_h5ad_file_path, output_file_path)
