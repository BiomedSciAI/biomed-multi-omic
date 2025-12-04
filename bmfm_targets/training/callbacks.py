import logging
import os
import pathlib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import transformers
from clearml.logger import Logger
from lightning_utilities.core.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import types as pl_types
from scipy.sparse import csr_matrix

from bmfm_targets.datasets.datasets_utils import (
    guess_if_raw,
    make_group_means,
    random_subsampling,
)
from bmfm_targets.training.metrics import perturbation_metrics as pm

logger = logging.getLogger(__name__)


class BatchSizeScheduler(pl.Callback):
    def __init__(
        self,
        schedule: list[dict[str, int]] | None = None,
        test_batch_size: int | None = None,
        test_max_length: int | None = None,
    ) -> None:
        super().__init__()
        self.schedule = schedule
        self._custom_epochs = False
        self._schedule_expanded = False
        self.test_batch_size = test_batch_size
        self.test_max_length = test_max_length

    def setup(self, trainer, pl_module, stage):
        if stage == "test":
            trainer.datamodule.max_length = self.test_max_length
            trainer.datamodule.batch_size = self.test_batch_size

    def _check_schedule(self, max_epochs: int):
        if max_epochs < 1:
            raise ValueError(
                "To use batch size scheduler you must train for more than one epoch."
            )

        n_epochs_schedule = []
        for i in range(len(self.schedule)):
            if not self.schedule[i].get("n_epochs"):
                self.schedule[i]["n_epochs"] = 1
            n_epochs_schedule.append(self.schedule[i]["n_epochs"])
        n_epochs_schedule = sum(n_epochs_schedule)

        if n_epochs_schedule > len(self.schedule):
            self._custom_epochs = True

        if n_epochs_schedule != max_epochs and self._custom_epochs:
            raise ValueError(
                "When defining a custom schedule, you must ensure the total `n_epochs` sums to `max_epochs`."
            )

        if n_epochs_schedule > max_epochs:
            raise ValueError(
                "Total epochs given in the BatchSizeScheduler bigger than total epochs."
            )

    def _expand_schedule(self, epochs: int) -> None:
        self._check_schedule(epochs)

        if not self._custom_epochs and epochs > len(self.schedule):
            q, r = divmod(epochs, len(self.schedule))
            repeats = [q + (i < r) for i in range(len(self.schedule))]
        else:
            repeats = [s["n_epochs"] for s in self.schedule]

        _schedule = [
            (s["max_length"], s["batch_size"])
            for s, r in zip(self.schedule, repeats)
            for _ in range(r)
        ]

        self.schedule = _schedule
        self._schedule_expanded = True

    def _apply_schedule(self, trainer: pl.Trainer, schedule_idx: int) -> None:
        max_length, batch_size = self.schedule[schedule_idx]
        trainer.datamodule.max_length = max_length
        trainer.datamodule.batch_size = batch_size

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._expand_schedule(trainer.max_epochs)
        self._apply_schedule(trainer, trainer.current_epoch)

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        desired_max_len, desired_batch_size = self.schedule[trainer.current_epoch]
        actual_max_length = getattr(trainer.datamodule, "max_length", None)
        actual_batch_size = getattr(trainer.datamodule, "batch_size", None)

        if (
            actual_max_length != desired_max_len
            or actual_batch_size != desired_batch_size
        ):
            self._apply_schedule(trainer, trainer.current_epoch)
            actual_max_length = getattr(trainer.datamodule, "max_length", None)
            actual_batch_size = getattr(trainer.datamodule, "batch_size", None)

        pl_module.log_dict(
            {"max_length": actual_max_length, "batch_size": actual_batch_size},
            prog_bar=False,
        )


class SavePretrainedModelCallback(pl.Callback):
    def __init__(
        self,
        save_dir: pathlib.Path,
        tokenizer: transformers.PreTrainedTokenizer,
        epoch_period: int | None = 1,
        step_period: int | None = None,
    ):
        self.epoch_period = epoch_period
        self.step_period = step_period
        self.save_dir = save_dir
        self.tokenizer = tokenizer

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module,
        outputs: pl_types.STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ):
        step = trainer.global_step
        if self.step_period is not None and step % self.step_period == 0:
            step_save_dir = self.save_dir / f"step_{step}"
            pl_module.save_transformer(step_save_dir, self.tokenizer)

    @rank_zero_only
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module):
        epoch = trainer.current_epoch
        if self.epoch_period is not None and epoch % self.epoch_period == 0:
            epoch_save_dir = self.save_dir / f"epoch_{trainer.current_epoch}"
            pl_module.save_transformer(epoch_save_dir, self.tokenizer)


class InitialCheckpoint(ModelCheckpoint):
    def __init__(self, dirpath, filename="initial.ckpt"):
        super().__init__(dirpath=dirpath, filename=filename, save_top_k=0)
        self.filename = filename

    def on_train_start(self, trainer, pl_module):
        logger.info("saving initial embedding")
        trainer.save_checkpoint(self.dirpath + "/" + self.filename)


class BatchIntegrationCallback(pl.Callback):
    def __init__(
        self,
        batch_column_name=None,
        counts_column_name=None,
        benchmarking_methods=[
            "Unintegrated",
            "Scanorama",
            "LIGER",
            "Harmony",
        ],
    ):
        super().__init__()
        self.batch_column_name = batch_column_name
        self.counts_column_name = counts_column_name
        self.benchmarking_methods = benchmarking_methods

    def on_predict_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.cl = Logger.current_logger()
        if self.cl:
            self.execute_batch_integration(trainer)

    def execute_batch_integration(self, trainer):
        adata_emb = self.get_adata_with_embeddings(trainer)
        self.batch_column_name = self.verify_batch_column_name(trainer)
        self.report_batch_integartion_to_clearml(adata_emb)

    def verify_batch_column_name(self, trainer):
        self.target_column_name = trainer.datamodule.label_columns[0].label_column_name
        if self.batch_column_name is None:
            return self.target_column_name
        else:
            return self.batch_column_name

    def get_adata_with_embeddings(self, trainer):
        batch_preds = trainer.predict_loop.predictions

        def _join_batches(k):
            return np.concatenate([d[k] for d in batch_preds], axis=0)

        predictions = {k: _join_batches(k) for k in batch_preds[0].keys()}
        adata_orig = trainer.datamodule.predict_dataset.processed_data
        adata_emb = self.add_embed_to_obsm(adata_orig, predictions)
        if not self.batch_column_name == "batch":
            adata_emb.obs["batch"] = adata_emb.obs[self.batch_column_name]
        adata_emb.obs["batch"] = adata_emb.obs["batch"].astype("category")
        return adata_emb

    def report_batch_integartion_to_clearml(self, adata_emb):
        batch_int_df = self.generate_table_batch_integration(adata_emb)
        self.cl.report_table(
            title="Batch Integration",
            series="Batch Integration",
            table_plot=batch_int_df.T,
        )
        self.cl.report_single_value(
            name="Average Bio",
            value=float(batch_int_df.loc[:, "Avg_bio"]),
        )
        self.cl.report_single_value(
            name="Average Batch",
            value=float(batch_int_df.loc[:, "Avg_batch"]),
        )

        fig = self.generate_fig_batch_integration(adata_emb)
        self.cl.report_matplotlib_figure(
            title="UMAP Visualization",
            series="umap_plot",
            figure=fig,
            report_image=True,
        )
        plt.close(fig)

        fig = self.generate_pretty_benchmarking_table(adata_emb)
        self.cl.report_matplotlib_figure(
            title="Integration Benchmark",
            series="scIB Summary",
            figure=fig,
            report_image=True,
        )

    def generate_fig_batch_integration(self, adata_emb):
        target_col = self.target_column_name
        batch_col = self.batch_column_name
        counts_col = self.counts_column_name
        sampling_adata_emb = random_subsampling(
            adata=adata_emb,
            n_samples=min((10000, adata_emb.obs.shape[0])),
            shuffle=False,
        )
        sc.pp.neighbors(sampling_adata_emb, use_rep="BMFM-RNA")
        sc.tl.umap(sampling_adata_emb)
        sampling_adata_emb.obs[batch_col] = sampling_adata_emb.obs[batch_col].astype(
            "category"
        )
        colors = [target_col, batch_col]
        titles = [
            f"Targets embeddings: {target_col} ",
            f"Batch embeddings: {batch_col}",
        ]
        if counts_col in sampling_adata_emb.obs.columns:
            colors.append(counts_col)
            titles.append("Embeddings colored by total counts per cell")
        else:
            logger.warning(
                f"{counts_col} not found in obs. Available columns: {sampling_adata_emb.obs.columns}"
            )
        fig, axs = plt.subplots(len(colors), 1, figsize=(15, 15))
        for i, ax in enumerate(axs):
            sc.pl.umap(
                sampling_adata_emb,
                color=colors[i],
                frameon=False,
                title=titles[i],
                ax=ax,
                show=False,
            )
        plt.tight_layout()
        return fig

    def generate_table_batch_integration(self, adata_emb):
        batch_col = self.batch_column_name
        label_col = self.target_column_name
        sc.pp.neighbors(adata_emb, use_rep="BMFM-RNA")
        sc.tl.umap(adata_emb)
        import scib.metrics.metrics as scm

        batch_int = scm(
            adata_emb,
            adata_int=adata_emb,
            batch_key=f"{batch_col}",
            label_key=f"{label_col}",
            embed="BMFM-RNA",
            isolated_labels_asw_=False,
            silhouette_=True,
            hvg_score_=False,
            graph_conn_=True,
            pcr_=False,
            isolated_labels_f1_=False,
            trajectory_=False,
            nmi_=True,
            ari_=True,
            cell_cycle_=False,
            kBET_=False,
            ilisi_=False,
            clisi_=False,
        )
        batch_int_dict = batch_int[0].to_dict()

        batch_int_dict["avg_bio"] = np.mean(
            [
                batch_int_dict["NMI_cluster/label"],
                batch_int_dict["ARI_cluster/label"],
                batch_int_dict["ASW_label"],
            ]
        )

        batch_int_dict["avg_batch"] = np.mean(
            [
                batch_int_dict["graph_conn"],
                batch_int_dict["ASW_label/batch"],
            ]
        )
        batch_int_dict = {k: v for k, v in batch_int_dict.items() if not np.isnan(v)}
        batch_int_df = pd.DataFrame(
            {k.capitalize(): [np.round(v, 2)] for k, v in batch_int_dict.items()}
        )
        batch_int_df = batch_int_df.rename(
            columns={
                "Nmi_cluster/label": f"NMI_cluster_by_{label_col}_(bio)",
                "Ari_cluster/label": f"ARI_cluster_by{label_col}_(bio)",
                "Asw_label": f"ASW_by_{label_col}_(bio)",
                "Graph_conn": f"graph_conn_by_{batch_col}_(batch)",
                "Asw_label/batch": f"ASW_by_{batch_col}_(batch)",
            }
        )
        return batch_int_df

    def add_embed_to_obsm(self, adata, results):
        adata_emb = adata.copy()
        embeddings = results["embeddings"]

        adata_cell_names = adata_emb.obs_names.values
        dict_cell_names = results["cell_name"]
        name_to_index = {name: idx for idx, name in enumerate(dict_cell_names)}

        aligned_embeddings = np.array(
            [embeddings[name_to_index[name]] for name in adata_cell_names]
        )
        adata_emb.obsm["BMFM-RNA"] = aligned_embeddings
        return adata_emb

    def generate_pretty_benchmarking_table(self, adata_emb):
        from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation

        biocons = BioConservation(isolated_labels=False)
        logger.info("Beginning Unintegrated...")
        adata_emb.obsm["Unintegrated"] = self.get_pca_of_x(adata_emb)
        logger.info("Beginning Harmony...")
        adata_emb.obsm["Harmony"] = self.harmony_emb(adata_emb)
        logger.info("Beginning Scanorama...")
        adata_emb.obsm["Scanorama"] = self.scanorama_emb(adata_emb)
        logger.info("Beginning LIGER...")
        adata_emb.obsm["LIGER"] = self.liger_emb(adata_emb)

        bm = Benchmarker(
            adata_emb,
            batch_key=self.batch_column_name,
            label_key=self.target_column_name,
            embedding_obsm_keys=["BMFM-RNA"] + self.benchmarking_methods,
            pre_integrated_embedding_obsm_key="Unintegrated",
            bio_conservation_metrics=biocons,
            batch_correction_metrics=BatchCorrection(),
            n_jobs=-1,
        )
        bm.prepare()
        bm.benchmark()
        fig = bm.plot_results_table(min_max_scale=False)
        return fig

    def harmony_emb(self, adata):
        from harmony import harmonize

        if not "Harmony" in self.benchmarking_methods:
            return np.zeros((adata.n_obs, 1))
        if "Unintegrated" not in adata.obsm:
            adata.obsm["X_pca"] = self.get_pca_of_x(adata)
        try:
            return harmonize(
                adata.obsm["X_pca"], adata.obs, batch_key=self.batch_column_name
            )
        except:
            return np.zeros((adata.n_obs, 1))

    def get_pca_of_x(self, adata_orig: sc.AnnData, flavor="cell_ranger"):
        """
        Calculate PCA of X.

        This function produces a valid PCA of the initial data whether it is already log
        normed, raw counts or lognormed and binned. It makes use of HVG to reduce the prePCA
        space to 2000 genes. This too is sensitive to whether the data is lognormed or not.
        It detects the kind of data via a detection heuristic and treats it accordingly.
        It flags the data as raw and applies the lognorm before PCA if at least 4 of these
        6 criteria are met:
         - integer
         - max > 50
         - >40% ones
         - mean_val < 2.5
         - median val <= 1
         - >60% one two or three

        It does all of the rescaling and transforming on a copy of the anndata, injecting just
        the PCA into the original anndata to preserve the data integrity.
        """
        adata = adata_orig.copy()
        looks_raw = guess_if_raw(adata.X.data)

        if looks_raw:
            logger.info("Detected raw counts — applying normalization and log1p.")
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        else:
            logger.info(
                "Detected log1p-transformed or binned input — skipping normalization."
            )
        adata.obs[self.batch_column_name] = adata.obs[self.batch_column_name].astype(
            "category"
        )
        try:
            sc.pp.highly_variable_genes(
                adata, flavor=flavor, batch_key=self.batch_column_name, n_top_genes=2000
            )
        except:
            logger.warning(
                "Batch level HVG calc failed, reverting to batch insensitive"
            )
            sc.pp.highly_variable_genes(adata, flavor=flavor, n_top_genes=2000)

        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, svd_solver="arpack", n_comps=30, mask_var="highly_variable")
        adata_orig.obsm["X_pca"] = adata.obsm["X_pca"]
        return adata_orig.obsm["X_pca"]

    def scanorama_emb(self, adata):
        import scanorama

        if not "Scanorama" in self.benchmarking_methods:
            return np.zeros((adata.n_obs, 1))
        try:
            batch_cats = adata.obs.batch.cat.categories
            adata_list = [adata[adata.obs.batch == b].copy() for b in batch_cats]
            scanorama.integrate_scanpy(adata_list)

            adata.obsm["Scanorama"] = np.zeros(
                (adata.shape[0], adata_list[0].obsm["X_scanorama"].shape[1])
            )
            for i, b in enumerate(batch_cats):
                adata.obsm["Scanorama"][adata.obs.batch == b] = adata_list[i].obsm[
                    "X_scanorama"
                ]

            return adata.obsm["Scanorama"]
        except:
            return np.zeros((adata.n_obs, 1))

    def liger_emb(self, adata):
        import pyliger

        k = min(adata.obs["batch"].value_counts().min() - 1, 10)

        if not "LIGER" in self.benchmarking_methods or k < 1:
            return np.zeros((adata.n_obs, 1))
        try:
            batch_cats = adata.obs.batch.cat.categories
            bdata = adata.copy()
            adata_list = [bdata[bdata.obs.batch == b].copy() for b in batch_cats]
            for i, ad in enumerate(adata_list):
                ad.uns["sample_name"] = batch_cats[i]
                ad.uns["var_gene_idx"] = np.arange(bdata.n_vars)

            liger_data = pyliger.create_liger(
                adata_list, remove_missing=False, make_sparse=False
            )

            liger_data.var_genes = bdata.var_names
            pyliger.normalize(liger_data)
            pyliger.scale_not_center(liger_data)
            pyliger.optimize_ALS(liger_data, k=k)
            pyliger.quantile_norm(liger_data)

            bdata.obsm["LIGER"] = np.zeros(
                (adata.shape[0], liger_data.adata_list[0].obsm["H_norm"].shape[1])
            )
            for i, b in enumerate(batch_cats):
                bdata.obsm["LIGER"][adata.obs.batch == b] = liger_data.adata_list[
                    i
                ].obsm["H_norm"]

            return bdata.obsm["LIGER"]
        except:
            return np.zeros((adata.n_obs, 1))


class SampleLevelLossCallback(pl.Callback):
    """
    Callback to calculate sample level loss metrics at the end of testing.

    Requires trainer.batch_prediction_behavior to be set to "track" or "dump".

    Works with MSE and BCE losses, currently assumes that the MSE loss ignores zero values.
    """

    def __init__(self, metric_key: str = "expressions_non_input_genes"):
        self.metric_key = metric_key
        self.output_file_name = None

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.metric_key not in pl_module.prediction_df:
            logger.warning(
                f"Metric {self.metric_key} not found in prediction_df, skipping sample level loss calculation."
            )
            return
        pred_df = pl_module.prediction_df[self.metric_key].copy()
        sample_loss_components = []
        if "logits_expressions_mse" in pred_df.columns:
            pred_df = pred_df.assign(
                error=(pred_df["logits_expressions_mse"] - pred_df["label_expressions"])
                ** 2
            )
            sample_mse = (
                pred_df.query("label_expressions > 0")
                .groupby(level=0)
                .error.mean()
                .rename("sample_mse")
            )
            sample_loss_components.append(sample_mse)

        if "logits_expressions_is_zero_bce" in pred_df.columns:
            import torch
            from torch.nn.functional import binary_cross_entropy_with_logits

            bce_loss = binary_cross_entropy_with_logits(
                torch.tensor(pred_df["logits_expressions_is_zero_bce"].values),
                torch.tensor(
                    pred_df["label_expressions"].values == 0, dtype=torch.float
                ),
                reduction="none",
            ).numpy()

            pred_df = pred_df.assign(bce_loss=bce_loss)
            sample_bce = (
                pred_df.groupby(level=0).bce_loss.mean().rename("sample_is_zero_bce")
            )
            sample_loss_components.append(sample_bce)

        if sample_loss_components:
            sample_loss = pd.concat(sample_loss_components, axis=1)
            sample_loss.assign(
                sample_mean_loss=sample_loss.mean(axis=1),
                sample_sum_loss=sample_loss.sum(axis=1),
            )

        output_file_name = pathlib.Path(trainer.log_dir) / "sample_level_loss.csv"

        if self.output_file_name is not None:
            output_file_name = pathlib.Path(self.output_file_name)

        sample_loss.to_csv(output_file_name)

        return super().on_test_end(trainer, pl_module)


class SavePredictionsH5ADCallback(pl.Callback):
    def __init__(
        self,
        output_file_name=None,
        train_h5ad_file=None,
        split_column_name: str | None = None,
        perturbation_column_name="target_gene",
        control_name: str | None = "non-targeting",
        predictions_key: str = "label_expressions",
    ):
        super().__init__()
        self.output_file_name = output_file_name
        self.train_data = None
        self.perturbation_column_name = perturbation_column_name
        self.control_name = control_name
        self.predictions_key = predictions_key
        if train_h5ad_file:
            self.train_data = sc.read_h5ad(train_h5ad_file)
            logger.info(f"train_data size: {self.train_data.shape}")

            group_means = make_group_means(
                self.train_data, perturbation_column_name, split_column_name
            )
            logger.info(f"train_group_means size: {group_means.shape}")

            self.grouped_ground_truth = pm.get_grouped_ground_truth(
                group_means, remove_always_zero=False
            )

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._save_predictions_h5ad(trainer, pl_module)

    def _save_predictions_h5ad(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        output_file = pathlib.Path(trainer.log_dir) / "predictions.h5ad"
        if self.output_file_name is not None:
            output_file = pathlib.Path(self.output_file_name)

        preds_df = pl_module.prediction_df[self.predictions_key]
        assert isinstance(preds_df, pd.DataFrame)
        logger.info(f"predictions_df size: {preds_df.shape}")

        grouped_predictions = pm.get_grouped_predictions(
            preds_df, self.grouped_ground_truth
        )
        preds_df = ensure_predicted_expressions_in_preds_df(
            preds_df, self.grouped_ground_truth
        )
        adata = create_adata_from_predictions_df(
            preds_df, grouped_predictions, self.perturbation_column_name
        )
        logger.info(f"anndata shape: {adata.shape}")
        if self.train_data:
            adata = add_control_samples(
                adata, self.train_data, self.perturbation_column_name, self.control_name
            )
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        adata.write(output_file)
        logger.info(f"saved completed predictions to {output_file}")


def ensure_predicted_expressions_in_preds_df(preds_df, grouped_ground_truth):
    if "predicted_expressions" in preds_df.columns:
        return preds_df
    if "predicted_delta_baseline_expressions" in preds_df.columns:
        avgpert = grouped_ground_truth["Average_Perturbation_Train"]
        predictions = (
            avgpert.loc[preds_df["input_genes"]].to_numpy()
            + preds_df["predicted_delta_baseline_expressions"].to_numpy()
        )
        preds_df["predicted_expressions"] = predictions
    return preds_df


def create_adata_from_predictions_df(
    preds_df, grouped_predicted_expressions, perturbation_col="target_gene"
):
    """
    Create AnnData object from predictions DataFrame.

    preds_df (pd.DataFrame): a long format matrix of gene predictions with columns
    [input_genes, predicted_expressions, perturbed_genes] where index is sample ids.
    grouped_predicted_expressions (pd.DataFrame): the mean of predicted expression for each gene in each perturbation, or baseline mean in case that gene
    was not predicted at all in that perturbation. These values are used to infill samples in which some genes were not predicted.
    perturbation_col (str): name of perturbation column in the dataset
    """
    # Get all unique samples and genes
    all_samples = preds_df.index.unique().sort_values()
    all_genes = (
        grouped_predicted_expressions.index.get_level_values(1).unique().sort_values()
    )
    n_samples = len(all_samples)
    n_genes = len(all_genes)
    logger.info(
        f"in create_adata_from_predictions_df. n_samples from perds_df: {n_samples}"
    )
    logger.info(f"n_genes from grouped_predicted_expressions: {n_genes}")

    # Create sample to perturbation mapping
    sample_to_pert = (
        preds_df.reset_index()[["sample_id", "perturbed_genes"]]
        .drop_duplicates()
        .set_index("sample_id")["perturbed_genes"]
    )

    # Initialize dense matrix with pseudobulk values
    # First, create a mapping of perturbation to pseudobulk expression
    pert_to_pseudobulk = {}
    for pert in sample_to_pert.unique():
        pert_to_pseudobulk[pert] = grouped_predicted_expressions.loc[
            pert, "predicted_expressions"
        ]

    # Create gene index mapping for fast lookup
    gene_to_idx = {gene: idx for idx, gene in enumerate(all_genes)}
    sample_to_idx = {sample: idx for idx, sample in enumerate(all_samples)}

    # Initialize matrix with pseudobulk values
    X = np.zeros((n_samples, n_genes))
    for sample_idx, sample in enumerate(all_samples):
        pert = sample_to_pert[sample]
        pseudobulk = pert_to_pseudobulk[pert]
        # Fill in pseudobulk values
        for gene, value in pseudobulk.items():
            if gene in gene_to_idx:
                X[sample_idx, gene_to_idx[gene]] = value

    # Overwrite with specific predictions from preds_df
    # Vectorized approach: get all row/col indices at once
    row_indices = [sample_to_idx[sample] for sample in preds_df.index]
    col_indices = [gene_to_idx[gene] for gene in preds_df["input_genes"]]
    values = preds_df["predicted_expressions"].values
    logger.info(f"row_indices: {len(row_indices)}, col_indices {len(col_indices)}")

    # Assign all values at once
    X[row_indices, col_indices] = values

    # Create AnnData object
    adata = sc.AnnData(
        X=X, obs=pd.DataFrame(index=all_samples), var=pd.DataFrame(index=all_genes)
    )

    adata.X = csr_matrix(adata.X)
    # Add perturbation metadata
    adata.obs[perturbation_col] = sample_to_pert
    logger.info(f"generated adata size: {adata.shape}")

    # Assertions
    assert adata.shape[0] == preds_df.index.nunique()
    assert (
        adata.shape[1]
        == grouped_predicted_expressions.index.get_level_values(1).unique().shape[0]
    )

    return adata


def add_control_samples(
    adata: sc.AnnData,
    train_adata: sc.AnnData,
    perturbation_column_name="target_gene",
    control_name="non-targeting",
) -> sc.AnnData:
    """
    Safely adds non-targeting control cells from train_adata to adata.

    This function treats the gene order of the non-targeting control cells
    as the "master" list. The other AnnData object (`adata`) will be
    re-indexed to match this master gene order before concatenation.

    Args:
    ----
        adata: The AnnData object to be conformed and used in the concatenation.
        train_adata: The AnnData object containing the master non-targeting cells.

    Returns:
    -------
        A new AnnData object with the combined data, with a gene order
        matching that of the non-targeting controls.

    """
    adata_control = train_adata[
        train_adata.obs[perturbation_column_name] == "Control"
    ].copy()
    adata_control.obs[perturbation_column_name] = control_name
    # logger.warning(
    #     f"In adata but not in adata_control: {set(adata.var_names) - set(adata_control.var_names)}"
    # )
    # logger.warning(
    #     f"In adata_control but not in adata: {set(adata_control.var_names) - set(adata.var_names)}"
    # )
    assert {*adata.var_names} == {*adata_control.var_names}

    adata = adata[:, adata_control.var_names]

    adata_combined = sc.concat(
        [adata, adata_control],
        axis=0,
        join="outer",  # With aligned vars, 'outer' and 'inner' are equivalent for genes
        merge="same",
    )

    return adata_combined
