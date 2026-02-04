import logging
import os
import pathlib
from collections.abc import Iterable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import transformers
from clearml.logger import Logger
from czbenchmarks.tasks import (
    MetadataLabelPredictionTask,
    MetadataLabelPredictionTaskInput,
)
from lightning_utilities.core.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import types as pl_types
from scipy.sparse import csr_matrix

from bmfm_targets.datasets.datasets_utils import make_group_means, random_subsampling
from bmfm_targets.training.metrics import perturbation_metrics as pm

logger = logging.getLogger(__name__)


def is_valid_embedding(arr: np.ndarray | None) -> bool:
    """Check if embedding array is valid for benchmarking."""
    if arr is None:
        return False
    if not isinstance(arr, np.ndarray):
        return False
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return False
    if np.all(arr == 0):
        return False
    if np.isnan(arr).all():
        return False
    return True


def discover_baseline_embeddings(
    adata: sc.AnnData, model_key: str, exclude_keys: Iterable[str]
) -> list[str]:
    """Find valid baseline embedding keys in obsm, excluding model and standard scanpy keys."""
    baselines = []
    for key in adata.obsm:
        if key == model_key or key in exclude_keys:
            continue
        if is_valid_embedding(adata.obsm[key]):
            baselines.append(key)
        else:
            logger.warning(f"Skipping invalid embedding: {key}")
    return baselines


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


MODEL_EMBED_KEY = "BMFM_RNA"


def get_adata_with_embeddings(
    trainer: pl.Trainer, batch_column_name, model_embed_key=MODEL_EMBED_KEY
) -> sc.AnnData:
    """Extract predictions and align embeddings with original AnnData."""
    predictions = extract_predictions(trainer)
    adata = trainer.datamodule.predict_dataset.processed_data
    aligned = align_embeddings(adata, predictions)
    adata_emb = adata.copy()
    adata_emb.obsm[model_embed_key] = aligned

    if batch_column_name is not None and batch_column_name in adata_emb.obs:
        if batch_column_name != "batch":
            adata_emb.obs["batch"] = adata_emb.obs[batch_column_name]
        adata_emb.obs["batch"] = adata_emb.obs["batch"].astype("category")
    return adata_emb


def extract_predictions(trainer: pl.Trainer) -> dict:
    """Collect and concatenate prediction outputs."""
    batch_preds = trainer.predict_loop.predictions
    return {
        k: np.concatenate([d[k] for d in batch_preds], axis=0) for k in batch_preds[0]
    }


def align_embeddings(adata: sc.AnnData, results: dict) -> sc.AnnData:
    """Align and add model embeddings to AnnData obsm."""
    adata_emb = adata.copy()
    embeddings = results["embeddings"]
    name_to_idx = {name: i for i, name in enumerate(results["cell_name"])}
    aligned = np.array([embeddings[name_to_idx[n]] for n in adata_emb.obs_names])
    return aligned


class BatchIntegrationCallback(pl.Callback):
    """Callback for evaluating batch integration quality after prediction."""

    # Keys to exclude when discovering baseline embeddings in obsm
    EXCLUDED_OBSM_KEYS = frozenset(
        {"X_pca", "X_umap", "X_tsne", "X_diffmap", "X_draw_graph_fr"}
    )

    def __init__(
        self,
        batch_column_name: str | None = None,
        counts_column_name: str | None = None,
        target_column_name: str | None = None,
    ):
        super().__init__()
        self.batch_column_name = batch_column_name
        self.counts_column_name = counts_column_name
        self.target_column_name = target_column_name
        self.cl: Logger | None = None

    def on_predict_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self.cl = Logger.current_logger()
        if not self.cl:
            logger.warning(
                "ClearML logger not found, skipping batch integration reporting."
            )
            return

        adata_emb = get_adata_with_embeddings(trainer, self.batch_column_name)
        if self.target_column_name is None:
            self.target_column_name = trainer.datamodule.label_columns[
                0
            ].label_column_name

        self._execute_batch_integration(adata_emb)

    def _execute_batch_integration(self, adata_emb: sc.AnnData) -> None:
        """Run all batch integration evaluations and report to ClearML."""
        # Core metrics table
        batch_int_df = self._generate_metrics_table(adata_emb)
        self.cl.report_table(
            title="Batch Integration",
            series="Batch Integration",
            table_plot=batch_int_df.T,
        )
        self.cl.report_single_value(
            name="Average Bio", value=float(batch_int_df["Avg_bio"].iloc[0])
        )
        if "Avg_batch" in batch_int_df.columns:
            self.cl.report_single_value(
                name="Average Batch", value=float(batch_int_df["Avg_batch"].iloc[0])
            )
        else:
            logger.warning("Avg_batch not found in batch integration results.")

        # UMAP visualization
        plt.close("all")  # Clear any stray figures before creating new ones
        fig = self._generate_umap_figure(adata_emb)
        self.cl.report_matplotlib_figure(
            title="UMAP Visualization",
            series="umap_plot",
            figure=fig,
            report_image=True,
        )
        plt.close(fig)

        # Benchmarking table (only if baselines exist)
        baselines = discover_baseline_embeddings(
            adata_emb, MODEL_EMBED_KEY, self.EXCLUDED_OBSM_KEYS
        )
        if baselines:
            plt.close("all")
            fig = self._generate_benchmarking_table(adata_emb, baselines)
            self.cl.report_matplotlib_figure(
                title="Integration Benchmark",
                series="scIB Summary",
                figure=fig,
                report_image=True,
            )
            plt.close(fig)
        else:
            logger.info("No baseline embeddings found, skipping comparison table.")
        plt.close("all")  # Final cleanup

    def _generate_umap_figure(self, adata_emb: sc.AnnData) -> plt.Figure:
        """Generate UMAP plots colored by target, batch, and counts."""
        sampling = random_subsampling(
            adata_emb, n_samples=min(10000, adata_emb.n_obs), shuffle=True
        )
        sc.pp.neighbors(sampling, use_rep=MODEL_EMBED_KEY)
        sc.tl.umap(sampling)

        colors, titles = [self.target_column_name], [
            f"Targets: {self.target_column_name}"
        ]

        if self.batch_column_name and self.batch_column_name in sampling.obs:
            sampling.obs[self.batch_column_name] = sampling.obs[
                self.batch_column_name
            ].astype("category")
            colors.append(self.batch_column_name)
            titles.append(f"Batch: {self.batch_column_name}")

        if self.counts_column_name and self.counts_column_name in sampling.obs:
            colors.append(self.counts_column_name)
            titles.append("Total counts per cell")

        fig, axs = plt.subplots(len(colors), 1, figsize=(12, 6 * len(colors)))
        axs = [axs] if len(colors) == 1 else axs
        for ax, col, title in zip(axs, colors, titles):
            sc.pl.umap(
                sampling,
                color=col,
                frameon=False,
                title=title,
                ax=ax,
                show=False,
                legend_loc="on data",
                legend_fontsize="x-small",
                legend_fontoutline=2,
            )
        plt.tight_layout()
        return fig

    def _generate_metrics_table(self, adata_emb: sc.AnnData) -> pd.DataFrame:
        """Compute scIB metrics for model embeddings."""
        import scib.metrics as scm
        from sklearn.metrics import (
            adjusted_rand_score,
            normalized_mutual_info_score,
            silhouette_score,
        )

        batch_col, label_col = self.batch_column_name, self.target_column_name
        sc.pp.neighbors(adata_emb, use_rep=MODEL_EMBED_KEY)

        has_batch = batch_col and batch_col in adata_emb.obs
        single_batch = (not has_batch) or (
            adata_emb.obs[batch_col].nunique(dropna=False) == 1  # noqa: PD101
        )
        results = {}

        if single_batch:
            cluster_key = "__tmp_scib_cluster"
            sc.tl.leiden(adata_emb, key_added=cluster_key, random_state=0)
            clusters = adata_emb.obs[cluster_key].to_numpy()
            labels = adata_emb.obs[label_col].to_numpy()
            X = adata_emb.obsm[MODEL_EMBED_KEY]

            mask = pd.notna(labels)
            clusters, labels, X = clusters[mask], labels[mask], X[mask]

            results["NMI_cluster/label"] = normalized_mutual_info_score(
                labels, clusters
            )
            results["ARI_cluster/label"] = adjusted_rand_score(labels, clusters)
            vc = pd.Series(labels).value_counts()
            results["ASW_label"] = (
                float(silhouette_score(X, labels))
                if (vc.size > 1 and vc.min() >= 2)
                else np.nan
            )
        else:
            results_df = scm.metrics(
                adata_emb,
                adata_int=adata_emb,
                batch_key=str(batch_col),
                label_key=str(label_col),
                embed=MODEL_EMBED_KEY,
                isolated_labels_asw_=False,
                silhouette_=True,
                hvg_score_=False,
                pcr_=False,
                isolated_labels_f1_=False,
                trajectory_=False,
                nmi_=True,
                ari_=True,
                cell_cycle_=False,
                kBET_=False,
                ilisi_=False,
                clisi_=False,
                graph_conn_=True,
            )
            results = results_df.iloc[:, 0].to_dict()

        bio_keys = ["NMI_cluster/label", "ARI_cluster/label", "ASW_label"]
        results["avg_bio"] = np.nanmean([results.get(k, np.nan) for k in bio_keys])

        if not single_batch:
            batch_keys = ["graph_conn", "ASW_label/batch"]
            results["avg_batch"] = np.nanmean(
                [results.get(k, np.nan) for k in batch_keys]
            )

        rename = {
            "NMI_cluster/label": f"NMI_cluster_by_{label_col}_(bio)",
            "ARI_cluster/label": f"ARI_cluster_by_{label_col}_(bio)",
            "ASW_label": f"ASW_by_{label_col}_(bio)",
            "graph_conn": f"graph_conn_by_{batch_col}_(batch)",
            "ASW_label/batch": f"ASW_by_{batch_col}_(batch)",
            "avg_bio": "Avg_bio",
            "avg_batch": "Avg_batch",
        }
        output = {}
        for old, new in rename.items():
            v = results.get(old, np.nan)
            if not (isinstance(v, float) and np.isnan(v)):
                output[new] = np.round(v, 3)
        return pd.DataFrame([output])

    def _generate_benchmarking_table(
        self, adata_emb: sc.AnnData, baselines: list[str]
    ) -> plt.Figure:
        """Generate scib-metrics benchmarking table comparing model to baselines."""
        from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation

        all_keys = [MODEL_EMBED_KEY] + baselines
        pre_integrated = "Unintegrated" if "Unintegrated" in baselines else None

        logger.info(f"Benchmarking embeddings: {all_keys}")
        bm = Benchmarker(
            adata_emb,
            batch_key=self.batch_column_name,
            label_key=self.target_column_name,
            embedding_obsm_keys=all_keys,
            pre_integrated_embedding_obsm_key=pre_integrated,
            bio_conservation_metrics=BioConservation(isolated_labels=False),
            batch_correction_metrics=BatchCorrection(),
            n_jobs=-1,
        )
        bm.prepare()
        bm.benchmark()
        table = bm.plot_results_table(min_max_scale=False)
        return table.figure


class CziBenchmarkCallback(pl.Callback):
    def __init__(
        self,
        batch_column_name: str | None = None,
        target_column_name: str | None = None,
        n_folds: int = 5,
    ):
        super().__init__()
        self.batch_column_name = batch_column_name
        self.target_column_name = target_column_name
        self.n_folds = n_folds
        self.cl: Logger | None = None

    def on_predict_end(self, trainer, pl_module):
        self.cl = Logger.current_logger()
        if not self.cl:
            logger.warning(
                "ClearML logger not found, skipping batch integration reporting."
            )
            return

        adata_emb = get_adata_with_embeddings(trainer, self.batch_column_name)
        if self.target_column_name is None:
            self.target_column_name = trainer.datamodule.label_columns[
                0
            ].label_column_name

        results = self.execute_czi_cell_type_classification_benchmark(adata_emb)
        self.cl.report_table(
            title="CZI Cell Type Classification",
            series="CZI Cell Type Classification",
            table_plot=results,
        )

        # Report logistic regression F1 as scalar for easier querying
        if "mean_fold_f1" in results.index and "lr" in results.columns:
            lr_f1 = float(results.loc["mean_fold_f1", "lr"])
            self.cl.report_single_value(name="f1", value=lr_f1)
        else:
            logger.warning(
                "Logistic regression F1 score (mean_fold_f1, lr) not found in CZI benchmark results"
            )

    def execute_czi_cell_type_classification_benchmark(self, adata_with_embeddings):
        label_prediction_task = MetadataLabelPredictionTask()

        # take ground-truth labels from the dataset
        prediction_task_input = MetadataLabelPredictionTaskInput(
            labels=adata_with_embeddings.obs[self.target_column_name],
            n_folds=self.n_folds,
        )

        # evaluate embeddings against the ground-truth labels using finetuning with classifcal ML classifiers.
        results = label_prediction_task.run(
            cell_representation=adata_with_embeddings.obsm[MODEL_EMBED_KEY],
            task_input=prediction_task_input,
        )

        results_table = pd.DataFrame(
            [(r.metric_type.value, r.params["classifier"], r.value) for r in results],
            columns=["metric", "classifier", "value"],
        )

        results_table_wide = results_table.pivot_table(
            index="metric",
            columns="classifier",
            values="value",
            aggfunc="mean",
            observed=False,
        )
        return results_table_wide


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
