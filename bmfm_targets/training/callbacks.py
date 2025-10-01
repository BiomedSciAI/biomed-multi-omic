import logging
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

from bmfm_targets.datasets.datasets_utils import random_subsampling

logger = logging.getLogger(__name__)


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
        x = adata.X.data
        is_int = np.issubdtype(x.dtype, np.integer)

        x_sample = x[:100_000] if x.size > 100_000 else x

        max_val = x_sample.max()
        mean_val = x_sample.mean()
        median_val = np.median(x_sample)
        pct_ones = np.mean(x_sample == 1)
        pct_small_ints = np.mean(np.isin(x_sample, [1, 2, 3]))

        # Scoring heuristic
        raw_score = sum(
            [
                is_int,
                max_val > 50,
                pct_ones > 0.4,
                mean_val < 2.5,
                median_val <= 1,
                pct_small_ints > 0.6,
            ]
        )

        looks_raw = raw_score >= 4  # majority vote

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


class SavePredictionsH5ADCallback(pl.Callback):
    def __init__(self, output_file_name=None):
        super().__init__()
        self.output_file_name = output_file_name

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._save_predictions_h5ad(trainer, pl_module)

    def _save_predictions_h5ad(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        preds_df = pl_module.prediction_df["label_expressions"]
        matrix = preds_df.pivot_table(
            index=preds_df.index,
            columns="input_genes",
            values="predicted_expressions",
            fill_value=0,
        )
        adata = sc.AnnData(
            X=matrix.values,
            obs=pd.DataFrame(index=matrix.index),  # samples
            var=pd.DataFrame(index=matrix.columns),  # genes
        )

        adata.obs["target_gene"] = preds_df.groupby(preds_df.index)[
            "perturbed_set"
        ].first()

        output_file = pathlib.Path(trainer.log_dir) / "predictions.h5ad"
        if self.output_file_name is not None:
            output_file = pathlib.Path(self.output_file_name)

        adata.write(output_file)

        print(f"Saved predictions to {output_file}")
