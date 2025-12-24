import logging
import pickle
from pathlib import Path

import clearml
import pandas as pd
from matplotlib import pyplot as plt

from ..metrics import perturbation_metrics as pm
from ..metrics import plots
from .base import BaseTrainingModule

logger = logging.getLogger(__name__)
DEFAULT_LABEL_SMOOTHING = 0.01


class SequenceLabelingTrainingModule(BaseTrainingModule):
    DEFAULT_METRICS = {"pcc", "nonzero_confusion_matrix"}
    MODELING_STRATEGY = "sequence_labeling"

    def _shared_test_val_on_end(self, split: str):
        super()._shared_test_val_on_end(split)
        if (
            not self.trainer_config.batch_prediction_behavior
            or self.trainer_config.batch_prediction_behavior == "no_metrics"
        ):
            # warned already in super()._shared_test_val_on_end
            return
        for field_metric_key in self.get_supported_field_metric_keys():
            logger.info(f"Calculating perturbation metrics for {field_metric_key}")
            if not any(
                i in field_metric_key
                for i in ("label_expressions", "delta_baseline_expressions")
            ):
                logger.warning(
                    "Unable to calculate perturbation specific metrics for predictions"
                    f"of field metric key {field_metric_key}, only 'label_expressions' is supported"
                )
                continue
            preds_df = self.prediction_df[field_metric_key]
            self.log_perturbation_specific_metrics(split, preds_df)
            break

    def log_perturbation_specific_metrics(self, split, predictions_df):
        group_means = self.kwargs.get("group_means")
        if group_means is None:
            logger.warning(
                "No group_means found in kwargs cannot do perturbation metrics"
            )
            return

        grouped_ground_truth = pm.get_grouped_ground_truth(group_means)
        logger.info("Averaging group predictions (pseudobulks)")
        grouped_predictions = pm.get_grouped_predictions(
            predictions_df, grouped_ground_truth
        )

        top20 = self.kwargs.get("top20_de")
        agg_metrics_list = []
        agg_metrics_top20_list = []

        logger.info("prepare args for discrimination score metric")

        (
            real_effects,
            pred_effects,
            perturbations,
            overlap_genes,
        ) = pm.prepare_args_for_discrimination_score(grouped_predictions)
        logger.info("Calculating discrimination score")
        norm_ranks, distances = pm.discrimination_score(
            real_effects, pred_effects, perturbations, overlap_genes
        )
        logger.info(f"Calculating metrics for {len(perturbations)} perturbations")
        for pert in perturbations:
            if pert == "":
                logger.warning(
                    "Encountered samples with no perturbation in sequence! This should not happen"
                )
                continue
            agg_metrics = pm.get_aggregated_perturbation_metrics(
                grouped_predictions.loc[pert]
            )
            agg_metrics["discrimination_score"] = norm_ranks[pert]
            if len(perturbations) <= 10:
                self.log_aggregate_perturbation_metrics(split, agg_metrics, pert, "")
            agg_metrics["pert"] = pert
            agg_metrics_list.append(agg_metrics)

            if top20:
                # it is possible that some highly variable genes are missing
                # mostly due to small batches and short max_length values
                valid_indices = [
                    g for g in top20[pert] if g in grouped_predictions.loc[pert].index
                ]
                if len(valid_indices) < 2:
                    logger.info("{pert} has less than 2 tp 20_de genes valid indices")
                else:
                    agg_metrics_top20 = pm.get_aggregated_perturbation_metrics(
                        grouped_predictions.loc[pert].loc[valid_indices]
                    )
                    if len(perturbations) <= 10:
                        self.log_aggregate_perturbation_metrics(
                            split,
                            agg_metrics_top20,
                            pert,
                            "top20_de_",
                        )
                        try:
                            logger.info(
                                f"plot aggregated perturbation metrics for top20, pert={pert}"
                            )

                            self.create_and_log_top20_deg_perturbation_plot(
                                split, predictions_df, grouped_ground_truth, top20, pert
                            )
                        except Exception as e:
                            # log & dump the input
                            input_dump_file = (
                                Path(self.trainer.log_dir)
                                / f"{split}_{pert}_failed_plot.pkl"
                            )
                            with open(input_dump_file, "wb") as f:
                                logging.error(
                                    f"Failed creating plog for {split}, pert={pert}: {e}"
                                )
                                logging.info(f"dumping data to {input_dump_file}")
                                pickle.dump(
                                    (
                                        split,
                                        predictions_df,
                                        grouped_ground_truth,
                                        top20,
                                        pert,
                                    ),
                                    f,
                                )
                    agg_metrics_top20["pert"] = pert
                    agg_metrics_top20_list.append(agg_metrics_top20)

        agg_metrics_df = pd.DataFrame().from_records(agg_metrics_list)
        mean_agg_metrics = agg_metrics_df.drop("pert", axis=1).mean()
        mean_row = pd.DataFrame(
            {"pert": ["mean"], **mean_agg_metrics.to_dict()},
            index=[len(mean_agg_metrics)],
        )
        agg_metrics_df_with_mean = pd.concat(
            [agg_metrics_df, mean_row], ignore_index=True
        )

        logger.info(
            f"logging table of aggregation metrics per perturbation for {split}"
        )

        self.log_table(
            split, "aggregation metrics per perturbation", agg_metrics_df_with_mean
        )
        self.log_mean_aggregate_perturbation_metrics(agg_metrics_df, split, "")
        self.plot_agg_pred_vs_baseline_scatter(
            split, agg_metrics_df, "agg_pcc", "baseline_agg_pcc_from_avg_perturbation"
        )
        self.plot_heatmap(
            split,
            distances,
            "ground-truth",
            "predictions",
            "Predicted vs ground-truth L1 distances of pseudo bulks",
        )

        agg_metrics_top20_df = pd.DataFrame().from_records(agg_metrics_top20_list)
        if "pert" in agg_metrics_top20_df.columns:
            mean_agg_metrics_top20 = agg_metrics_top20_df.drop("pert", axis=1).mean()
            mean_row = pd.DataFrame(
                {"pert": ["mean"], **mean_agg_metrics_top20.to_dict()},
                index=[len(agg_metrics_top20_df)],
            )
            agg_metrics_top20_df_with_mean = pd.concat(
                [agg_metrics_top20_df, mean_row], ignore_index=True
            )

        self.log_table(
            split,
            "aggregation metrics top20 differentially expressed genes per perturbation",
            agg_metrics_top20_df_with_mean,
        )
        self.log_mean_aggregate_perturbation_metrics(
            agg_metrics_top20_df, split, "top20_de_"
        )
        self.plot_agg_pred_vs_baseline_scatter(
            split,
            agg_metrics_top20_df,
            "agg_pcc",
            "baseline_agg_pcc_from_avg_perturbation",
            "Top20 DE ",
        )

    def plot_agg_pred_vs_baseline_scatter(
        self,
        split,
        results_df,
        model_agg_metric_column,
        baseline_agg_metric_column,
        identifier=None,
    ):
        fig = plots.make_predictions_gt_density_plot(
            predictions_df=results_df,
            predicted_label=model_agg_metric_column,
            gt_label=baseline_agg_metric_column,
            kind="scatter",
        )
        cl = clearml.Logger.current_logger()

        if cl:
            if identifier is None:
                identifier = ""
            title = f"{identifier} Aggregated {model_agg_metric_column} vs {baseline_agg_metric_column}"
            cl.report_matplotlib_figure(
                title=title,
                series=split,
                figure=fig,
                iteration=self.global_step,
            )
        plt.close(fig)

    def plot_heatmap(self, split, distances_df, x_label, y_label, title):
        fig = plots.make_heatmap_plot(distances_df, x_label, y_label, title)
        cl = clearml.Logger.current_logger()
        if cl:
            cl.report_matplotlib_figure(
                title=title,
                series=split,
                figure=fig,
                iteration=self.global_step,
                report_image=True,  # so it will go to debug_samples, better for large images
            )
        plt.close(fig)

    def log_mean_aggregate_perturbation_metrics(
        self, agg_metrics_df, split, identifier
    ):
        logger.info(
            f"logging mean_aggregate_perturbation_metrics for {split},{identifier}"
        )
        mean_agg_metrics = agg_metrics_df.drop("pert", axis=1).mean()
        for metric_name, mean_value in mean_agg_metrics.items():
            self.logger.experiment.add_scalar(
                f"{split}/mean_{identifier}{metric_name}", mean_value, self.global_step
            )

    def log_aggregate_perturbation_metrics(self, split, agg_metrics, pert, identifier):
        logger.info(
            f"logging aggregate_perturbation_metrics for {split},{pert},{identifier}"
        )
        for k, v in agg_metrics.items():
            self.logger.experiment.add_scalar(
                f"{split}/{pert}_{identifier}{k}", v, self.global_step
            )

    def create_and_log_top20_deg_perturbation_plot(
        self, split, preds_df, mean_expressions, top20, pert
    ):
        top20_df = preds_df[preds_df.input_genes.isin(top20[pert])]
        mean_control = mean_expressions["Control"].loc[top20[pert]]
        fig = plots.make_top20_deg_perturbation_plot(top20_df, mean_control)
        cl = clearml.Logger.current_logger()
        if cl:
            cl.report_matplotlib_figure(
                title=f"{pert} - Difference from Control for Top 20 DEG",
                series=split,
                figure=fig,
                iteration=self.global_step,
            )
        plt.close(fig)
