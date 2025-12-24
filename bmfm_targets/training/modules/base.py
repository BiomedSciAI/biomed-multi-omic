import logging
import pathlib
from collections import Counter, defaultdict, deque
from itertools import chain
from pathlib import Path
from typing import Literal

import clearml
import pytorch_lightning as pl
import torch
import transformers
from lightning_utilities.core.rank_zero import rank_zero_only
from matplotlib import pyplot as plt
from peft import get_peft_model
from torchmetrics import MetricCollection
from torchmetrics.wrappers import MultitaskWrapper

from bmfm_targets.config import LabelColumnInfo, TrainerConfig
from bmfm_targets.config.model_config import SCModelConfigBase
from bmfm_targets.models import get_model_from_config, instantiate_classification_model
from bmfm_targets.models.model_utils import SequenceClassifierOutputWithEmbeddings
from bmfm_targets.models.predictive.layers import get_embeddings_from_outputs
from bmfm_targets.tokenization import MultiFieldTokenizer
from bmfm_targets.training import metrics
from bmfm_targets.training.metrics import (
    FieldLossTask,
    LabelLossTask,
    get_loss_tasks,
    log_confusion_matrix_to_clearml,
    plots,
)
from bmfm_targets.training.metrics.batch_prediction_metrics import (
    create_field_predictions_df,
    create_label_predictions_df,
    field_predictions_df_columns,
    get_best_and_worst_genes,
    get_gene_level_expression_error,
    get_gene_metrics_from_gene_errors,
)
from bmfm_targets.training.metrics.metric_handling import (
    limit_confusion_matrix_to_numerical_labels,
)

logger = logging.getLogger(__name__)


class BaseTrainingModule(pl.LightningModule):
    DEFAULT_METRICS = {
        "accuracy",
        "f1",
        "mse",
        "pcc",
        "confusion_matrix",
        "nonzero_confusion_matrix",
    }
    PERPLEXITY_LOGGING = False  # whether to log perplexity of CE loss if present
    MODELING_STRATEGY = ""  # set in base class to "mlm", "sequence_labeling" etc

    def __init__(
        self,
        model_config: SCModelConfigBase,
        trainer_config: TrainerConfig,
        tokenizer: MultiFieldTokenizer | None = None,
        label_dict: dict[str, dict[str, int]] | None = None,
        **kwargs,
    ):
        """
        Pytorch Lightning module for training a masked language model.

        Args:
        ----
            model_config_dict (dict): Dictionary containing the model configuration.
            trainer_config (TrainerConfig): Training configuration.

        """
        super().__init__()

        if "config_is_loaded_from_ckpt" in kwargs.keys():
            model_config.checkpoint = None

        # this is needed when model_config is loaded from old checkpoints which don't contain the label_columns item in "hyperparameter" section
        if (
            not hasattr(model_config, "label_columns")
            or model_config.label_columns is None
            and label_dict is not None
        ):
            logger.warning(
                f"Adding label columns to model_config from label_dict: {label_dict.keys()}"
            )
            setattr(
                model_config,
                "label_columns",
                [
                    LabelColumnInfo(
                        label_column_name=label_column_name, n_unique_values=len(labels)
                    )
                    for label_column_name, labels in label_dict.items()
                ],
            )

        self.model_config = model_config
        self.tokenizer = tokenizer
        self.trainer_config = trainer_config
        self.label_dict = label_dict
        self.loss_tasks = get_loss_tasks(
            self.trainer_config.losses,
            tokenizer=self.tokenizer,
            fields=self.model_config.fields,
            label_columns=self.model_config.label_columns,
        )
        self.kwargs = kwargs
        metric_requests = self.trainer_config.metrics
        if metric_requests is None:
            metric_requests = self.default_metrics()
        self.initialize_metrics(metric_requests)
        if self.MODELING_STRATEGY == "sequence_classification":
            self.model = instantiate_classification_model(
                self.model_config, self.loss_tasks, self.device
            )
        else:
            self.model = get_model_from_config(
                self.model_config, modeling_strategy=self.MODELING_STRATEGY
            )

        self.lora_config = self.trainer_config.get_lora_config()
        if self.lora_config:
            self.model = get_peft_model(self.model, self.lora_config.to_peft_config())

        if self.tokenizer:
            self.token_values = {
                loss_task.field.field_name: self.tokenizer.get_token_values(
                    loss_task.field.field_name
                )
                for loss_task in self.loss_tasks
                if isinstance(loss_task, FieldLossTask)
            }
        else:
            self.token_values = None

        self.prediction_df = {}
        self.token_level_errors = {}
        self.sample_metadata_keys = ["cell_name", "seq_id", "perturbed_genes"]
        self.save_hyperparameters(ignore=["tokenizer"])

    def update_metrics(
        self,
        labels: torch.Tensor,
        outputs: SequenceClassifierOutputWithEmbeddings,
        split: Literal["test", "train", "validation"],
    ):
        """
        Process model outputs and calculate metrics for the given split.

        Extracts and formats inputs for each task's metrics, then applies the
        appropriate metric collection based on the data split.

        Args:
        ----
            labels: Dictionary of ground truth labels for each task
            outputs: Model output object containing logits
            split: Data split name ("test", "train", "validation")

        Returns:
        -------
            Dictionary of metric values from the appropriate metrics collection

        """
        model_outputs = {}
        gt_labels = {}
        metric_inputs = [
            loss_task.extract_metric_inputs(outputs.logits, labels)
            for loss_task in self.loss_tasks
        ]

        duplicated_metric_keys = [
            k for k, v in Counter([i[0] for i in metric_inputs]).items() if v > 1
        ]
        predictions = metrics.calculate_predictions(self.loss_tasks, outputs.logits)
        for metric_key, task_outputs, task_labels in metric_inputs:
            if metric_key in duplicated_metric_keys:
                task_outputs = predictions[metric_key].view(task_labels.shape)
                if metric_key in model_outputs:
                    continue
            # ignore -100 labels explicitly when dealing with PCC-type outputs
            if task_labels.shape == task_outputs.shape:
                ignore_mask = task_labels != -100
                task_labels = task_labels[ignore_mask]
                task_outputs = task_outputs[ignore_mask]
            model_outputs[metric_key] = task_outputs
            gt_labels[metric_key] = task_labels
        return self.split_metrics(split)(model_outputs, gt_labels)

    @classmethod
    def default_metrics(cls):
        return [{"name": metric_name} for metric_name in cls.DEFAULT_METRICS]

    def initialize_metrics(self, metric_requests: list[dict]):
        """
        Initialize the metrics.

        Note that `torchmetrics` metrics MUST be direct attributes of the training module
        in order for the automated device synchronization to work.
        As of early 2024, if they are defined in a data structure it does not work.
        """
        metrics_dict = {}
        for loss_task in self.loss_tasks:
            relevant_metrics = metrics.get_relevant_metrics(
                metric_requests, loss_task.output_size
            )
            metric_key = loss_task.metric_key
            metrics_dict[metric_key] = MetricCollection(
                {
                    mt["name"]: metrics.get_metric_object(mt, loss_task.output_size)
                    for mt in relevant_metrics
                }
            )

        self.train_metrics = MultitaskWrapper(metrics_dict).clone()
        self.val_metrics = MultitaskWrapper(metrics_dict).clone()
        self.test_metrics = MultitaskWrapper(metrics_dict).clone()

        deque_len = self.trainer_config.batch_prediction_behavior
        if not isinstance(deque_len, int):
            deque_len = None
        self.val_batch_predictions = defaultdict(lambda: deque(maxlen=deque_len))
        self.test_batch_predictions = defaultdict(lambda: deque(maxlen=deque_len))

    def on_test_start(self) -> None:
        self.test_batch_predictions.clear()

    def on_validation_start(self) -> None:
        self.val_batch_predictions.clear()

    def split_batch_predictions(self, split):
        if split == "test":
            return self.test_batch_predictions
        if split == "validation":
            return self.val_batch_predictions

        raise ValueError(f"No batch predictions for split: {split}")

    def split_metrics(
        self, split: Literal["test", "train", "validation"]
    ) -> MultitaskWrapper:
        if split == "test":
            return self.test_metrics
        if split == "train":
            return self.train_metrics
        if split == "validation":
            return self.val_metrics
        raise ValueError(f"Invalid metrics object {split}")

    def forward(self, batch) -> dict:  # type: ignore[override]
        """
        Forward pass of the model.

        Args:
        ----
            batch (dict): Batch of data.

        Returns:
        -------
            dict: Dictionary containing the model outputs.

        """
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch.get("labels"),
        )

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        batch_size = batch["input_ids"].shape[0]

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=labels,
        )

        all_losses = metrics.calculate_losses(self.loss_tasks, outputs.logits, labels)
        loss = all_losses["loss"]
        step_metrics = self.update_metrics(labels, outputs, split="train")

        with torch.no_grad():
            self.log_losses(
                all_losses, batch_size, on_step=True, on_epoch=True, sync_dist=True
            )
            self.log_metrics(
                step_metrics, split="train", batch_size=batch_size, suffix="_step"
            )
        if loss != 0.0:
            return loss
        else:
            return None

    def validation_step(self, batch, batch_idx):
        labels = batch["labels"]
        batch_size = batch["input_ids"].shape[0]

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=labels,
        )

        all_losses = metrics.calculate_losses(self.loss_tasks, outputs.logits, labels)
        loss = all_losses["loss"]
        self.update_metrics(labels, outputs, split="validation")

        with torch.no_grad():
            if self.trainer_config.batch_prediction_behavior:
                self.record_batch_predictions(batch, outputs, "validation")
            self.log_losses(
                all_losses,
                batch_size,
                split="validation",
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        return loss

    def test_step(self, batch, batch_idx):
        labels = batch["labels"]
        batch_size = batch["input_ids"].shape[0]

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=labels,
        )

        all_losses = metrics.calculate_losses(self.loss_tasks, outputs.logits, labels)
        loss = all_losses["loss"]
        step_metrics = self.update_metrics(labels, outputs, split="test")
        with torch.no_grad():
            if self.trainer_config.batch_prediction_behavior:
                self.record_batch_predictions(batch, outputs, "test")
            self.log_metrics(step_metrics, split="test", batch_size=batch_size)
            self.log_losses(
                all_losses,
                batch_size,
                split="test",
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        )

        return self.get_predict_step_output(batch, outputs)

    def get_predict_step_output(self, batch, outputs):
        predictions_dict = {}
        embeddings = get_embeddings_from_outputs(
            outputs,
            batch["attention_mask"],
            pooling_method=self.trainer_config.pooling_method,
        )
        predictions_dict["embeddings"] = embeddings.to(torch.float32).cpu().numpy()

        for key in self.sample_metadata_keys:
            if key in batch:
                predictions_dict[key] = batch[key]

        for loss_task in filter(
            lambda x: isinstance(x, LabelLossTask), self.loss_tasks
        ):
            metric_key = loss_task.metric_key
            predictions_dict[f"{metric_key}_predictions"] = loss_task.get_predictions(
                outputs.logits
            )
            predictions_dict[f"{metric_key}_logits"] = (
                outputs.logits[metric_key].cpu().numpy()
            )
        return predictions_dict

    def log_losses(
        self,
        all_losses,
        batch_size,
        split="train",
        prefix="",
        suffix="",
        on_step=True,
        on_epoch=False,
        sync_dist=False,
    ):
        # note: 'validation/loss' is hardcoded as the metric to monitor for best
        # checkpoint saving, dont change the string without syncing checkpointing code
        for loss_name, loss_value in all_losses.items():
            self.log(
                f"{split}/{prefix}{loss_name}{suffix}",
                loss_value,
                batch_size=batch_size,
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=sync_dist,
                prog_bar=True,
            )
        if self.PERPLEXITY_LOGGING:
            self.log_perplexity(
                all_losses,
                batch_size,
                split=split,
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=sync_dist,
            )

    def log_perplexity(
        self,
        all_losses,
        batch_size,
        split="train",
        prefix="",
        on_step=True,
        on_epoch=False,
        sync_dist=False,
    ):
        for loss_name, loss_value in all_losses.items():
            if loss_name.endswith(("cross_entropy_loss", "focal_loss")):
                perplexity = torch.exp(loss_value)
                field_name = loss_name.replace("_cross_entropy_loss", "").replace(
                    "_focal_loss", ""
                )
                self.log(
                    f"{split}/{prefix}{field_name}_perplexity",
                    perplexity,
                    batch_size=batch_size,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    sync_dist=sync_dist,
                )

    def log_metrics(
        self,
        computed_metrics,
        split,
        batch_size: int | None = None,
        suffix="",
        on_step=True,
        on_epoch=False,
        sync_dist=False,
    ):
        for label, metric_collection in computed_metrics.items():
            for metric_name, metric_or in metric_collection.items():
                if "confusion_matrix" in metric_name:
                    continue
                if (
                    "auc" in metric_name and metric_or.numel() > 1
                ):  # # If in case for the multi-label classification with no reduction
                    metric_or[torch.isnan(metric_or)] = 0.5
                    metric_or[metric_or == 0] = 0.5
                    metric = metric_or.mean()
                else:
                    metric = metric_or
                self.log(
                    f"{split}/{label}_{metric_name}{suffix}",
                    metric,
                    sync_dist=sync_dist,
                    on_epoch=on_epoch,
                    on_step=on_step,
                    batch_size=batch_size,
                )

    def get_prediction_active_keys(self, label_task_only=False):
        """Returns a list of keys to save batch predictions."""
        active_keys = {
            loss_task.metric_key: loss_task
            for loss_task in self.loss_tasks
            if loss_task.loss_name != "hce"
            and (not label_task_only or isinstance(loss_task, LabelLossTask))
        }
        return active_keys

    def record_batch_predictions(self, batch, outputs, split):
        predictions = metrics.calculate_predictions(self.loss_tasks, outputs.logits)
        # predictions are on the level of the field, not the loss task
        # each field/label gets all the logits combined for the tracking code
        active_keys = self.get_prediction_active_keys()
        for _, loss_task in active_keys.items():
            batch_tensors = loss_task.concat_batch_tensors(batch, outputs, predictions)
            batch_tensors = batch_tensors.detach().cpu()
            self.split_batch_predictions(split)[loss_task.metric_key].append(
                batch_tensors
            )

        for key in self.sample_metadata_keys:
            if key in batch:
                self.split_batch_predictions(split)[key].append(batch[key])

    def on_test_epoch_end(self):
        self._shared_test_val_on_end("test")

    def on_validation_epoch_end(self) -> None:
        self.log_epoch_metrics_and_reset("validation")
        self._shared_test_val_on_end("validation")

    def on_train_epoch_start(self) -> None:
        # This is a temporary partial fix for the issue: https://github.com/Lightning-AI/pytorch-lightning/issues/19604
        if self.global_step > 0:
            self.log_epoch_metrics_and_reset("train", suffix="_epoch")

    def _shared_test_val_on_end(self, split: str):
        if not self.trainer_config.batch_prediction_behavior:
            logger.warning(
                "Unable to calculate rich metrics because batches were not tracked. "
                'Please re-run with `trainer_config.batch_prediction_behavior` set to "track" or "dump"'
            )
            return
        self.process_batch_predictions(split)
        logger.info(f"plotting batch_predictions for split {split}")
        self.plot_batch_predictions_for_split(split)

        logger.info(f"logging token level errors for split {split}")
        self.log_token_level_errors_for_split(split)

        if self.trainer_config.batch_prediction_behavior == "dump":
            logger.info(f"dumping batch predictions {split}")

            self.dump_batch_predictions(split)

    def log_token_level_errors_for_split(self, split):
        if len(self.get_supported_field_metric_keys()) == 0:
            return

        for metric_key, gene_level_error in self.token_level_errors.items():
            gene_metrics = get_gene_metrics_from_gene_errors(gene_level_error)
            for k, v in gene_metrics.items():
                if hasattr(self.logger.experiment, "add_scalar"):
                    self.logger.experiment.add_scalar(
                        f"{split}/{metric_key}_{k}", v, self.global_step
                    )
            best_genes, worst_genes = get_best_and_worst_genes(gene_level_error)
            cl = clearml.Logger.current_logger()
            if cl:
                cl.report_table(
                    f"Best performing common genes {metric_key} (top decile nonzero count, lowest avg err)",
                    series=split,
                    iteration=self.global_step,
                    table_plot=best_genes,
                )
                cl.report_table(
                    f"Worst performing common genes {metric_key} (top decile nonzero count, highest avg err)",
                    series=split,
                    iteration=self.global_step,
                    table_plot=worst_genes,
                )

    def plot_batch_predictions_for_split(self, split):
        for field_metric_key in self.get_supported_field_metric_keys(
            limit_to_continuous_value_encoder=True
        ):
            preds_df = self.prediction_df[field_metric_key]
            if "delta_" in field_metric_key:
                field = [
                    f.field for f in self.loss_tasks if f.metric_key == field_metric_key
                ][0]
                gt_label = f"label_{field.field_name}"
                predicted_label = f"predicted_{field.field_name}"
                plot_only_non_zeros = False
            else:
                gt_label = "label_expressions"
                predicted_label = "predicted_expressions"
                plot_only_non_zeros = True

            logger.info(f"creating density plot for split {split}")
            self.create_and_log_predictions_density_plot(
                split,
                preds_df,
                suffix=field_metric_key,
                gt_label=gt_label,
                predicted_label=predicted_label,
                plot_only_non_zeros=plot_only_non_zeros,
            )
        for label_task in filter(
            lambda x: isinstance(x, LabelLossTask), self.loss_tasks
        ):
            preds_df = self.prediction_df.get(label_task.metric_key, None)
            if preds_df is not None:
                if label_task.label_column.is_regression_label:
                    gt_label = label_task.metric_key + "_label"
                    predicted_label = label_task.metric_key + "_prediction"
                    logger.info(f"creating density plot for {split}, {predicted_label}")
                    self.create_and_log_predictions_density_plot(
                        split,
                        preds_df,
                        gt_label=gt_label,
                        predicted_label=predicted_label,
                        suffix=label_task.metric_key,
                        plot_only_non_zeros=False,
                    )
                else:
                    logger.info(
                        f"creating accuracy by target for {split}, {label_task.output_key}"
                    )

                    self.create_and_log_accuracy_by_targets_w_ci(
                        split,
                        preds_df,
                        label_column_name=label_task.output_key,
                    )

    def dump_batch_predictions(self, split):
        active_keys = self.get_prediction_active_keys()
        for output_key, lt in active_keys.items():
            if isinstance(lt, FieldLossTask):
                columns = field_predictions_df_columns(
                    self.model_config.fields, lt.field, self.MODELING_STRATEGY
                )
                include_nonmasked = lt.wced_target is None
                preds_df = self._get_field_predictions_df(
                    split,
                    lt.metric_key,
                    include_nonmasked=include_nonmasked,
                    columns=columns,
                )
            else:
                preds_df = self._get_label_predictions_df(split, lt.output_key)
            ofname = f"{split}_{output_key}_iteration_{self.global_step}.csv"
            preds_df.to_csv(Path(self.logger.log_dir) / ofname)

    def create_and_log_accuracy_by_targets_w_ci(
        self, split, preds_df, label_column_name
    ):
        title = f"Accuracy with binomial CI per {label_column_name}"
        fig = plots.make_accuracy_by_target_plot(preds_df, label_column_name)
        cl = clearml.Logger.current_logger()
        if cl:
            cl.report_matplotlib_figure(
                title=f"{title} - {split}",
                series=split,
                figure=fig,
                iteration="",
            )
        plt.close(fig)

    def create_and_log_predictions_density_plot(
        self,
        split,
        preds_df,
        predicted_label="predicted_expressions",
        gt_label="label_expressions",
        plot_only_non_zeros=True,
        suffix="",
    ):
        if plot_only_non_zeros:
            mask = preds_df[gt_label] > 0
            mask &= preds_df[predicted_label] > 0
            nonzero_preds = preds_df[mask]
            logger.info(f"computed mask, new shape: {nonzero_preds.shape}")
            fig = plots.make_predictions_gt_density_plot(
                nonzero_preds,
                predicted_label=predicted_label,
                gt_label=gt_label,
            )
        else:
            fig = plots.make_predictions_gt_density_plot(
                preds_df,
                predicted_label=predicted_label,
                gt_label=gt_label,
            )
        cl = clearml.Logger.current_logger()
        if cl:
            cl.report_matplotlib_figure(
                title=f"Predicted expression vs ground truth expression {suffix}",
                series=split,
                figure=fig,
                iteration=self.global_step,
            )
        plt.close(fig)

    def configure_optimizers(self):
        optimizer_grouped_parameters = get_weight_decay_groups(
            self.trainer_config.weight_decay, self.model
        )
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            betas=self.trainer_config.betas,
            lr=self.trainer_config.learning_rate,
            eps=self.trainer_config.epsilon,
        )
        if self.trainer_config.lr_decay_steps:
            if self.trainer_config.lr_decay_steps == -1:
                num_training_steps = (
                    self.trainer.estimated_stepping_batches
                    - self.trainer_config.warmup_steps
                )
                logger.info(
                    f"Number of lr decay steps set at {num_training_steps} since -1 was asked"
                )
            else:
                if (
                    self.trainer_config.lr_decay_steps
                    > 2 * self.trainer.estimated_stepping_batches
                ):
                    logger.warning(
                        f"Asked for {self.trainer_config .lr_decay_steps} LR decay steps but the"
                        f" model will only be trained for {self.trainer.estimated_stepping_batches}"
                        " steps, this might be an oversight."
                    )
                num_training_steps = self.trainer_config.lr_decay_steps

            schedule = transformers.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.trainer_config.warmup_steps,
                num_training_steps=num_training_steps
                + self.trainer_config.warmup_steps,
            )
            schedulers = [{"scheduler": schedule, "interval": "step"}]
        elif self.trainer_config.warmup_steps > 0:
            schedule = transformers.get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.trainer_config.warmup_steps,
            )
            schedulers = [{"scheduler": schedule, "interval": "step"}]
        else:
            schedulers = []

        return [optimizer], schedulers

    @rank_zero_only
    def save_transformer(
        self,
        save_dir: pathlib.Path,
        tokenizer: transformers.PreTrainedTokenizer | None = None,
    ):
        """
        Save the model and tokenizer to the specified directory.

        Args:
        ----
            save_dir (pathlib.Path): Directory to save the model to.
            tokenizer (transformers.PreTrainedTokenizer, optional): Tokenizer to save. Defaults to None.

        """
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving model to {save_dir}")
        self.model.save_pretrained(str(save_dir))
        if tokenizer is not None:
            logger.info(f"Saving tokenizer to {save_dir}")
            tokenizer.save_pretrained(
                str(save_dir), legacy_format=not tokenizer.is_fast
            )

    def compute_confusion_matrix_dict(self, prefix, cm_name="confusion_matrix"):
        metrics_obj = self.split_metrics(prefix)
        cm = {}
        for label in metrics_obj.task_metrics.keys():
            if cm_name in metrics_obj.task_metrics[label]:
                cm[label] = (
                    metrics_obj.task_metrics[label][cm_name].compute().cpu().numpy()
                )
                metrics_obj.task_metrics[label][cm_name].reset()
        return cm

    def log_epoch_metrics_and_reset(self, split, suffix=""):
        epoch_metrics = self.split_metrics(split).compute()
        self.log_metrics(
            epoch_metrics,
            split,
            suffix=suffix,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        self.log_split_confusion_matrix(split)
        self.split_metrics(split).reset()

    def get_confusion_matrices(
        self, prefix, cm_types=("confusion_matrix", "nonzero_confusion_matrix")
    ):
        """Retrieve and reset confusion matrices from metrics."""
        metrics_obj = self.split_metrics(prefix)
        cm_dict = {name: {} for name in cm_types}

        for output_key, output_key_metrics in metrics_obj.task_metrics.items():
            for cm_name in cm_types:
                if cm_name in output_key_metrics:
                    cm_dict[cm_name][output_key] = (
                        output_key_metrics[cm_name].compute().cpu().numpy()
                    )
                    output_key_metrics[cm_name].reset()

        return cm_dict

    def log_confusion_matrices(self, split, cm_dict):
        """Log confusion matrices using ClearML."""
        if not clearml.Logger.current_logger():
            return

        for cm_type, cm_data in cm_dict.items():
            for output_key, cm in cm_data.items():
                labels = None
                if (
                    cm_type == "confusion_matrix"
                    and self.token_values is not None
                    and output_key in self.token_values
                ):
                    labels, cm = limit_confusion_matrix_to_numerical_labels(
                        self.token_values[output_key], cm
                    )
                elif self.label_dict and output_key in self.label_dict:
                    labels = [
                        k
                        for k, _ in sorted(
                            self.label_dict[output_key].items(), key=lambda x: x[1]
                        )
                    ]

                if cm_type == "nonzero_confusion_matrix":
                    labels = ["non-zero", "zero"]

                log_confusion_matrix_to_clearml(
                    cm, split, labels, output_key.capitalize(), self.global_step
                )

    def log_split_confusion_matrix(self, split):
        cm_dict = self.get_confusion_matrices(split)
        self.log_confusion_matrices(split, cm_dict)

    def process_batch_predictions(self, split):
        for metric_key in self.get_supported_field_metric_keys():
            field = [f.field for f in self.loss_tasks if f.metric_key == metric_key][0]
            columns = field_predictions_df_columns(
                self.model_config.fields, field, self.MODELING_STRATEGY
            )
            preds_df = self._get_field_predictions_df(
                split, metric_key, include_nonmasked=False, columns=columns
            )
            self.prediction_df[metric_key] = preds_df
            logger.info(f"calculating get_gene_level_expression_error for {metric_key}")
            try:
                self.token_level_errors[metric_key] = get_gene_level_expression_error(
                    preds_df
                )
            except Exception as e:
                logger.warning(
                    f"Unable to calculate gene level errors for {metric_key} due to {e}"
                )
        for label_metric_key in self.get_prediction_active_keys(label_task_only=True):
            self.prediction_df[label_metric_key] = self._get_label_predictions_df(
                split, label_metric_key
            )

    def get_supported_field_metric_keys(
        self, limit_to_continuous_value_encoder=False
    ) -> set[str]:
        is_supported_filter = (
            lambda lt: isinstance(lt, FieldLossTask) and "expressions" in lt.output_key
        )
        if limit_to_continuous_value_encoder:
            filter_func = (
                lambda lt: is_supported_filter(lt)
                and lt.field.tokenization_strategy == "continuous_value_encoder"
            )
        else:
            filter_func = is_supported_filter
        return {lt.metric_key for lt in filter(filter_func, self.loss_tasks)}

    def _get_field_predictions_df(self, split, metric_key, include_nonmasked, columns):
        predictions_list = self.split_batch_predictions(split)[metric_key]
        if "genes" in self.tokenizer.field_to_tokenizer_map:
            id2gene = {v: k for k, v in self.tokenizer.get_field_vocab("genes").items()}
        elif "dna_chunks" in self.tokenizer.field_to_tokenizer_map:
            id2gene = {
                v: k for k, v in self.tokenizer.get_field_vocab("dna_chunks").items()
            }
        else:
            raise ValueError("Expected 'genes'/'dna_chunks' tokenizer field missing")
        sample_level_metadata = {}
        for key in self.sample_metadata_keys:
            if key in self.split_batch_predictions(split):
                sample_level_metadata[key] = list(
                    chain.from_iterable(self.split_batch_predictions(split)[key])
                )

        return create_field_predictions_df(
            predictions_list=predictions_list,
            id2gene=id2gene,
            columns=columns,
            sample_names=sample_level_metadata.pop("cell_name", None),
            include_nonmasked=include_nonmasked,
            sample_level_metadata=sample_level_metadata,
        )

    def _get_label_predictions_df(self, split, label_name):
        predictions_list = self.split_batch_predictions(split)[label_name]
        if "cell_name" in self.split_batch_predictions(split):
            sample_names = list(
                chain.from_iterable(self.split_batch_predictions(split)["cell_name"])
            )
        else:
            sample_names = None
        return create_label_predictions_df(
            predictions_list, label_name, sample_names, self.label_dict[label_name]
        )

    def log_table(self, split, table_title, df):
        cl = clearml.Logger.current_logger()
        if cl:
            cl.report_table(
                title=table_title,
                series=split,
                iteration=self.global_step,
                table_plot=df,
            )


def get_weight_decay_groups(weight_decay, model):
    """Code from MinGPT https://github.com/karpathy/minGPT/blob/master/mingpt/model.py."""
    if weight_decay is not None:
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (
            torch.nn.LayerNorm,
            torch.nn.Embedding,
            torch.nn.Conv2d,
        )
        for mn, m in model.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if (
                    pn.endswith("bias")
                    or pn.endswith("basis")
                    or pn.endswith("scale")
                    or pn.endswith("aux_tokens")
                ):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                if "lora_magnitude_vector" in pn:
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = dict(model.named_parameters())
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
        assert (
            len(param_dict.keys() - union_params) == 0
        ), f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(no_decay)],
                "weight_decay": 0.0,
            },
        ]
    else:
        optim_groups = [
            {
                "params": [p for n, p in model.named_parameters()],
                "weight_decay": 0.0,
            },
        ]

    return optim_groups
