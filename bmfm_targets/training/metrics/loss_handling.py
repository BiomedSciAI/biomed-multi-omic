import logging
from typing import Literal

import torch

from bmfm_targets.config import FieldInfo, LabelColumnInfo
from bmfm_targets.tokenization import MultiFieldTokenizer
from bmfm_targets.training import metrics

logger = logging.getLogger(__name__)


def lookup_wced_output_index(loss_name: str, field: FieldInfo) -> int | None:
    for name, kwargs in field.decode_modes.items():
        if name == "wced":
            if len(kwargs["logit_outputs"]) > 1:
                return kwargs["logit_outputs"].index(loss_name)
            else:
                return None
    raise ValueError("No WCED decoder found for field!")


class LossTask:
    def __init__(self, loss_name: str, output_size: int, weight: float = 1.0):
        """
        Base class for all loss tasks.

        Args:
        ----
            loss_name (str): Name of the loss function (e.g., 'cross_entropy').
            weight (float): Weight for the loss.

        """
        self.loss_name = loss_name
        self.output_size = output_size
        self.weight = weight

    def calculate_loss(
        self, logits: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method.")

    def get_predictions(self, logits: dict[str, torch.Tensor]):
        raise NotImplementedError("Subclasses must implement this method.")

    def get_logits(self, logits: dict[str, torch.Tensor]):
        raise NotImplementedError("Subclasses must implement this method.")

    def output_suffix_for_loss(self) -> str:
        raise NotImplementedError("Subclasses must implement this method.")

    def extract_metric_inputs(
        self, logits: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]
    ) -> tuple[str, torch.Tensor, torch.Tensor]:
        """
        Extract and format model outputs and labels for metric calculation.

        Processes raw model logits and ground truth labels into the appropriate format
        for metric evaluation, handling different loss types appropriately.

        Args:
        ----
            logits: Dictionary of model logits
            labels: Dictionary of ground truth labels

        Returns:
        -------
            tuple: (model_outputs, gt_labels) where:
                - model_outputs: Processed model predictions/logits in appropriate format
                - gt_labels: Processed ground truth labels in appropriate format
        """
        these_labels = labels[self.output_key]
        if getattr(self, "label_set", None) is not None:
            these_labels = these_labels[self.label_set]
        if self.loss_name in (
            "mse",
            "token_mse",
            "is_zero_bce",
            "is_zero_focal",
            "mae",
        ):
            model_outputs = self.get_predictions(logits)
            label_dtype = model_outputs.dtype
            gt_labels = these_labels.to(label_dtype).view(model_outputs.shape)
        # Multi-label binary class...
        elif self.loss_name == "BCEWithLogitsLoss":
            model_outputs = torch.sigmoid(
                self.get_logits(logits).view(-1, self.output_size)
            )
            # Don't flatten for multi-label binary classification to preserve class structure
            gt_labels = these_labels.to(torch.int64)
        else:  # Multiclass classification
            model_outputs = self.get_logits(logits).view(-1, self.output_size)
            gt_labels = these_labels.to(torch.int64).view(-1)
        return self.metric_key, model_outputs, gt_labels

    @property
    def output_key(self) -> str:
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    def metric_key(self) -> str:
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    def logit_key(self) -> str:
        raise NotImplementedError("Subclasses must implement this method.")

    def concat_batch_tensors(self, batch, outputs, predictions):
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    def loss_display_name(self):
        loss_display_name_components = [self.output_key, self.loss_name, "loss"]
        if getattr(self, "loss_group", None):
            loss_display_name_components = [
                self.output_key,
                self.loss_group,
                self.loss_name,
                "loss",
            ]

        loss_display_name = "_".join(loss_display_name_components)
        return loss_display_name


class FieldLossTask(LossTask):
    def __init__(
        self,
        field: FieldInfo,
        loss_name: str,
        weight: float = 1.0,
        label_smoothing: float = 0.01,
        focal_gamma: float = 0.0,
        ignore_zero: bool = False,
        link_function: str | None = None,
        decoder_key: str | None = None,
        token_values: list[float] | None = None,
        loss_group: str | None = None,
        wced_target: str | None = None,
        shrinkage: float = 0.0,
    ):
        """
        Initialize a FieldLossTask.

        Args:
        ----
            field: FieldInfo object
            loss_name: Name of the loss function.
            weight: Weight for the loss.
            label_smoothing: Label smoothing factor for classification loss.
            focal_gamma: The focal loss' focusing parameter. Higher gamma means more focus on hard examples
                            (those with low prediction confidence).
            ignore_zero: Whether to ignore zeros in the loss computation.
            link_function: Optional transformation applied to logits.
            token_values: list of ints

        """
        output_size = self._get_num_classes_for_field(loss_name, field)
        super().__init__(loss_name, output_size, weight)
        self.field = field
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        self.ignore_zero = ignore_zero
        self.link_function = link_function
        self.token_values = token_values
        self.decoder_key = decoder_key
        self.loss_group = loss_group
        self.shrinkage = shrinkage

        # only used for wced
        self.decode_token_index = None
        self.decoder_output_index = None
        self.label_set = None
        self.wced_target = wced_target
        if wced_target is not None:
            self.decode_token_index = 0
            self.decoder_output_index = lookup_wced_output_index(loss_name, field)
            vocab_field = field.decode_modes["wced"]["vocab_field"]
            self.label_set = wced_target.rstrip(f"_{vocab_field}")
            if decoder_key is None:
                self.decoder_key = f"{self.field.field_name}_wced"
            if loss_group is None:
                self.loss_group = wced_target

    @property
    def output_key(self):
        return self.field.field_name

    @property
    def metric_key(self):
        if self.loss_group:
            return self.field.field_name + "_" + self.loss_group
        else:
            return self.field.field_name

    @property
    def logit_key(self):
        return (
            self.decoder_key
            if self.decoder_key is not None
            else self.field.field_name + self.output_suffix_for_loss()
        )

    @classmethod
    def from_loss_request(
        cls, loss_request: dict, field: FieldInfo, token_values: list[float] | None
    ) -> "FieldLossTask":
        """
        Factory method to create a FieldLossTask from a loss request and a single field.

        Args:
        ----
            loss_request (dict): Loss task configuration.
            field (Field): The Field object associated with this loss task.

        Returns:
        -------
            FieldLossTask: An instantiated FieldLossTask.

        """
        exclude_keys = ["field_name", "name"]
        kwargs = {k: v for k, v in loss_request.items() if not k in exclude_keys}
        kwargs["loss_name"] = loss_request.get("name", "cross_entropy")
        return cls(field=field, token_values=token_values, **kwargs)

    @staticmethod
    def _get_num_classes_for_field(loss_name: str, field: FieldInfo) -> int:
        """
        Determine the number of classes based on the loss and field.

        Args:
        ----
            loss_name (str): The name of the loss function.
            field (Field): The Field object associated with this loss task.

        Returns:
        -------
            int: The number of classes for the field.

        Raises:
        ------
            ValueError: If the field and loss name combination are unsupported.

        """
        if loss_name in ("token_value", "cross_entropy", "focal"):
            return field.vocab_size
        # if "regression" in field.decode_modes and loss_name in (
        if loss_name in ("token_mse", "mse", "is_zero_bce", "is_zero_focal", "mae"):
            return 1

        raise ValueError(
            f"Unable to deduce number of labels for loss '{loss_name}' and field '{field.field_name}'"
        )

    def get_predictions(self, logits: dict[str, torch.Tensor]):
        these_logits = self.get_logits(logits)

        if self.loss_name in ["cross_entropy", "focal", "token_value"]:
            predictions = torch.argmax(these_logits, dim=-1)

        elif self.loss_name in ["token_mse"]:
            predictions = these_logits
            if self.link_function == "exp":
                predictions = torch.exp(predictions)
            predictions = torch.round(predictions)
            predictions = torch.clip(predictions, 0, self.field.vocab_size - 1)

        elif self.loss_name in ["mse", "mae"]:
            predictions = these_logits
            if self.link_function == "exp":
                predictions = torch.exp(predictions)

        elif self.loss_name in ["is_zero_bce", "is_zero_focal"]:
            # prediction for is_zero is True (aka 1) if the value is 0 and False (aka 0)
            #  if it is non-zero
            predictions = torch.where(these_logits > 0, 1, 0)

        else:
            raise ValueError("Requested predictions for field without a valid loss.")

        return predictions.view(predictions.shape[0], -1)

    def get_logits(self, logits: dict[str, torch.Tensor]):
        these_logits = logits[self.logit_key]
        if self.decoder_output_index is not None:
            these_logits = these_logits[..., self.decoder_output_index]
        if self.wced_target is not None:
            these_logits = these_logits[:, self.decode_token_index, :]

        return these_logits

    def _construct_loss_key(self) -> str:
        return f"{self.field.field_name}_{self.loss_name}_loss"

    def output_suffix_for_loss(self) -> str:
        """
        For a given field, return the matching suffix to the loss name.

        Raises
        ------
            ValueError: If an unsupported loss_name is received

        Returns
        -------
            str: the appropriate suffix to access the relevant logits for the requested
              loss

        """
        if self.loss_name in ("cross_entropy", "focal", "token_value"):
            return "_token_scores"
        elif self.loss_name in ("token_mse", "mse", "mae"):
            return "_regression"
        elif self.loss_name in ("is_zero_bce"):
            return "_is_zero"

        else:
            raise ValueError("Unsupported loss name: " + self.loss_name)

    def calculate_loss(
        self, logits: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Calculate a single loss based on task loss for a specific field.

        Args:
        ----
            logits (torch.Tensor): Model output for the specific field (e.g., genes, expressions).
            labels (torch.Tensor): Ground truth token labels, which are cast to long if
            the loss is cross-entropy but may be float if the loss is regression.
            token_values (list[float] | None): Token values for token_value loss.

        Raises:
        ------
            ValueError: Raised when an unknown loss name is passed.


        Returns:
        -------
            tuple[str, torch.Tensor]: The loss name and the computed loss value.

        """
        request_logits = self.get_logits(logits).clone()
        labels = labels[self.output_key]
        if self.label_set is not None:
            labels = labels[self.label_set]

        if self.loss_name in ("token_mse", "mse"):
            if self.link_function == "exp":
                request_logits = torch.exp(request_logits)

            return metrics.mse_loss(
                request_logits.squeeze(),
                labels,
                ignore_zero=self.ignore_zero,
                shrinkage=self.shrinkage,
            )

        if self.loss_name in ("mae"):
            if self.link_function == "exp":
                request_logits = torch.exp(request_logits)

            return metrics.mae_loss(
                request_logits.squeeze(),
                labels,
                ignore_zero=self.ignore_zero,
                shrinkage=self.shrinkage,
            )

        elif self.loss_name == "cross_entropy":
            return metrics.ce_loss(
                request_logits.reshape(-1, self.output_size),
                labels.reshape(-1).long(),
                label_smoothing=self.label_smoothing,
            )

        elif self.loss_name == "focal":
            return metrics.focal_loss(
                request_logits.reshape(-1, self.output_size),
                labels.reshape(-1).long(),
                self.focal_gamma,
            )

        elif self.loss_name == "token_value":
            if self.token_values is None:
                raise ValueError(
                    f"Token values must be provided to compute '{self.loss_name}' loss."
                )
            return metrics.token_value_loss(
                request_logits.reshape(-1, self.output_size),
                labels.reshape(-1),
                self.token_values,
            )

        elif self.loss_name in ["is_zero_bce"]:
            return metrics.is_zero_bce_loss(
                request_logits.reshape(-1),
                labels.reshape(-1),
            )
        elif self.loss_name in ["is_zero_focal"]:
            return metrics.focal_loss(
                request_logits.reshape(-1),
                torch.where(labels == -100, labels, (labels == 0)).reshape(-1).long(),
                focal_gamma=self.focal_gamma,
            )

        raise ValueError(f"Unsupported loss name: {self.loss_name}")

    def concat_batch_tensors(self, batch, outputs, predictions):
        from bmfm_targets.training.metrics.batch_prediction_metrics import (
            concat_field_loss_batch_tensors,
            concat_wced_field_loss_batch_tensors,
        )

        if self.label_set is None:
            logits_to_record = {
                k: v
                for k, v in outputs.logits.items()
                if k.startswith(self.output_key) and (v.shape[-1] == 1)
            }
            batch_tensors = concat_field_loss_batch_tensors(
                input_ids=batch["input_ids"],
                predictions=predictions[self.metric_key],
                labels=batch["labels"][self.output_key],
                **logits_to_record,
            )

        else:
            logits_to_record = {
                k: v[:, self.decode_token_index, :]
                for k, v in outputs.logits.items()
                if k == self.logit_key
            }
            batch_tensors = concat_wced_field_loss_batch_tensors(
                predictions=predictions[self.metric_key],
                labels=batch["labels"][self.output_key][self.label_set],
                **logits_to_record,
            )
        return batch_tensors


class LabelLossTask(LossTask):
    """A LossTask for label-level losses, e.g., sequence classification or regression."""

    def __init__(
        self,
        label_column: LabelColumnInfo,
        loss_name: str,
        weight: float = 1.0,
        ignore_zero: bool = False,
        label_smoothing: float = 0.0,
        class_weight: list | None = None,
        focal_gamma: float = 2.0,
        link_function: Literal["exp"] | None = None,
    ):
        """
        Initialize a LabelLossTask.

        Args:
        ----
            label_column: label column object.
            loss_name: Name of the loss function.
            weight: Weight for the loss.
            **kwargs: Additional attributes specific to the task.

        """
        super().__init__(loss_name, label_column.output_size, weight)
        self.label_column = label_column
        self.ignore_zero = ignore_zero
        self.label_smoothing = label_smoothing
        self.class_weight = class_weight
        self.focal_gamma = focal_gamma
        self.link_function = link_function

    def _construct_loss_key(self) -> str:
        return f"{self.output_key}_{self.loss_name}_loss"

    @property
    def output_key(self):
        return self.label_column.label_column_name

    @property
    def metric_key(self):
        return self.output_key

    @property
    def logit_key(self):
        return self.label_column.label_column_name

    def extract_metric_inputs(
        self, logits: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]
    ) -> tuple[str, torch.Tensor, torch.Tensor]:
        if self.loss_name != "hce":
            return super().extract_metric_inputs(logits, labels)
        logit_key = self.logit_key
        logits, labels = adapt_hce_prediction_and_labels_to_metrics_entries(
            logits[logit_key], labels[logit_key]
        )
        return self.logit_key, logits, labels

    def get_predictions(self, logits: dict[str, torch.Tensor]):
        logit_key = self.logit_key
        if logits[logit_key].shape[1] == 1:
            to_return = logits[logit_key].view(-1)
            if self.link_function == "exp":
                return torch.exp(to_return)
            else:
                return to_return
        else:
            return torch.argmax(logits[logit_key], dim=1)

    def get_logits(self, logits: dict[str, torch.Tensor]):
        logit_key = self.logit_key
        return logits[logit_key].reshape(-1, self.output_size)

    def output_suffix_for_loss(self) -> str:
        """
        For a given label_column, return the matching suffix to the loss name.

        Raises
        ------
            ValueError: If an unsupported loss_name is received

        Returns
        -------
            str: the appropriate suffix to access the relevant logits for the requested
              loss

        """
        if self.loss_name in ("cross_entropy", "focal", "BCEWithLogitsLoss"):
            return "_classification"
        elif self.loss_name in ("mse", "mae"):
            return "_regression"
        else:
            raise ValueError("Unsupported loss name: " + self.loss_name)

    def calculate_loss(
        self, logits: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        logits = logits[self.logit_key]
        labels = labels[self.output_key]
        if self.link_function == "exp":
            logits = torch.exp(logits)
        return metrics.classification_loss(
            logits,
            labels,
            self.loss_name,
            self.output_size,
            ignore_zero=self.ignore_zero,
            label_smoothing=self.label_smoothing,
            class_weight=self.class_weight,
            focal_gamma=self.focal_gamma,
        )

    def concat_batch_tensors(self, batch, outputs, predictions):
        from bmfm_targets.training.metrics.batch_prediction_metrics import (
            concat_label_loss_batch_tensors,
        )

        logits_to_record = {
            k: v for k, v in outputs.logits.items() if k == self.output_key
        }
        batch_tensors = concat_label_loss_batch_tensors(
            input_ids=batch["input_ids"],
            predictions=predictions[self.metric_key],
            labels=batch["labels"][self.output_key],
            **logits_to_record,
        )
        return batch_tensors

    @classmethod
    def from_loss_request(cls, loss_request: dict, label_column: LabelColumnInfo):
        loss_name = loss_request.get("name")
        if loss_name is None:
            if label_column.is_regression_label:
                loss_name = "mse"
            else:
                loss_name = "cross_entropy"
        exclude_keys = ["label_column_name", "name"]
        kwargs = {k: v for k, v in loss_request.items() if not k in exclude_keys}
        kwargs["loss_name"] = loss_name
        return cls(label_column=label_column, **kwargs)


def get_loss_tasks(
    loss_requests: list[dict],
    tokenizer: MultiFieldTokenizer | None = None,
    fields: list[FieldInfo] | None = None,
    label_columns: list[LabelColumnInfo] | None = None,
) -> list[LossTask]:
    """
    Factory function to create multiple LossTask instances from configuration,
    focusing only on masked fields for FieldLossTask.

    Args:
    ----
        loss_requests (list[dict]): A list of dictionaries, each specifying a loss task.
        fields (list[Field] | None): A list of Field objects, or None if not applicable.
        label_columns (list[LabelColumnInfo] | None): A list of LabelColumnInfo objects, or None if not applicable.

    Returns:
    -------
        list[LossTask]: A list of instantiated loss tasks.

    """
    loss_tasks = []
    for loss_request in loss_requests:
        try:
            loss_task = make_loss_task(tokenizer, fields, label_columns, loss_request)
            loss_tasks.append(loss_task)
        except ValueError as e:
            logger.warning(
                f"Unable to create loss: {loss_request}. Failed with error {e}"
            )

    return loss_tasks


def make_loss_task(tokenizer, fields, label_columns, loss_request):
    if "field_name" in loss_request:
        field_name = loss_request["field_name"]
        matched_field = lookup_field(fields, field_name)
        token_values = tokenizer.get_token_values(field_name) if tokenizer else None
        loss_task = FieldLossTask.from_loss_request(
            loss_request, matched_field, token_values=token_values
        )
        _verify_token_values_compatible_with_loss(loss_task, tokenizer)
        return loss_task

    elif "label_column_name" in loss_request:
        return LabelLossTask.from_loss_request(
            loss_request,
            lookup_label_column(label_columns, loss_request["label_column_name"]),
        )

    raise ValueError(
        "Invalid loss request: Missing 'field_name' or 'label_column_name'."
    )


def lookup_field(fields, field_name):
    if fields is None:
        raise ValueError(
            "Fields are not provided, but a field-based loss is requested."
            "Try label-column-based loss specification in Trainer config if you run sequence_classification."
        )

    field = next((f for f in fields if f.field_name == field_name), None)
    if field:
        return field
    raise ValueError(f"Field with name '{field}' not found.")


def lookup_label_column(label_columns, label_column_name):
    if label_columns is None:
        raise ValueError(
            "Label columns are not provided, but a label-column-based loss is requested."
        )
    matched_label_column = next(
        (
            label_column
            for label_column in label_columns
            if label_column.label_column_name == label_column_name
        ),
        None,
    )
    if not matched_label_column:
        raise ValueError(f"Label column with name '{label_column_name}' not found.")

    return matched_label_column


def _verify_token_values_compatible_with_loss(
    loss_task: FieldLossTask, tokenizer: MultiFieldTokenizer | None
):
    if loss_task.loss_name != "token_mse":
        return
    if tokenizer is None:
        raise ValueError("If using token_mse, you must pass a tokenizer")
    field_name = loss_task.field.field_name
    loss_name = loss_task.loss_name
    token_values = tokenizer.get_token_values(field_name)
    assert (
        token_values is not None
    ), f"To use {loss_name} on field {field_name} you must have token values"
    special_token_count = len(tokenizer.all_special_ids)
    invalid_tokenizer_message = f"To use {loss_name} on field {field_name} you must have a vocab with consecutive integers starting at 0"
    assert token_values[special_token_count:] == [
        *range(len(token_values) - special_token_count)
    ], invalid_tokenizer_message


def calculate_losses(
    loss_tasks: list[LossTask],
    logits: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Calculates the losses across multiple tasks."""
    all_losses = {}
    total_weight, total_loss = 0, 0
    for loss_task in loss_tasks:
        loss_val = loss_task.calculate_loss(logits, labels)
        if loss_val is None or torch.isnan(loss_val):
            continue
        # *= syntax breaks when loss_val is float and weight is long
        all_losses[loss_task.loss_display_name] = loss_val

        weighted_loss_val = loss_task.weight * loss_val
        total_weight += loss_task.weight
        total_loss += weighted_loss_val

    all_losses["loss"] = (
        (total_loss / total_weight)
        if total_weight > 0
        else torch.tensor(0.0, device=[*logits.values()][0].device)
    )
    return all_losses


def calculate_predictions(
    loss_tasks: list[LossTask],
    logits: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Calculates the predictions across fields and losses."""
    from collections import defaultdict

    partial_predictions = defaultdict(dict)

    for lt in loss_tasks:
        metric_key = lt.metric_key
        pred_key = lt.loss_name  # (lt.loss_name, lt.logit_key)
        partial_predictions[metric_key][pred_key] = lt.get_predictions(logits)

    final_predictions = {}
    for metric_key in partial_predictions:
        final_predictions[metric_key] = combine_partial_predictions(
            partial_predictions[metric_key]
        )

    return final_predictions


def combine_partial_predictions(
    partial_predictions: dict[str, torch.Tensor],
) -> torch.Tensor:
    if len(partial_predictions) == 1:
        return [*partial_predictions.values()][0]
    if "mse" in partial_predictions.keys():
        is_zero_key = [k for k in partial_predictions if "is_zero" in k][0]

        return torch.where(
            partial_predictions[is_zero_key] == 1, 0, partial_predictions["mse"]
        )
    if "mae" in partial_predictions.keys():
        is_zero_key = [k for k in partial_predictions if "is_zero" in k][0]

        return torch.where(
            partial_predictions[is_zero_key] == 1, 0, partial_predictions["mae"]
        )

    # cross_entropy is the "default" -- if there are others present let's report
    # cross_entropy as the "prediction" even though all are used for the loss
    for loss_name in [
        "cross_entropy",
        "focal",
        "token_value",
        "token_mse",
        "BCEWithLogitsLoss",
    ]:
        if loss_name in partial_predictions.keys():
            return partial_predictions[loss_name]

    raise ValueError(
        f"Received non-commensurate partial predictions: {partial_predictions.keys()}"
    )


def adapt_hce_prediction_and_labels_to_metrics_entries(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """
    Finds prediction label with argmax and replaces a set of target labels with a single target label.
    If predicted label in a set of target labels, set target label equal to predicted label.
    Otherwise, set random label from the set of target labels to target label.
    If flag is zero, set target label to -100.
    """
    labels = labels.clone().detach()
    flag = labels[..., -1].to(torch.bool)
    labels = labels[..., :-1]
    pred = torch.argmax(logits, dim=1)
    mask_value_at_pred = (
        torch.gather(input=labels, dim=1, index=pred[..., None])
        .squeeze(1)
        .to(torch.bool)
    )
    labels[~flag, 0] = 1.0
    new_labels = torch.multinomial(labels, 1).squeeze(1)
    new_labels[mask_value_at_pred] = pred[mask_value_at_pred]
    new_labels[~flag] = -100
    return logits, new_labels
