"""
Utility functions for loss handling.

This module provides helper functions for loss calculation, prediction combination,
and backward compatibility with old-style dict configs.
"""

from __future__ import annotations

from collections import defaultdict

import torch

from bmfm_targets.config.model_config import FieldInfo, LabelColumnInfo
from bmfm_targets.tokenization.multifield_tokenizer import MultiFieldTokenizer

from .compat import loss_dict_to_task
from .task import LossTask


def get_loss_tasks(
    losses: list[dict | LossTask],
    fields: list[FieldInfo] | None = None,
    label_columns: list[LabelColumnInfo] | None = None,
    tokenizer: MultiFieldTokenizer | None = None,
) -> list[LossTask]:
    """
    Backward compatibility function for creating loss tasks from dict configs.

    This function provides compatibility with code that uses the old factory pattern.
    New code should use Hydra instantiation at the config layer.

    Args:
    ----
        losses: List of loss configuration dicts or LossTask instances
        fields: List of FieldInfo objects (optional, for field-based losses)
        label_columns: List of LabelColumnInfo objects (for label-based losses)
        tokenizer: Optional tokenizer for token value objectives

    Returns:
    -------
        list[LossTask]: List of instantiated and bound loss tasks

    Raises:
    ------
        ValueError: If Hydra config (_target_) is passed (should be instantiated at config layer)

    """
    loss_tasks = []
    fields = fields or []
    label_columns = label_columns or []

    for loss_config in losses:
        # If it's already a LossTask, bind and use it
        if isinstance(loss_config, LossTask):
            loss_config.bind(fields, label_columns, tokenizer)
            loss_tasks.append(loss_config)
            continue

        # If it has _target_, it should have been instantiated at config layer
        if isinstance(loss_config, dict) and "_target_" in loss_config:
            raise ValueError(
                "Hydra configs with _target_ should be instantiated at config layer. "
                "This function only handles old-style dict configs and already-instantiated LossTask objects. "
                f"Got: {loss_config}"
            )

        # Otherwise, use compat module to convert old-style dict config
        loss_task = loss_dict_to_task(loss_config, fields, label_columns)
        loss_task.bind(fields, label_columns, tokenizer)
        loss_tasks.append(loss_task)

    return loss_tasks


def calculate_losses(
    loss_tasks: list[LossTask],
    logits: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Calculates the losses across multiple tasks."""
    all_losses = {}
    total_weight, total_loss = 0, 0

    # Create mock objects for new LossTask interface
    mock_outputs = type("Outputs", (), {"logits": logits})()
    mock_batch = {"labels": labels}

    for loss_task in loss_tasks:
        loss_val = loss_task.calculate_loss(mock_outputs, mock_batch)
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
    partial_predictions = defaultdict(dict)

    for lt in loss_tasks:
        # Use new interface: source.name for metric key
        metric_key = lt.metric_key
        # Use objective name as pred_key
        pred_key = lt.objective.name
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
    """
    Combine partial predictions from multiple objectives for the same field.

    Args:
    ----
        partial_predictions: Dict mapping objective names to predictions

    Returns:
    -------
        torch.Tensor: Combined predictions

    """
    if len(partial_predictions) == 1:
        return [*partial_predictions.values()][0]

    # Check if there's an is_zero prediction to combine with mse/mae
    is_zero_keys = [k for k in partial_predictions if "is_zero" in k]

    if "mse" in partial_predictions.keys() and is_zero_keys:
        is_zero_key = is_zero_keys[0]
        return torch.where(
            partial_predictions[is_zero_key] == 1, 0, partial_predictions["mse"]
        )
    if "mae" in partial_predictions.keys() and is_zero_keys:
        is_zero_key = is_zero_keys[0]
        return torch.where(
            partial_predictions[is_zero_key] == 1, 0, partial_predictions["mae"]
        )

    # cross_entropy is the "default" -- if there are others present let's report
    # cross_entropy as the "prediction" even though all are used for the loss
    for loss_name in [
        "cross_entropy",
        "focal",
        "token_value",
        "mse",
        "mse",
        "mae",
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
    Adapt HCE predictions and labels for metric calculation.

    Finds prediction label with argmax and replaces a set of target labels with a single target label.
    If predicted label in a set of target labels, set target label equal to predicted label.
    Otherwise, set random label from the set of target labels to target label.
    If flag is zero, set target label to -100.

    Args:
    ----
        logits: Model logits
        labels: Ground truth labels (last dimension is flag)

    Returns:
    -------
        tuple: (logits, adapted_labels)

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


# Made with Bob
