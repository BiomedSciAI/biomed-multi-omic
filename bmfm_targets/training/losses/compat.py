"""
Backward compatibility layer for old-style dict configs.

This module provides functions to convert old-style loss configuration dictionaries
into new LossTask instances. This allows existing configs to work without modification
while new configs can use Hydra instantiation directly.
"""

from __future__ import annotations

from bmfm_targets.config.model_config import FieldInfo, LabelColumnInfo

from .objectives import (
    BCEWithLogitsObjective,
    CrossEntropyObjective,
    FocalObjective,
    HCEObjective,
    IsZeroBCEObjective,
    IsZeroFocalObjective,
    MAEObjective,
    MSEObjective,
    TokenValueObjective,
)
from .sources import FieldSource, LabelSource, WCEDFieldSource
from .task import LossTask


def loss_dict_to_task(
    loss_config: dict,
    fields: list[FieldInfo],
    label_columns: list[LabelColumnInfo],
) -> LossTask:
    """
    Convert old-style loss dict config to LossTask.

    This function handles backward compatibility with configs that use dicts
    instead of Hydra instantiation.

    Args:
    ----
        loss_config: Loss configuration dict with either:
            - label_column_name: For label-based losses
            - field_name: For field-based losses
        fields: List of FieldInfo objects
        label_columns: List of LabelColumnInfo objects

    Returns:
    -------
        LossTask: Instantiated (but not bound) loss task

    Raises:
    ------
        ValueError: If config is invalid or references unknown field/label

    Example Config Keys:
    -------------------
        field_name: Name of field (required for field losses)
        label_column_name: Name of label column (required for label losses)
        name: Loss type (mse, cross_entropy, focal, etc.)
        weight: Loss weight (default: 1.0)
        decoder_key: Explicit decoder key (optional, for MVC etc.)
        loss_group: Metric grouping (optional, auto-derived for WCED)
        wced_target: WCED target name (optional)

        # Loss-specific parameters:
        label_smoothing / smoothing: For cross_entropy
        focal_gamma: For focal losses
        ignore_zero: For regression losses
        link_function: For regression losses (e.g., "exp")
        shrinkage: For regression losses
        ignore_index: For classification losses

    """
    if "label_column_name" in loss_config:
        return _create_label_loss(loss_config, label_columns)
    elif "field_name" in loss_config:
        return _create_field_loss(loss_config, fields)
    else:
        raise ValueError(
            f"Invalid loss config: must have 'field_name' or 'label_column_name'. "
            f"Got: {loss_config}"
        )


def _create_label_loss(
    loss_config: dict,
    label_columns: list[LabelColumnInfo],
) -> LossTask:
    """Create LossTask for label-based loss."""
    label_name = loss_config["label_column_name"]
    weight = loss_config.get("weight", 1.0)
    loss_group = loss_config.get("loss_group")
    metrics = loss_config.get("metrics")

    # Find label column
    label_column = next(
        (lc for lc in label_columns if lc.label_column_name == label_name),
        None,
    )
    if not label_column:
        raise ValueError(f"Label column '{label_name}' not found")

    # Auto-detect loss type if not specified
    if "name" in loss_config:
        loss_name = loss_config["name"]
    else:
        # Auto-detect: regression (n_unique_values==1) vs classification
        loss_name = "mse" if label_column.n_unique_values == 1 else "cross_entropy"

    # Create appropriate objective
    objective = _create_objective(loss_name, loss_config)

    return LossTask(
        source=LabelSource(label_name=label_name),
        objective=objective,
        weight=weight,
        metrics=metrics,
        loss_group=loss_group,
    )


def _create_field_loss(
    loss_config: dict,
    fields: list[FieldInfo],
) -> LossTask:
    """Create LossTask for field-based loss."""
    field_name = loss_config["field_name"]
    loss_name = loss_config.get("name", "cross_entropy")
    weight = loss_config.get("weight", 1.0)
    loss_group = loss_config.get("loss_group")
    metrics = loss_config.get("metrics")
    wced_target = loss_config.get("wced_target")

    # Get explicit decoder_key if provided, otherwise let it be derived
    decoder_key = loss_config.get("decoder_key")

    # Find field (for validation only - actual resolution happens in bind())
    field = next((f for f in fields if f.field_name == field_name), None)
    if not field:
        raise ValueError(f"Field '{field_name}' not found")

    # Create appropriate objective
    objective = _create_objective(loss_name, loss_config)

    # Use WCEDFieldSource for WCED losses, FieldSource otherwise
    if wced_target is not None:
        source = WCEDFieldSource(
            field_name=field_name,
            wced_target=wced_target,
        )

    else:
        source = FieldSource(
            field_name=field_name,
            decoder_key=decoder_key,
        )

    return LossTask(
        source=source,
        objective=objective,
        weight=weight,
        metrics=metrics,
        loss_group=loss_group,
    )


def _create_objective(loss_name: str, loss_config: dict):
    """Create objective from loss name and config."""
    if loss_name == "cross_entropy":
        return CrossEntropyObjective(
            label_smoothing=loss_config.get(
                "label_smoothing", loss_config.get("smoothing", 0.0)
            ),
            ignore_index=loss_config.get("ignore_index", -100),
        )
    elif loss_name == "focal":
        return FocalObjective(
            focal_gamma=loss_config.get("focal_gamma", 2.0),
            ignore_index=loss_config.get("ignore_index", -100),
        )
    elif loss_name in ("mse"):
        return MSEObjective(
            ignore_zero=loss_config.get("ignore_zero", False),
            link_function=loss_config.get("link_function"),
            shrinkage=loss_config.get("shrinkage", 0.0),
        )
    elif loss_name == "mae":
        return MAEObjective(
            ignore_zero=loss_config.get("ignore_zero", False),
            link_function=loss_config.get("link_function"),
            shrinkage=loss_config.get("shrinkage", 0.0),
        )
    elif loss_name == "token_value":
        return TokenValueObjective()
    elif loss_name == "is_zero_bce":
        return IsZeroBCEObjective()
    elif loss_name == "is_zero_focal":
        return IsZeroFocalObjective(focal_gamma=loss_config.get("focal_gamma", 2.0))
    elif loss_name in ("bce_with_logits", "BCEWithLogitsLoss"):
        return BCEWithLogitsObjective()
    elif loss_name == "hce":
        return HCEObjective()
    else:
        raise ValueError(f"Unknown loss name: {loss_name}")


# Made with Bob
