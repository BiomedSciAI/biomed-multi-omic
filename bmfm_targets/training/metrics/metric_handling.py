"""Functions for handling the metrics functions requested during training."""
import inspect
import logging

import numpy as np
import torchmetrics
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef

from bmfm_targets.training.metrics import get_token_labels

from .metric_functions import NonZeroBinaryConfusionMatrix, Perplexity

logger = logging.getLogger(__name__)

KNOWN_CATEGORICAL_METRICS = {
    "accuracy": torchmetrics.Accuracy,
    "f1": torchmetrics.F1Score,
    "mcc": torchmetrics.MatthewsCorrCoef,
    "precision": torchmetrics.Precision,
    "recall": torchmetrics.Recall,
    "auc": torchmetrics.AUROC,
    "auprc": torchmetrics.AveragePrecision,
    "confusion_matrix": MulticlassConfusionMatrix,
    "perplexity": Perplexity,
}
KNOWN_REGRESSION_METRICS = {
    "mae": MeanAbsoluteError,
    "mse": MeanSquaredError,
    "pcc": PearsonCorrCoef,
    "nonzero_confusion_matrix": NonZeroBinaryConfusionMatrix,
}
DEFAULT_CATEGORICAL_KWARGS = {"ignore_index": -100, "task": "multiclass"}

SPECIAL_DEFAULT_CATEGORICAL_KWARGS = {
    "f1": {"ignore_index": -100, "task": "multiclass", "average": "macro"},
    "accuracy": {"ignore_index": -100, "task": "multiclass", "average": "macro"},
    "precision": {"ignore_index": -100, "task": "multiclass", "average": "macro"},
    "recall": {"ignore_index": -100, "task": "multiclass", "average": "macro"},
    "confusion_matrix": {"ignore_index": -100, "normalize": None},
    "perplexity": {"ignore_index": -100},
}


def _filter_unsupported_kwargs(metric_class: type, kwargs: dict) -> dict:
    """Filter kwargs to only those supported by metric_class.__init__ or __new__."""
    # Check __new__ first (some metrics like Accuracy use it), fall back to __init__
    try:
        sig = inspect.signature(metric_class.__new__)
    except (ValueError, TypeError):
        sig = inspect.signature(metric_class.__init__)
    supported = set(sig.parameters.keys()) - {"self", "cls"}
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    if dropped := set(kwargs) - set(filtered):
        logger.warning(
            f"{metric_class.__name__} ignoring unsupported params: {dropped}"
        )
    return filtered


def get_metric_object(mt: dict, num_classes: int) -> torchmetrics.Metric:
    """
    Construct metric based on metric request dict and number of classes.

    Args:
    ----
        mt (dict): metric request dict
        num_classes (int): number of classes, 1 for regression

    Returns:
    -------
        torchmetrics.Metric: metric object
    """
    if num_classes > 1:
        return _get_categorical_metric(mt, num_classes)
    else:
        return _get_regression_metric(mt)


def _get_categorical_metric(mt: dict, num_classes: int) -> torchmetrics.Metric:
    kwargs = {"num_classes": num_classes}
    kwargs.update(
        SPECIAL_DEFAULT_CATEGORICAL_KWARGS.get(mt["name"], DEFAULT_CATEGORICAL_KWARGS)
    )
    kwargs.update({k: v for k, v in mt.items() if k != "name"})
    if "task" in kwargs and kwargs["task"] == "multilabel":
        kwargs["num_labels"] = num_classes
        del kwargs["num_classes"]

    metric_class = KNOWN_CATEGORICAL_METRICS[mt["name"]]
    return metric_class(**_filter_unsupported_kwargs(metric_class, kwargs))


def _get_regression_metric(mt: dict) -> torchmetrics.Metric:
    kwargs = {k: v for k, v in mt.items() if k != "name"}
    metric_class = KNOWN_REGRESSION_METRICS[mt["name"]]
    return metric_class(**_filter_unsupported_kwargs(metric_class, kwargs))


def limit_confusion_matrix_to_numerical_labels(token_values, cm_original):
    keep_idx, field_labels = get_token_labels(token_values)
    cm = cm_original[np.ix_(keep_idx, keep_idx)]

    return field_labels, cm
