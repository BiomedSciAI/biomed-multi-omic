"""Functions for handling the metrics functions requested during training."""
import numpy as np
import torchmetrics
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef

from bmfm_targets.training.metrics import get_token_labels

from .metric_functions import NonZeroBinaryConfusionMatrix

KNOWN_CATEGORICAL_METRICS = {
    "accuracy": torchmetrics.Accuracy,
    "f1": torchmetrics.F1Score,
    "mcc": torchmetrics.MatthewsCorrCoef,
    "precision": torchmetrics.Precision,
    "recall": torchmetrics.Recall,
    "auc": torchmetrics.AUROC,
    "auprc": torchmetrics.AveragePrecision,
    "confusion_matrix": MulticlassConfusionMatrix,
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
    "confusion_matrix": {"ignore_index": -100, "normalize": None},
}


def get_relevant_metrics(
    desired_metrics: list[dict], output_size: int, max_confusion_matrix_size=200
) -> list[dict]:
    """
    Limit the list of all desired metrics to the relevant metrics.

    We begin with an inclusive list of all desired metrics but some are not relevant
    because some are categorical only, regression only, or if there are too many outputs
    for a confusion matrix. This function restricts the full list to those which are relevant
    for this output.

    Args:
    ----
        desired_metrics (list[dict]): the full list of desired metrics, usually from trainer_config
        output_size (int): number of labels for this task (1 for regression, >1 for categorical)
        max_confusion_matrix_size (int, optional): Maximum confusion matrix size. Defaults to 200.

    Returns:
    -------
        list[dict]: list of metrics which can be instantiated for this label
    """
    if output_size == 1:
        return [mt for mt in desired_metrics if mt["name"] in KNOWN_REGRESSION_METRICS]
    metrics = [mt for mt in desired_metrics if mt["name"] in KNOWN_CATEGORICAL_METRICS]
    if output_size > max_confusion_matrix_size:
        metrics = [mt for mt in metrics if mt["name"] != "confusion_matrix"]
    return metrics


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
    kwargs = {"num_classes": num_classes}  # default task is multiclass...
    kwargs.update(
        SPECIAL_DEFAULT_CATEGORICAL_KWARGS.get(mt["name"], DEFAULT_CATEGORICAL_KWARGS)
    )
    kwargs.update({k: v for k, v in mt.items() if k != "name"})
    if "task" in kwargs and kwargs["task"] == "multilabel":
        kwargs["num_labels"] = num_classes
        del kwargs["num_classes"]
    return KNOWN_CATEGORICAL_METRICS[mt["name"]](**kwargs)


def _get_regression_metric(mt: dict) -> torchmetrics.Metric:
    kwargs = {}
    kwargs.update({k: v for k, v in mt.items() if k != "name"})
    return KNOWN_REGRESSION_METRICS[mt["name"]](**kwargs)


def limit_confusion_matrix_to_numerical_labels(token_values, cm_original):
    keep_idx, field_labels = get_token_labels(token_values)
    cm = cm_original[np.ix_(keep_idx, keep_idx)]

    return field_labels, cm
