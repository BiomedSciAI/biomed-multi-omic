import logging

import numpy as np
import scipy.stats as stats
import torch
import torch.nn.functional as F
from clearml.logger import Logger
from focal_loss.focal_loss import (
    FocalLoss,  # https://github.com/mathiaszinnen/focal_loss_torch
)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torchmetrics import Metric
from torchmetrics.classification import BinaryConfusionMatrix


class NonZeroBinaryConfusionMatrix(BinaryConfusionMatrix):
    def __init__(
        self, ignore_index: int = -100, threshold: float = 0.5, normalize="none"
    ):
        super().__init__(
            ignore_index=ignore_index, threshold=threshold, normalize=normalize
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        valid_mask = target != self.ignore_index

        preds = preds[valid_mask]
        target = target[valid_mask]
        preds = (preds == 0).to(torch.int)
        target = (target == 0).to(torch.int)
        super().update(preds, target)


class MaskedAccuracy(Metric):
    """
    Computes the accuracy over a batch of data.

    Attributes
    ----------
        ignore_index: The index to ignore when computing the accuracy.
        dist_sync_on_step: Whether to synchronize the metric state across processes at each ``forward()``
            before returning the value at the step. Defaults to ``False``.
    """

    full_state_update: bool = False
    higher_is_better: bool | None = True
    is_differentiable: bool = False

    def __init__(
        self,
        field_names: list[str],
        ignore_index: int = -100,
        dist_sync_on_step: bool = False,
    ):
        """
        Initializes the metric.

        Args:
        ----
            ignore_index: The index to ignore when computing the accuracy.
            dist_sync_on_step: Whether to synchronize the metric state across processes at each ``forward()``
                before returning the value at the step. Defaults to ``False``.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.ignore_index = ignore_index

        for field_name in field_names:
            setattr(self, field_name + "_correct", torch.tensor(0))
            setattr(self, field_name + "_total", torch.tensor(0))
            self.add_state(
                field_name + "_correct", default=torch.tensor(0), dist_reduce_fx="sum"
            )
            self.add_state(
                field_name + "_total", default=torch.tensor(0), dist_reduce_fx="sum"
            )

    def update(self, preds: torch.Tensor, target: torch.Tensor, field_name: str):  # type: ignore[override]
        """
        Updates the accuracy.

        Args:
        ----
            preds: The predictions.
            target: The targets.
        """
        assert preds.shape == target.shape, f"{preds.shape} != {target.shape}"
        mask = target.ne(self.ignore_index)
        if mask.any():
            correct = torch.eq(preds, target)[mask].sum()
            total = mask.sum()
            self.__dict__[field_name + "_correct"] += correct
            self.__dict__[field_name + "_total"] += total

    def compute(self, field_name: str) -> torch.Tensor:  # type: ignore[override]
        """
        Computes the accuracy.


        Returns
        -------
            torch.Tensor: The accuracy.
        """
        return (
            self.__dict__[field_name + "_correct"].float()
            / self.__dict__[field_name + "_total"]
        )


def ce_loss(logits, labels, label_smoothing=0.01):
    """Calculates cross entropy loss function."""
    loss_fct = CrossEntropyLoss(label_smoothing=label_smoothing)
    return loss_fct(logits, labels)


def focal_loss(logits, labels, focal_gamma=2.0, **kwargs):
    """
    Calculates focal loss. Default value of focal_gamma values returns the cross entropy loss.
    focal_gamma is the focousing parameter, i.e. it reduces the relative loss for well-classified examples.
    Classes can have different weights by using the weights parameter (default is torch.FloatTensor([1, 1, 1])),
    in order to get a higher importance for rare classes.
    """
    if len(logits.shape) == 1:
        m = torch.nn.Sigmoid()
    else:
        m = torch.nn.Softmax(dim=-1)

    criterion = FocalLoss(
        gamma=focal_gamma,
        ignore_index=-100,
        eps=torch.finfo(logits.dtype).eps,
        **kwargs,
    )
    probs = m(logits)
    return criterion(probs, labels)


def weighted_logsumexp(x, weights, dim=None, keepdim=False):
    """
    Numerically stable log-sum-exp implementation with weights.

    Args:
    ----
        x (Tensor): Input tensor.
        dim (int or tuple of ints, optional): Dimension(s) to reduce.
        keepdim (bool): Whether to keep reduced dimensions.

    Returns:
    -------
        Tensor: log(sum(exp(x))).
    """
    max_x, _ = (
        torch.max(x, dim=dim, keepdim=True) if dim is not None else (torch.max(x), None)
    )
    stable_x = x - max_x if dim is not None else x - max_x
    sum_exp = torch.sum(
        weights * torch.exp(stable_x), dim=dim, keepdim=True if keepdim else False
    )
    out = torch.log(sum_exp) + (max_x if keepdim or dim is None else max_x.squeeze(dim))
    return out


def hce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Hierarchical cross-entropy loss.

    Args:
    ----
    labels (torch.Tensor): A tensor containing 0s and 1s, where 1 indicates a
        selected label. The last element serves as a flag:
        - 0 if all values are 0
        - 1 otherwise
    logits (torch.Tensor): logits of predictions
    """
    flag = labels[..., -1]
    labels = labels[..., :-1]
    labels = labels - flag[..., None] + 1.0  # Set all 1 if flag is 0 to avoid NANs
    loss = -torch.mean(
        (
            weighted_logsumexp(x=logits, weights=labels, dim=-1)
            - torch.logsumexp(logits, dim=-1)
        )
        * flag
    )
    return loss


class TokenValueLoss(torch.nn.Module):
    """
    Calculated the difference between a token's *value* and the predicted logits.

    This metric performs a weighted average of the predicted tokens and compares it to
    the ground truth tokens, returning the mean squared error.

    """

    def __init__(self, token_values: list[float]):
        """
        Args:
        ----
            token_values: list[float] a list of token values indexed by token_id. The
                first few values are presumed to be nan and discarded, because the
                special tokens have no "value". Those logits are also ignored.
        """
        super().__init__()
        self.token_values = token_values

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        token_values = torch.Tensor(self.token_values).to(logits.device)

        logits = logits[..., ~token_values.isnan()]

        probs = F.softmax(logits, dim=-1)

        preds = torch.sum(probs * token_values[~token_values.isnan()], dim=-1)

        ground_truth_values = token_values[labels.int()]

        token_value_loss = F.mse_loss(preds, ground_truth_values.float())

        return token_value_loss


def token_value_loss(logits, labels, token_values, ignore_index=-100):
    ignore_mask = labels == ignore_index
    token_value_loss = TokenValueLoss(token_values)
    field_loss = token_value_loss(logits[~ignore_mask], labels[~ignore_mask])

    return field_loss


def classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor | None,
    loss_name: str | None = None,
    output_size: int | None = None,
    problem_type: str | None = None,
    ignore_zero: bool = False,
    label_smoothing: float = 0.0,
    class_weight: list | None = None,
    focal_gamma: float = None,
):
    if labels is None:
        return None
    if loss_name == "hce":
        return hce_loss(logits, labels)
    if problem_type is None:
        problem_type = deduce_problem_type(labels, output_size)
    if problem_type == "regression" and loss_name == "mse":
        loss = mse_loss(logits.squeeze(), labels, ignore_zero=ignore_zero)
    elif problem_type == "regression" and loss_name == "mae":
        loss = mae_loss(logits.squeeze(), labels, ignore_zero=ignore_zero)
    elif problem_type == "single_label_classification":
        if loss_name == "focal":
            loss = focal_loss(
                logits.view(-1, output_size), labels.view(-1), focal_gamma=focal_gamma
            )
        elif loss_name == "cross_entropy":
            loss = ce_loss(
                logits.view(-1, output_size),
                labels.view(-1),
                label_smoothing=label_smoothing,
            )
    elif (
        problem_type == "multi_label_classification"
        and loss_name == "BCEWithLogitsLoss"
    ):
        if class_weight and len(class_weight) == 1:
            class_weight = torch.tensor(class_weight * output_size).to(labels.device)
        loss_fct = BCEWithLogitsLoss(pos_weight=class_weight)
        loss = loss_fct(logits, labels)
    else:
        raise ValueError(
            f"Unsupported problem type: {problem_type} for the loss_name {loss_name}."
        )
    return loss


def deduce_problem_type(labels: torch.Tensor, output_size: int):
    if output_size == 1:
        problem_type = "regression"
    elif output_size > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
        problem_type = "single_label_classification"
    else:
        problem_type = "multi_label_classification"
    return problem_type


def mse_loss(logits, labels, ignore_index=-100.0, ignore_zero=False, shrinkage=0.0):
    """
    MSE loss with optional L2 shrinkage towards zero prediction.

    Args:
    ----
        shrinkage: L2 penalty weight on predictions (Ridge-style). Encourages predicting zero.
    """
    if labels.shape != logits.shape:
        labels = labels.view(logits.shape)

    if ignore_zero:
        ignore_mask = (labels == ignore_index) | (labels == 0.0)
    else:
        ignore_mask = labels == ignore_index

    valid_mask = ~ignore_mask
    n_valid = valid_mask.sum()

    # Main loss
    loss_unreduced = F.mse_loss(logits, labels.to(logits.dtype), reduction="none")
    loss = (loss_unreduced * valid_mask).sum() / (n_valid + 1e-6)

    # L2 shrinkage on predictions (encourages zero)
    if shrinkage > 0:
        shrink_loss = (logits.pow(2) * valid_mask).sum() / (n_valid + 1e-6)
        loss = loss + shrinkage * shrink_loss

    return loss


def mae_loss(
    logits,
    labels,
    ignore_index=-100.0,
    ignore_zero=False,
    shrinkage=0.0,
    soft_threshold=0.0,
):
    """
    MAE loss with optional shrinkage.

    Args:
    ----
        shrinkage: L1 penalty weight on predictions (Lasso-style). Encourages predicting zero.
        soft_threshold: Soft thresholding on residuals before computing loss.
                       Treats small deviations as zero. Good for your deviation prediction use case.
    """
    if labels.shape != logits.shape:
        labels = labels.view(logits.shape)

    if ignore_zero:
        ignore_mask = (labels == ignore_index) | (labels == 0.0)
    else:
        ignore_mask = labels == ignore_index

    valid_mask = ~ignore_mask
    n_valid = valid_mask.to(torch.float64).sum()

    # Compute residuals
    residuals = logits - labels.to(logits.dtype)

    # Optional soft thresholding (dampens small deviations)
    if soft_threshold > 0:
        residuals = torch.sign(residuals) * F.relu(residuals.abs() - soft_threshold)

    # Main loss
    loss = (residuals.abs() * valid_mask).sum() / (n_valid + 1e-6)

    # L1 shrinkage on predictions (encourages zero)
    if shrinkage > 0:
        shrink_loss = (logits.abs() * valid_mask).sum() / (n_valid + 1e-6)
        loss = loss + shrinkage * shrink_loss

    return loss


def is_zero_bce_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index=-100):
    ignore_mask = labels == ignore_index
    n_valid_elements = (~ignore_mask).sum()
    field_loss = F.binary_cross_entropy_with_logits(
        logits, (labels == 0.0).float(), reduction="none"
    )
    field_masked_loss = field_loss * ~ignore_mask
    field_loss = field_masked_loss.sum() / (n_valid_elements + 1e-6)
    return field_loss


def masked_mean(last_hidden_states, attention_mask, dim=1):
    masked_hidden_states = last_hidden_states * attention_mask.unsqueeze(-1).float()
    sum_hidden_states = torch.sum(masked_hidden_states, dim=dim)
    count_nonzero = torch.clamp(torch.sum(attention_mask, dim=dim, keepdim=True), min=1)
    mean_pooled_hidden_states = sum_hidden_states / count_nonzero.float()
    return mean_pooled_hidden_states


def log_confusion_matrix_to_clearml(cm, prefix, labels, title, iteration=None):
    cm_normalized_by_targets = np.where(
        cm.sum(axis=1, keepdims=True) > 0, cm / cm.sum(axis=1, keepdims=True), 0
    )
    cm_normalized_by_predictions = np.where(
        cm.sum(axis=0, keepdims=True) > 0, cm / cm.sum(axis=0, keepdims=True), 0
    )
    cl = Logger.current_logger()
    cl.report_confusion_matrix(
        f"{title} confusion matrix - {prefix}",
        series=prefix,
        iteration=iteration,
        matrix=cm,
        xaxis="predictions",
        yaxis="ground truth",
        xlabels=labels,
        ylabels=labels,
    )
    cl.report_confusion_matrix(
        f"{title} confusion matrix - {prefix} normalized by targets",
        series=prefix,
        iteration=iteration,
        matrix=cm_normalized_by_targets,
        xaxis="predictions",
        yaxis="ground truth",
        xlabels=labels,
        ylabels=labels,
    )
    cl.report_confusion_matrix(
        f"{title} confusion matrix - {prefix} normalized by predictions",
        series=prefix,
        iteration=iteration,
        matrix=cm_normalized_by_predictions,
        xaxis="predictions",
        yaxis="ground truth",
        xlabels=labels,
        ylabels=labels,
    )


def get_token_labels(token_values: list[float]):
    """
    Get labels for tokens that have numerical values and their indices in vocab.

    This is necessary to produce a confusion matrix that is correctly labeled even if
    the tokens are not in numerical order and even if the vocab includes non-numerical
    tokens (aka special tokens.)
    """
    indices, labels = zip(
        *[
            i
            for i in sorted(enumerate(token_values), key=lambda x: x[1])
            if not np.isnan(i[1])
        ]
    )
    labels = [f"token {int(i) if i.is_integer() else i}" for i in labels]
    return indices, labels


def calculate_95_ci(data, n, ci_method="bootstrap_quantiles"):
    """
    Generates 95% CI for evaluation metrics.
    Three types are available thorugh the ci_method argument.
    - bootstrap_quantiles: takes the 2.5% and 97.5% quantiles of the bootstrap sample.
    - bootstrap_t_interval: based on the bootstrap sample distribution around the sample's mean.
        If len(data)=1 returns None values for the CI's bounds.
    - binomial: CI based on the binomial distribution for proportion metrics. Does not require
        repeated samplings from the data.
    - wilson: The Wilson score interval, it is assymetric, doesn't overshoot the [0,1] range and
        does not result with a zero-width length intervals.
        Does not require repeated samplings from the data.

    Args:
    ----
        data (_type_): a list or scalar of evaluation metrics, e.g. accuracy rate.
        n (int): Can either be number of bootstraps (if sent from a bootstrap run)
                 or number of observations in the test set (if task.num_bootstrap_runs is None or 0).
        ci_method (str, optional): _description_. Defaults to "bootstrap_quantiles".

    Raises:
    ------
        ValueError: if binomial CI was chosen but input values extending [0,1]

    Returns:
    -------
        _type_: mean value, lower and upper CI bounds.
    """
    if isinstance(data, int):
        data = [data]
    mean = np.mean(data)
    if ci_method == "bootstrap_quantiles":
        lower_bound = np.percentile(data, 2.5)
        upper_bound = np.percentile(data, 97.5)
    elif ci_method == "bootstrap_t_interval":
        std_error = np.std(data)
        ci = stats.t.interval(
            0.95,
            n - 1,
            loc=mean,
            scale=std_error,
        )
        lower_bound = ci[0]
        upper_bound = ci[1]
    elif ci_method == "binomial":
        if np.max(data) > 1 or np.min(data) < 0:
            logging.warning("Binomial based CI's are meant to be used for proportions")
            return mean, np.nan, np.nan
        ci_length = 1.96 * np.sqrt((mean * (1 - mean)) / n)
        lower_bound = mean - ci_length
        upper_bound = mean + ci_length
    elif ci_method == "wilson":
        z = 1.96
        denominator = 1 + ((z**2) / n)
        center = (mean + ((z**2) / (2 * n))) / denominator
        margin = (
            z * np.sqrt((mean * (1 - mean) / n) + (z**2)) / (2 * n)
        ) / denominator
        lower_bound = max(0, center - margin)
        upper_bound = min(1, center + margin)
    return mean, lower_bound, upper_bound
