"""
Objective implementations for loss handling.

This module provides concrete implementations of the Objective abstract base class,
which handle loss computation, predictions, and metric preparation for different
types of learning objectives (classification, regression, etc.).
"""

from __future__ import annotations

import torch

from bmfm_targets.tokenization.multifield_tokenizer import MultiFieldTokenizer
from bmfm_targets.training import metrics

from .base import Objective


class CrossEntropyObjective(Objective):
    """
    Cross-entropy loss objective for classification tasks.

    Based on FieldLossTask.calculate_loss() lines 370-375.
    """

    def __init__(self, label_smoothing: float = 0.0, ignore_index: int = -100):
        """
        Initialize CrossEntropyObjective.

        Args:
        ----
            label_smoothing: Label smoothing factor (default: 0.0)
            ignore_index: Index to ignore in loss computation (default: -100)

        """
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.output_size: int | None = None

    def bind(
        self, output_size: int, tokenizer: MultiFieldTokenizer | None = None
    ) -> None:
        """Bind to output size."""
        self.output_size = output_size

    def compute(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss."""
        if self.output_size is None:
            raise ValueError("Objective not bound. Call bind() first.")
        return metrics.ce_loss(
            logits.reshape(-1, self.output_size),
            labels.reshape(-1).long(),
            label_smoothing=self.label_smoothing,
        )

    def get_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """Get predictions via argmax."""
        return torch.argmax(logits, dim=-1)

    def default_metrics(self) -> list[dict]:
        """Return default metrics for classification."""
        return [{"name": "accuracy"}]

    @property
    def decoder_suffix(self) -> str:
        """Return decoder suffix for classification objectives."""
        return "_token_scores"

    @property
    def name(self) -> str:
        """Return objective name."""
        return "cross_entropy"


class FocalObjective(Objective):
    """
    Focal loss objective for handling class imbalance.

    Based on FieldLossTask.calculate_loss() lines 377-382.
    """

    def __init__(self, focal_gamma: float = 2.0, ignore_index: int = -100):
        """
        Initialize FocalObjective.

        Args:
        ----
            focal_gamma: Focusing parameter (default: 2.0)
            ignore_index: Index to ignore in loss computation (default: -100)

        """
        self.focal_gamma = focal_gamma
        self.ignore_index = ignore_index
        self.output_size: int | None = None

    def bind(
        self, output_size: int, tokenizer: MultiFieldTokenizer | None = None
    ) -> None:
        """Bind to output size."""
        self.output_size = output_size

    def compute(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute focal loss."""
        if self.output_size is None:
            raise ValueError("Objective not bound. Call bind() first.")
        return metrics.focal_loss(
            logits.reshape(-1, self.output_size),
            labels.reshape(-1).long(),
            self.focal_gamma,
        )

    def get_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """Get predictions via argmax."""
        return torch.argmax(logits, dim=-1)

    def default_metrics(self) -> list[dict]:
        """Return default metrics for classification."""
        return [{"name": "accuracy"}]

    @property
    def decoder_suffix(self) -> str:
        """Return decoder suffix for classification objectives."""
        return "_token_scores"

    @property
    def name(self) -> str:
        """Return objective name."""
        return "focal"


class MSEObjective(Objective):
    """
    Mean squared error objective for regression tasks.

    Based on FieldLossTask.calculate_loss() lines 348-357.
    """

    def __init__(
        self,
        ignore_zero: bool = False,
        link_function: str | None = None,
        shrinkage: float = 0.0,
    ):
        """
        Initialize MSEObjective.

        Args:
        ----
            ignore_zero: Whether to ignore zero values (default: False)
            link_function: Optional link function ('exp' or None)
            shrinkage: Shrinkage parameter (default: 0.0)

        """
        self.ignore_zero = ignore_zero
        self.link_function = link_function
        self.shrinkage = shrinkage

    def bind(
        self, output_size: int, tokenizer: MultiFieldTokenizer | None = None
    ) -> None:
        """MSE doesn't need output_size."""
        pass

    def compute(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss."""
        if self.link_function == "exp":
            logits = torch.exp(logits)

        return metrics.mse_loss(
            logits.squeeze(),
            labels,
            ignore_zero=self.ignore_zero,
            shrinkage=self.shrinkage,
        )

    def get_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Get predictions (apply link function if specified).

        For regression with shape [batch, 1], flatten to [batch] to match old behavior.
        """
        if logits.ndim == 2 and logits.shape[1] == 1:
            preds = logits.view(-1)
        else:
            preds = logits
        if self.link_function == "exp":
            preds = torch.exp(preds)
        return preds

    def prepare_metric_inputs(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare inputs for regression metrics.

        Uses get_predictions() to properly handle shape and link functions.
        """
        predictions = self.get_predictions(logits)
        return (predictions, labels.to(predictions.dtype))

    def default_metrics(self) -> list[dict]:
        """Return default metrics for regression."""
        # Regression losses typically don't have separate metrics to avoid conflicts
        # when multiple regression losses target the same field
        return []

    @property
    def output_size(self) -> int:
        """Return 1 for regression (single output per prediction)."""
        return 1

    @property
    def decoder_suffix(self) -> str:
        """Return decoder suffix for regression objectives."""
        return "_regression"

    @property
    def name(self) -> str:
        """Return objective name."""
        return "mse"


class MAEObjective(Objective):
    """
    Mean absolute error objective for regression tasks.

    Based on FieldLossTask.calculate_loss() lines 359-368.
    """

    def __init__(
        self,
        ignore_zero: bool = False,
        link_function: str | None = None,
        shrinkage: float = 0.0,
    ):
        """
        Initialize MAEObjective.

        Args:
        ----
            ignore_zero: Whether to ignore zero values (default: False)
            link_function: Optional link function ('exp' or None)
            shrinkage: Shrinkage parameter (default: 0.0)

        """
        self.ignore_zero = ignore_zero
        self.link_function = link_function
        self.shrinkage = shrinkage

    def bind(
        self, output_size: int, tokenizer: MultiFieldTokenizer | None = None
    ) -> None:
        """MAE doesn't need output_size."""
        pass

    def compute(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute MAE loss."""
        if self.link_function == "exp":
            logits = torch.exp(logits)

        return metrics.mae_loss(
            logits.squeeze(),
            labels,
            ignore_zero=self.ignore_zero,
            shrinkage=self.shrinkage,
        )

    def get_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Get predictions (apply link function if specified).

        For regression with shape [batch, 1], flatten to [batch] to match old behavior.
        """
        if logits.ndim == 2 and logits.shape[1] == 1:
            preds = logits.view(-1)
        else:
            preds = logits
        if self.link_function == "exp":
            preds = torch.exp(preds)
        return preds

    def default_metrics(self) -> list[dict]:
        """Return default metrics for regression."""
        # Regression losses typically don't have separate metrics to avoid conflicts
        # when multiple regression losses target the same field
        return []

    @property
    def output_size(self) -> int:
        """Return 1 for regression (single output per prediction)."""
        return 1

    @property
    def decoder_suffix(self) -> str:
        """Return decoder suffix for regression objectives."""
        return "_regression"

    @property
    def name(self) -> str:
        """Return objective name."""
        return "mae"


class TokenValueObjective(Objective):
    """
    Token value loss objective that uses token values from tokenizer.

    CRITICAL: Requires tokenizer during bind() to get token values.
    Based on FieldLossTask lines 384-393.
    """

    def __init__(self):
        """Initialize TokenValueObjective."""
        self.output_size: int | None = None
        self.token_values: list[float] | None = None

    def bind(
        self, output_size: int, tokenizer: MultiFieldTokenizer | None = None
    ) -> None:
        """
        Bind to output size and tokenizer.

        CRITICAL: TokenValueObjective requires tokenizer.
        Token values will be set by LossTask.bind() after this call.
        """
        self.output_size = output_size
        if tokenizer is None:
            raise ValueError("TokenValueObjective requires tokenizer during bind()")

    def set_token_values(self, token_values: list[float]):
        """
        Set token values.

        Called by LossTask after getting values from tokenizer.
        """
        self.token_values = token_values

    def compute(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute token value loss."""
        if self.token_values is None:
            raise ValueError("Token values not set. Call set_token_values() first.")
        if self.output_size is None:
            raise ValueError("Objective not bound. Call bind() first.")
        return metrics.token_value_loss(
            logits.reshape(-1, self.output_size),
            labels.reshape(-1),
            self.token_values,
        )

    def get_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """Get predictions via argmax."""
        return torch.argmax(logits, dim=-1)

    def default_metrics(self) -> list[dict]:
        """Return default metrics."""
        # TokenValue losses typically don't have separate metrics to avoid conflicts
        # when multiple regression losses target the same field
        return []

    @property
    def decoder_suffix(self) -> str:
        """Return decoder suffix for token value objectives."""
        return "_token_scores"

    @property
    def name(self) -> str:
        """Return objective name."""
        return "token_value"


class IsZeroBCEObjective(Objective):
    """
    Binary cross-entropy objective for zero detection.

    Based on FieldLossTask.calculate_loss() lines 395-399.
    """

    def __init__(self):
        """Initialize IsZeroBCEObjective."""
        pass

    def bind(
        self, output_size: int, tokenizer: MultiFieldTokenizer | None = None
    ) -> None:
        """IsZeroBCE doesn't need output_size."""
        pass

    def compute(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute is-zero BCE loss."""
        return metrics.is_zero_bce_loss(
            logits.reshape(-1),
            labels.reshape(-1),
        )

    def get_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """Get predictions (1 if zero, 0 if non-zero)."""
        return torch.where(logits > 0, 1, 0)

    def prepare_metric_inputs(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare inputs for binary metrics - ensure 1D tensors."""
        predictions = self.get_predictions(logits).view(-1)
        return (predictions, labels.to(predictions.dtype).view(-1))

    def default_metrics(self) -> list[dict]:
        """Return default metrics for binary is_zero classification."""
        return [{"name": "nonzero_confusion_matrix"}]

    @property
    def decoder_suffix(self) -> str:
        """Return decoder suffix for is_zero objectives."""
        return "_is_zero"

    @property
    def name(self) -> str:
        """Return objective name."""
        return "is_zero_bce"


class IsZeroFocalObjective(Objective):
    """
    Focal loss objective for zero detection.

    Based on FieldLossTask.calculate_loss() lines 400-405.
    """

    def __init__(self, focal_gamma: float = 2.0):
        """
        Initialize IsZeroFocalObjective.

        Args:
        ----
            focal_gamma: Focusing parameter (default: 2.0)

        """
        self.focal_gamma = focal_gamma

    def bind(
        self, output_size: int, tokenizer: MultiFieldTokenizer | None = None
    ) -> None:
        """IsZeroFocal doesn't need output_size."""
        pass

    def compute(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute is-zero focal loss."""
        return metrics.focal_loss(
            logits.reshape(-1),
            torch.where(labels == -100, labels, (labels == 0)).reshape(-1).long(),
            focal_gamma=self.focal_gamma,
        )

    def get_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """Get predictions (1 if zero, 0 if non-zero)."""
        return torch.where(logits > 0, 1, 0)

    def prepare_metric_inputs(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare inputs for binary metrics - ensure 1D tensors."""
        predictions = self.get_predictions(logits).view(-1)
        return (predictions, labels.to(predictions.dtype).view(-1))

    def default_metrics(self) -> list[dict]:
        """Return default metrics for binary is_zero classification."""
        return [{"name": "nonzero_confusion_matrix"}]

    @property
    def decoder_suffix(self) -> str:
        """Return decoder suffix for is_zero objectives."""
        return "_is_zero"

    @property
    def name(self) -> str:
        """Return objective name."""
        return "is_zero_focal"


class BCEWithLogitsObjective(Objective):
    """Binary cross-entropy with logits for multi-label classification."""

    def __init__(self, ignore_index: int = -100):
        """
        Initialize BCEWithLogitsObjective.

        Args:
        ----
            ignore_index: Index to ignore in loss computation (default: -100)

        """
        self.ignore_index = ignore_index
        self.output_size: int | None = None

    def bind(
        self, output_size: int, tokenizer: MultiFieldTokenizer | None = None
    ) -> None:
        """Bind to output size."""
        self.output_size = output_size

    def compute(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute BCE with logits loss."""
        if self.output_size is None:
            raise ValueError("Objective not bound. Call bind() first.")
        return metrics.classification_loss(
            logits,
            labels,
            "BCEWithLogitsLoss",
            self.output_size,
            ignore_zero=False,
            label_smoothing=0.0,
            class_weight=None,
            focal_gamma=0.0,
        )

    def get_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """Get predictions via sigmoid."""
        return torch.sigmoid(logits)

    def prepare_metric_inputs(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Multi-label: apply sigmoid, preserve class structure."""
        if self.output_size is None:
            raise ValueError("Objective not bound. Call bind() first.")
        return (
            torch.sigmoid(logits.view(-1, self.output_size)),
            labels.to(torch.int64),
        )

    def default_metrics(self) -> list[dict]:
        """Return default metrics."""
        return [{"name": "accuracy"}]

    @property
    def decoder_suffix(self) -> str:
        """Return decoder suffix for multi-label classification."""
        return "_token_scores"

    @property
    def name(self) -> str:
        """Return objective name."""
        return "bce_with_logits"


class HCEObjective(Objective):
    """
    Hierarchical Cross Entropy objective.

    CRITICAL: Requires special metric input preparation.
    Based on LabelLossTask.extract_metric_inputs() lines 490-499.
    """

    def __init__(self, ignore_index: int = -100):
        """
        Initialize HCEObjective.

        Args:
        ----
            ignore_index: Index to ignore in loss computation (default: -100)

        """
        self.ignore_index = ignore_index
        self.output_size: int | None = None

    def bind(
        self, output_size: int, tokenizer: MultiFieldTokenizer | None = None
    ) -> None:
        """Bind to output size."""
        self.output_size = output_size

    def compute(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute HCE loss."""
        if self.output_size is None:
            raise ValueError("Objective not bound. Call bind() first.")
        return metrics.classification_loss(
            logits,
            labels,
            "hce",
            self.output_size,
            ignore_zero=False,
            label_smoothing=0.0,
            class_weight=None,
            focal_gamma=0.0,
        )

    def get_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """Get predictions via argmax."""
        return torch.argmax(logits, dim=1)

    def prepare_metric_inputs(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        CRITICAL: Override to use special HCE adaptation.

        Based on LabelLossTask.extract_metric_inputs() lines 493-499.
        """
        # Import here to avoid circular dependency
        from .utils import adapt_hce_prediction_and_labels_to_metrics_entries

        (
            adapted_logits,
            adapted_labels,
        ) = adapt_hce_prediction_and_labels_to_metrics_entries(logits, labels)
        return adapted_logits, adapted_labels

    def default_metrics(self) -> list[dict]:
        """Return default metrics."""
        return [{"name": "accuracy"}]

    @property
    def decoder_suffix(self) -> str:
        """Return decoder suffix for HCE objectives."""
        return "_token_scores"

    @property
    def name(self) -> str:
        """Return objective name."""
        return "hce"


# Made with Bob
