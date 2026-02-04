"""
Abstract base classes for composition-based loss handling.

This module defines the core abstractions:
- DataSource: Handles data extraction and schema resolution
- Objective: Handles loss computation and predictions
"""

from abc import ABC, abstractmethod

import torch

from bmfm_targets.config import FieldInfo, LabelColumnInfo
from bmfm_targets.tokenization import MultiFieldTokenizer


class DataSource(ABC):
    """
    Abstract base class for data sources.

    A DataSource is responsible for:
    - Resolving schema information (finding FieldInfo or LabelColumnInfo)
    - Extracting logits and labels from model outputs and batch
    - Providing output size information
    - Concatenating batch tensors for prediction tracking
    """

    @abstractmethod
    def resolve_schema(
        self,
        fields: list[FieldInfo],
        labels: list[LabelColumnInfo],
        tokenizer: MultiFieldTokenizer | None = None,
        decoder_suffix: str | None = None,
        objective_name: str | None = None,
    ) -> None:
        """
        Resolve and store schema information (output_size, etc.).

        Args:
        ----
            fields: List of FieldInfo objects from model config
            labels: List of LabelColumnInfo objects from model config
            tokenizer: Optional tokenizer for accessing token values
            decoder_suffix: Suffix to append to field name for decoder_key derivation
                           (e.g., '_token_scores', '_regression'). Only used by FieldSource.
            objective_name: Name of the objective (e.g., 'mse', 'cross_entropy').
                           Used by WCEDFieldSource to determine output index.

        """
        pass

    @abstractmethod
    def extract_logits(self, logits: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract logits from model outputs without requiring labels.

        Args:
        ----
            logits: Dictionary of logits from model outputs

        Returns:
        -------
            torch.Tensor: Extracted logits for this data source

        """
        pass

    @abstractmethod
    def extract_labels(self, labels: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract labels from batch labels dictionary.

        Args:
        ----
            labels: Dictionary of labels from batch

        Returns:
        -------
            torch.Tensor: Extracted labels for this data source

        """
        pass

    @abstractmethod
    def extract(self, outputs, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract (logits, labels) from outputs and batch.

        Args:
        ----
            outputs: Model outputs object with .logits attribute
            batch: Batch dictionary with 'labels' key

        Returns:
        -------
            tuple: (logits, labels) as tensors

        """
        pass

    @abstractmethod
    def get_output_size(self) -> int:
        """
        Return the output size for this data source.

        Returns
        -------
            int: Output size (vocab size for fields, n_classes for labels)

        """
        pass

    @abstractmethod
    def concat_batch_tensors(
        self, batch: dict, outputs, predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        Concatenate batch tensors for tracking predictions.

        Args:
        ----
            batch: Batch dictionary
            outputs: Model outputs
            predictions: Prediction tensor

        Returns:
        -------
            torch.Tensor: Concatenated batch tensor for tracking

        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the name of this data source.

        Returns
        -------
            str: Name used for logging and metric keys

        """
        pass

    @property
    def loss_group(self) -> str | None:
        """
        Return natural loss grouping for this source, or None.

        Override in subclasses that have inherent grouping (e.g., WCEDFieldSource).
        Used by LossTask to auto-derive loss_group if not explicitly set.

        Returns
        -------
            str | None: Natural grouping identifier or None

        """
        return None


class Objective(ABC):
    """
    Abstract base class for loss objectives.

    An Objective is responsible for:
    - Binding to a specific output size
    - Computing loss from logits and labels
    - Converting logits to predictions
    - Preparing inputs for metric calculation
    - Providing default metrics

    Attributes
    ----------
        output_size: The output size from the data source (set by bind())
    """

    def __init__(self):
        """Initialize objective with unbound output_size."""
        self.output_size: int | None = None

    @abstractmethod
    def bind(
        self, output_size: int, tokenizer: MultiFieldTokenizer | None = None
    ) -> None:
        """
        Bind to a specific output size and optionally tokenizer.

        Args:
        ----
            output_size: The output size from the data source
            tokenizer: Optional tokenizer for objectives that need it (e.g., TokenValueObjective)

        """
        pass

    @abstractmethod
    def compute(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute loss value.

        Args:
        ----
            logits: Model logits
            labels: Ground truth labels

        Returns:
        -------
            torch.Tensor: Loss value (scalar)

        """
        pass

    @abstractmethod
    def get_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to predictions.

        Args:
        ----
            logits: Model logits

        Returns:
        -------
            torch.Tensor: Predictions

        """
        pass

    def prepare_metric_inputs(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare inputs for metric calculation.

        Default implementation handles both classification and regression cases:
        - If output_size is set (classification): reshape to (batch, classes)
        - If output_size is None (regression/binary): keep same shape

        Args:
        ----
            logits: Model logits (already extracted by DataSource)
            labels: Ground truth labels (already extracted by DataSource)

        Returns:
        -------
            tuple: (model_outputs, gt_labels) formatted for metrics

        """
        output_size = getattr(self, "output_size", None)
        if output_size is not None and output_size > 1:
            # Multiclass classification: reshape to (batch, classes)
            return (logits.view(-1, output_size), labels.to(torch.int64).view(-1))
        else:
            # Regression or binary: predictions and labels same shape
            return (logits, labels.to(logits.dtype))

    @abstractmethod
    def default_metrics(self) -> list[dict]:
        """
        Return default metric configurations.

        Returns
        -------
            list[dict]: List of metric configuration dicts

        """
        pass

    @property
    @abstractmethod
    def decoder_suffix(self) -> str:
        """
        Return decoder suffix for this objective type.

        Examples: '_token_scores', '_regression', '_is_zero'

        Returns
        -------
            str: Decoder suffix to append to source name

        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this objective."""


# Made with Bob
