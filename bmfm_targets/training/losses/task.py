"""
LossTask container for composition-based loss handling.

This module provides the LossTask class that composes a DataSource and an Objective
to create a complete loss computation unit.

Name Hierarchy:
    Schema Level:     source.name           (field_name or label_name)
    Task Level:       loss_group            (optional: mvc, wced_target, custom)
    Objective Level:  objective.name        (mse, cross_entropy, etc.)

Derived Names:
    metric_key:       "{source.name}_{loss_group}" if loss_group else source.name
    name:             "{source.name}_{loss_group}_{objective.name}" or "{source.name}_{objective.name}"
    loss_display_name: "{name}_loss"
"""

from __future__ import annotations

import torch
from torchmetrics import MetricCollection

from bmfm_targets.config.model_config import FieldInfo, LabelColumnInfo
from bmfm_targets.tokenization.multifield_tokenizer import MultiFieldTokenizer
from bmfm_targets.training import metrics

from .base import DataSource, Objective
from .objectives import TokenValueObjective
from .sources import FieldSource, LabelSource, WCEDFieldSource


class LossTask:
    """
    Container that composes a DataSource and an Objective.

    This replaces the old FieldLossTask and LabelLossTask with a composition-based
    approach that separates data extraction (DataSource) from loss computation (Objective).

    The loss_group parameter enables metric grouping for:
    - WCED (auto-derived from wced_target during bind)
    - MVC decoders (explicit in config)
    - Custom groupings
    """

    def __init__(
        self,
        source: DataSource,
        objective: Objective,
        weight: float = 1.0,
        metrics: list[dict] | None = None,
        loss_group: str | None = None,
    ):
        """
        Initialize LossTask with composition.

        Args:
        ----
            source: DataSource for extracting logits and labels
            objective: Objective for computing loss and predictions
            weight: Weight for this loss in multi-loss scenarios (default: 1.0)
            metrics: Optional list of metric configs (overrides objective defaults)
            loss_group: Optional grouping for metrics. Used to distinguish:
                - Different decoders on same field (e.g., "mvc" vs default)
                - WCED label subsets (auto-derived from wced_target if not set)
                - Custom groupings for metric tracking

        """
        self.source = source
        self.objective = objective
        self.weight = weight
        self._metrics = metrics
        self._loss_group = loss_group

    def bind(
        self,
        fields: list[FieldInfo],
        label_columns: list[LabelColumnInfo],
        tokenizer: MultiFieldTokenizer | None = None,
    ):
        """
        Bind to schema and initialize objective.

        This method:
        1. Resolves schema in the source (finds FieldInfo/LabelColumnInfo, sets decoder_key)
        2. Binds the objective to the output size
        3. Auto-derives loss_group from source if not explicitly set
        4. Handles TokenValueObjective special case

        Args:
        ----
            fields: List of FieldInfo from model config
            label_columns: List of LabelColumnInfo from model config
            tokenizer: Optional tokenizer (required for TokenValueObjective)

        """
        # Source resolves schema, sets decoder_key and any source-specific setup
        self.source.resolve_schema(
            fields,
            label_columns,
            tokenizer,
            decoder_suffix=self.objective.decoder_suffix,
            objective_name=self.objective.name,
        )

        # Objective binds to output size
        self.objective.bind(self.source.get_output_size(), tokenizer)

        # Auto-derive loss_group from source if not set
        if self._loss_group is None:
            self._loss_group = self.source.loss_group

        # TokenValueObjective needs token values from tokenizer
        if isinstance(self.objective, TokenValueObjective):
            if tokenizer is None:
                raise ValueError("TokenValueObjective requires tokenizer")
            if not isinstance(self.source, FieldSource):
                raise ValueError("TokenValueObjective requires a FieldSource")
            token_values = tokenizer.get_token_values(self.source.field_name)
            if token_values is None:
                raise ValueError(
                    f"Token values not found for field '{self.source.field_name}'"
                )
            self.objective.set_token_values(token_values)

    @property
    def loss_group(self) -> str | None:
        """
        Return the loss group for metric grouping.

        May be:
        - Explicitly set via config
        - Auto-derived from wced_target during bind()
        - None for simple cases (no grouping needed)
        """
        return self._loss_group

    def calculate_loss(self, outputs, batch) -> torch.Tensor:
        """
        Calculate loss (weight is applied in calculate_losses utility).

        Args:
        ----
            outputs: Model outputs with .logits attribute
            batch: Batch dictionary with 'labels' key

        Returns:
        -------
            torch.Tensor: Loss value (unweighted)

        """
        logits, labels = self.source.extract(outputs, batch)
        return self.objective.compute(logits, labels)

    def get_predictions(self, logits: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get predictions from logits dict.

        Used by calculate_predictions() utility.

        Args:
        ----
            logits: Dictionary of logits from model outputs

        Returns:
        -------
            torch.Tensor: Predictions

        """
        extracted_logits = self.source.extract_logits(logits)
        return self.objective.get_predictions(extracted_logits)

    def extract_metric_inputs(
        self, logits: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]
    ) -> tuple[str, torch.Tensor, torch.Tensor]:
        """
        Extract and format inputs for metric calculation.

        Uses source extraction methods and delegates formatting to objective.

        Args:
        ----
            logits: Dictionary of model logits
            labels: Dictionary of ground truth labels

        Returns:
        -------
            tuple: (metric_key, model_outputs, gt_labels)

        """
        these_logits = self.source.extract_logits(logits)
        these_labels = self.source.extract_labels(labels)

        # Let objective prepare the inputs (handles all formatting)
        model_outputs, gt_labels = self.objective.prepare_metric_inputs(
            these_logits, these_labels
        )

        return self.metric_key, model_outputs, gt_labels

    def get_metrics(self) -> MetricCollection:
        """
        Get metric collection for this task.

        Uses objective's default metrics unless overridden via metrics parameter.
        Filters metrics based on output size (classification vs regression).

        Returns
        -------
            MetricCollection: Collection of metrics for this task

        """
        metric_configs = (
            self._metrics
            if self._metrics is not None
            else self.objective.default_metrics()
        )

        # Determine num_classes for metric creation
        # Use objective.output_size if set (classification), otherwise use 1 (regression/binary)
        output_size = getattr(self.objective, "output_size", None)
        if output_size is not None and output_size > 1:
            num_classes = output_size
        else:
            num_classes = 1

        return MetricCollection(
            {
                mt["name"]: metrics.get_metric_object(mt, num_classes)
                for mt in metric_configs
            }
        )

    def concat_batch_tensors(self, batch, outputs, predictions):
        """
        Concatenate batch tensors for tracking.

        Delegates to source.

        Args:
        ----
            batch: Batch dictionary
            outputs: Model outputs
            predictions: Predictions dictionary (keyed by metric_key)

        Returns:
        -------
            torch.Tensor: Concatenated batch tensor

        """
        return self.source.concat_batch_tensors(
            batch, outputs, predictions[self.metric_key]
        )

    @property
    def name(self) -> str:
        """
        Task name for logging.

        Format: "{source.name}_{loss_group}_{objective.name}" or "{source.name}_{objective.name}"
        """
        if self._loss_group:
            return f"{self.source.name}_{self._loss_group}_{self.objective.name}"
        return f"{self.source.name}_{self.objective.name}"

    @property
    def loss_display_name(self) -> str:
        """
        Loss name for logging.

        Format: "{name}_loss"
        """
        return f"{self.name}_loss"

    @property
    def metric_key(self) -> str:
        """
        Key for metric grouping.

        Format: "{source.name}_{loss_group}" or just "{source.name}"

        Multiple loss tasks with the same metric_key will have their predictions
        combined for metric calculation.
        """
        if self._loss_group:
            return f"{self.source.name}_{self._loss_group}"
        return self.source.name

    @property
    def decoder_key(self) -> str:
        """
        Decoder key for model outputs.

        Uses source's decoder_key (which should be set after bind()).
        For FieldSource, this is either:
        - Explicit from config
        - Auto-derived for WCED
        - Derived by bind() using objective.decoder_suffix
        """
        if hasattr(self.source, "decoder_key") and self.source.decoder_key:
            return self.source.decoder_key
        # Fallback (shouldn't happen if bind() was called)
        return f"{self.source.name}{self.objective.decoder_suffix}"

    @property
    def logit_key(self) -> str:
        """
        Alias for decoder_key (for backward compatibility).

        Some code uses logit_key instead of decoder_key.
        """
        return self.decoder_key

    # Convenience properties for backward compatibility with BaseTrainingModule
    @property
    def field(self) -> FieldInfo | None:
        """Return the field (for FieldSource) or None."""
        if isinstance(self.source, FieldSource):
            return self.source.field
        return None

    @property
    def output_size(self) -> int:
        """Return output size from source (for backward compatibility with MultiTaskClassifier)."""
        return self.source.get_output_size()

    @property
    def label_column(self) -> LabelColumnInfo | None:
        """Return the label column (for LabelSource) or None."""
        if isinstance(self.source, LabelSource):
            return self.source.label_column
        return None

    @classmethod
    def create(
        cls,
        source: DataSource,
        objective: Objective,
        weight: float = 1.0,
        metrics: list[dict] | None = None,
        loss_group: str | None = None,
    ) -> LossTask:
        """
        Factory method to create a LossTask.

        This is the most explicit way to create a LossTask when you already have
        source and objective instances.

        Args:
        ----
            source: DataSource instance
            objective: Objective instance
            weight: Loss weight (default: 1.0)
            metrics: Optional metric configs
            loss_group: Optional metric grouping

        Returns:
        -------
            LossTask: New LossTask instance

        Example:
        -------
            >>> task = LossTask.create(
            ...     source=FieldSource("cell_type"),
            ...     objective=CrossEntropyObjective(),
            ...     weight=1.0
            ... )

        """
        return cls(source, objective, weight, metrics, loss_group)

    @classmethod
    def from_field(
        cls,
        field_name: str,
        objective: Objective,
        weight: float = 1.0,
        metrics: list[dict] | None = None,
        loss_group: str | None = None,
        decoder_key: str | None = None,
        wced_target: str | None = None,
    ) -> LossTask:
        """
        Factory method to create a LossTask from a field name.

        Convenience method that creates a FieldSource or WCEDFieldSource
        automatically.

        Args:
        ----
            field_name: Name of the field to use
            objective: Objective instance
            weight: Loss weight (default: 1.0)
            metrics: Optional metric configs
            loss_group: Optional metric grouping
            decoder_key: Optional explicit decoder key (ignored if wced_target set)
            wced_target: Optional WCED target name (creates WCEDFieldSource)

        Returns:
        -------
            LossTask: New LossTask instance

        Example:
        -------
            >>> # Regular field loss
            >>> task = LossTask.from_field(
            ...     field_name="expressions",
            ...     objective=MSEObjective(),
            ...     loss_group="mvc",
            ...     decoder_key="expressions_mvc_regression"
            ... )
            >>> # WCED loss
            >>> task = LossTask.from_field(
            ...     field_name="expressions",
            ...     objective=MSEObjective(),
            ...     wced_target="input_genes"
            ... )

        """
        if wced_target is not None:
            source = WCEDFieldSource(field_name, wced_target)
        else:
            source = FieldSource(field_name, decoder_key=decoder_key)

        return cls(
            source,
            objective,
            weight,
            metrics,
            loss_group,
        )

    @classmethod
    def from_label(
        cls,
        label_name: str,
        objective: Objective,
        weight: float = 1.0,
        metrics: list[dict] | None = None,
        loss_group: str | None = None,
    ) -> LossTask:
        """
        Factory method to create a LossTask from a label name.

        Convenience method that creates a LabelSource automatically.

        Args:
        ----
            label_name: Name of the label to use
            objective: Objective instance
            weight: Loss weight (default: 1.0)
            metrics: Optional metric configs
            loss_group: Optional metric grouping (rarely needed for labels)

        Returns:
        -------
            LossTask: New LossTask instance

        Example:
        -------
            >>> task = LossTask.from_label(
            ...     label_name="disease_status",
            ...     objective=CrossEntropyObjective(),
            ...     weight=1.0
            ... )

        """
        return cls(LabelSource(label_name), objective, weight, metrics, loss_group)


def lookup_wced_output_index(loss_name: str, field: FieldInfo) -> int | None:
    """
    Look up the output index for a WCED (Whole Cell Expression Decoder) loss.

    The WCED decoder can output multiple logit tensors (e.g., one for MSE, one for
    is_zero classification). This function finds which dimension/index corresponds
    to the given loss type.

    Args:
    ----
        loss_name: Name of the loss type (e.g., 'mse', 'is_zero_bce')
        field: FieldInfo object containing decode_modes configuration

    Returns:
    -------
        int | None: Index of the loss in the WCED logit_outputs list, or None
            if no WCED mode, only one output, or loss_name not in outputs

    """
    if "wced" not in field.decode_modes:
        return None
    logit_outputs = field.decode_modes["wced"].get("logit_outputs", [])
    if len(logit_outputs) <= 1:
        return None
    if loss_name not in logit_outputs:
        return None
    return logit_outputs.index(loss_name)


# Made with Bob
