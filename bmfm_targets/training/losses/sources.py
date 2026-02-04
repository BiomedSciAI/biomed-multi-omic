"""
Data source implementations for loss handling.

This module provides concrete implementations of the DataSource abstract base class,
which handle extraction of logits and labels from model outputs and batches.
"""

from __future__ import annotations

import torch

from bmfm_targets.config.model_config import FieldInfo, LabelColumnInfo
from bmfm_targets.tokenization.multifield_tokenizer import MultiFieldTokenizer

from .base import DataSource


class FieldSource(DataSource):
    """
    DataSource for field-based losses (e.g., gene expression, DNA sequences).

    Handles extraction of logits and labels from model fields.
    For WCED (Whole Cell Expression Decoder) use WCEDFieldSource instead.

    Note: FieldSource focuses on data extraction. Metric grouping (loss_group)
    is handled at the LossTask level.
    """

    def __init__(
        self,
        field_name: str,
        decoder_key: str | None = None,
    ):
        """
        Initialize FieldSource.

        Args:
        ----
            field_name: Name of the field (e.g., 'gene_expression')
            decoder_key: Optional explicit decoder key for logits.
                If not provided, it will be derived by LossTask as
                "{field_name}{objective.decoder_suffix}"

        """
        self.field_name = field_name
        self._decoder_key = decoder_key  # User-provided override
        self.field: FieldInfo | None = None
        self.tokenizer: MultiFieldTokenizer | None = None

    @property
    def decoder_key(self) -> str | None:
        """
        Get decoder key for extracting logits.

        Returns the explicit decoder_key if provided, otherwise None.
        LossTask will set this during bind() if not explicitly provided.
        """
        return self._decoder_key

    @decoder_key.setter
    def decoder_key(self, value: str | None):
        """Allow setting decoder_key (used by LossTask.bind())."""
        self._decoder_key = value

    def resolve_schema(
        self,
        fields: list[FieldInfo],
        labels: list[LabelColumnInfo],
        tokenizer: MultiFieldTokenizer | None = None,
        decoder_suffix: str | None = None,
        objective_name: str | None = None,
    ) -> None:
        """
        Find matching FieldInfo and set decoder_key if not explicit.

        Args:
        ----
            fields: List of FieldInfo from model config
            labels: List of LabelColumnInfo (unused for FieldSource)
            tokenizer: Optional tokenizer
            decoder_suffix: Suffix for decoder_key derivation (e.g., '_regression')
            objective_name: Name of objective (unused for regular FieldSource)

        """
        self.tokenizer = tokenizer

        # Find matching field
        self.field = next((f for f in fields if f.field_name == self.field_name), None)
        if not self.field:
            raise ValueError(f"Field '{self.field_name}' not found")

        # Set decoder_key if not explicitly provided
        if self._decoder_key is None and decoder_suffix is not None:
            self._decoder_key = f"{self.field_name}{decoder_suffix}"

    def extract_logits(self, logits: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract logits from model outputs.

        Uses the decoder_key which should be set by LossTask.bind().
        """
        if self.decoder_key is None:
            raise ValueError(
                f"decoder_key not set for field '{self.field_name}'. "
                f"This should be set by LossTask.bind()"
            )
        return logits[self.decoder_key]

    def extract_labels(self, labels: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract labels for this field."""
        return labels[self.field.field_name]

    def extract(self, outputs, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract logits and labels."""
        return self.extract_logits(outputs.logits), self.extract_labels(batch["labels"])

    def get_output_size(self) -> int:
        """Return vocab size for this field."""
        if self.field is None:
            raise ValueError("Field not resolved. Call resolve_schema() first.")
        return self.field.vocab_size

    def concat_batch_tensors(
        self, batch: dict, outputs, predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        Concatenate batch tensors for tracking.

        Based on FieldLossTask.concat_batch_tensors() lines 409-439.
        """
        from bmfm_targets.training.metrics.batch_prediction_metrics import (
            concat_field_loss_batch_tensors,
        )

        # Regular field loss
        logits_to_record = {
            k: v
            for k, v in outputs.logits.items()
            if k.startswith(self.field.field_name) and (v.shape[-1] == 1)
        }
        return concat_field_loss_batch_tensors(
            input_ids=batch["input_ids"],
            predictions=predictions,
            labels=batch["labels"][self.field.field_name],
            **logits_to_record,
        )

    @property
    def name(self) -> str:
        """
        Return the base name of this data source (field name only).

        Note: This is the schema-level name. Metric grouping (via loss_group)
        is handled at the LossTask level.
        """
        return self.field_name


class WCEDFieldSource(FieldSource):
    """
    Specialized DataSource for WCED (Whole Cell Expression Decoder) losses.

    WCED is a special decoding mode that predicts expression values for all genes
    in a cell, not just the masked ones. This source handles the specific indexing
    and label extraction required for WCED.

    Key differences from FieldSource:
    - decoder_key is always "{field_name}_wced"
    - Extracts specific token index (decode_token_index, always 0)
    - May extract specific output index (decoder_output_index) for multi-output WCED
    - Uses label subsets (label_set) for different prediction targets
    """

    def __init__(self, field_name: str, wced_target: str):
        super().__init__(field_name, decoder_key=None)
        self.wced_target = wced_target
        self.decode_token_index = 0
        self.decoder_output_index: int | None = None
        self.label_set: str | None = None

    @property
    def decoder_key(self) -> str:
        """WCED always uses {field_name}_wced as decoder key."""
        return f"{self.field_name}_wced"

    @decoder_key.setter
    def decoder_key(self, value: str | None):
        """WCED decoder_key is fixed and cannot be changed."""
        if value is not None and value != f"{self.field_name}_wced":
            raise ValueError(
                f"WCEDFieldSource decoder_key is fixed as '{self.field_name}_wced', "
                f"cannot set to '{value}'"
            )

    def resolve_schema(
        self,
        fields: list[FieldInfo],
        labels: list[LabelColumnInfo],
        tokenizer: MultiFieldTokenizer | None = None,
        decoder_suffix: str | None = None,
        objective_name: str | None = None,
    ) -> None:
        """
        Resolve schema for WCED source.

        Sets up label_set and decoder_output_index based on field config
        and objective name.
        """
        # Call parent but don't use decoder_suffix (WCED has fixed decoder_key)
        super().resolve_schema(
            fields, labels, tokenizer, decoder_suffix=None, objective_name=None
        )

        vocab_field = self.field.decode_modes["wced"]["vocab_field"]
        self.label_set = self.wced_target.rstrip(f"_{vocab_field}")

        # Set output index based on objective name
        if objective_name is not None:
            from .task import lookup_wced_output_index

            self.decoder_output_index = lookup_wced_output_index(
                objective_name, self.field
            )

    def extract_logits(self, logits: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract WCED logits with proper indexing.

        WCED logits have shape: [batch, num_tokens, num_outputs, vocab_size]
        We need to:
        1. Extract the WCED decoder output
        2. Select the specific output index if multiple outputs exist
        3. Select the decode_token_index (always 0)
        """
        result = logits[self.decoder_key]

        # Handle multi-output WCED (e.g., separate outputs for mse and is_zero)
        if self.decoder_output_index is not None:
            result = result[..., self.decoder_output_index]

        # Extract the specific token (always index 0 for WCED)
        result = result[:, self.decode_token_index, :]

        return result

    def extract_labels(self, labels: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract WCED labels with label_set indexing.

        WCED labels are organized by label_set (e.g., 'input_genes', 'non_input_genes').
        """
        result = labels[self.field.field_name]
        if self.label_set is not None:
            result = result[self.label_set]
        return result

    def concat_batch_tensors(
        self, batch: dict, outputs, predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        Concatenate batch tensors for WCED tracking.

        Based on FieldLossTask.concat_batch_tensors() lines 409-439.
        """
        from bmfm_targets.training.metrics.batch_prediction_metrics import (
            concat_wced_field_loss_batch_tensors,
        )

        # WCED field loss - extract logits at decode_token_index
        logits_to_record = {
            k: v[:, self.decode_token_index, :]
            for k, v in outputs.logits.items()
            if k == self.decoder_key
        }
        return concat_wced_field_loss_batch_tensors(
            predictions=predictions,
            labels=batch["labels"][self.field.field_name][self.label_set],
            **logits_to_record,
        )

    @property
    def name(self) -> str:
        """
        Return the base name of this data source (field name only).

        Note: This is the schema-level name. The wced_target is used for
        loss_group at the LossTask level, not in the source name.
        """
        return self.field_name

    @property
    def loss_group(self) -> str | None:
        """WCED sources use wced_target as their natural loss grouping."""
        return self.wced_target


class LabelSource(DataSource):
    """
    DataSource for label-based losses (e.g., cell type classification).

    Handles extraction of logits and labels from label columns for
    sequence classification tasks.
    """

    def __init__(self, label_name: str):
        """
        Initialize LabelSource.

        Args:
        ----
            label_name: Name of the label column (e.g., 'cell_type')

        """
        self.label_name = label_name
        self.label_column: LabelColumnInfo | None = None

    def resolve_schema(
        self,
        fields: list[FieldInfo],
        labels: list[LabelColumnInfo],
        tokenizer: MultiFieldTokenizer | None = None,
        decoder_suffix: str | None = None,
        objective_name: str | None = None,
    ) -> None:
        """Find matching LabelColumnInfo."""
        # decoder_suffix and objective_name not used for label sources
        self.label_column = next(
            (lc for lc in labels if lc.label_column_name == self.label_name),
            None,
        )
        if not self.label_column:
            raise ValueError(f"Label column '{self.label_name}' not found")

    def extract_logits(self, logits: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract logits for label column."""
        return logits[self.label_name]

    def extract_labels(self, labels: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract labels for label column."""
        return labels[self.label_name]

    def extract(self, outputs, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract logits and labels (refactored to use new methods)."""
        return self.extract_logits(outputs.logits), self.extract_labels(batch["labels"])

    def get_output_size(self) -> int:
        """Return output size for this label column."""
        if self.label_column is None:
            raise ValueError("Label column not resolved. Call resolve_schema() first.")
        return self.label_column.output_size

    def concat_batch_tensors(
        self, batch: dict, outputs, predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        Concatenate batch tensors for label loss.

        Based on LabelLossTask.concat_batch_tensors() lines 555-569.
        """
        from bmfm_targets.training.metrics.batch_prediction_metrics import (
            concat_label_loss_batch_tensors,
        )

        logits_to_record = {
            k: v for k, v in outputs.logits.items() if k == self.label_name
        }
        return concat_label_loss_batch_tensors(
            input_ids=batch["input_ids"],
            predictions=predictions,
            labels=batch["labels"][self.label_name],
            **logits_to_record,
        )

    @property
    def name(self) -> str:
        """Return the name of this data source."""
        return self.label_name


# Made with Bob
