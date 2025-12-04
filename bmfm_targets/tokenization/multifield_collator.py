"""Multifield data collator for language modeling."""

import logging
from typing import Literal

import torch
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from bmfm_targets.config import FieldInfo, LabelColumnInfo
from bmfm_targets.datasets.label_ontology import LabelOntology

from .multifield_instance import MultiFieldInstance
from .multifield_tokenizer import MultiFieldTokenizer

logger = logging.getLogger(__name__)


class MultiFieldCollator:
    """
    Class to multi-field data collator.

    Attributes
    ----------
        tokenizer (MultiFieldTokenizer): Tokenizer
        fields (list[FieldInfo]): List of FieldInfo
        label_columns (list[LabelColumnInfo]): List of LabelColumnInfo
        label_dict (dict[dict[str, int]] | None): Label dictionary
        pad_to_multiple_of (int): Pad inputs to multiple of
        max_length (int | None, optional): Maximum length of input sequence. Defaults to None.
        return_attention_mask (bool, optional): Whether to return attention mask. Defaults to True.
        return_special_tokens_mask (bool, optional): Whether to return special tokens mask. Defaults to True.
        padding (PaddingStrategy, optional): Padding strategy. Defaults to PaddingStrategy.LONGEST. Available options are PaddingStrategy.LONGEST, True, PaddingStrategy.DO_NOT_PAD.
        truncation (TruncationStrategy, optional): Truncation strategy. Defaults to TruncationStrategy.ONLY_FIRST. Available options are TruncationStrategy.ONLY_FIRST, TruncationStrategy.ONLY_SECOND, True, TruncationStrategy.LONGEST_SECOND, TruncationStrategy.DO_NOT_TRUNCATE.

    """

    def __init__(
        self,
        tokenizer: MultiFieldTokenizer,
        fields: list[FieldInfo],
        label_columns: list[LabelColumnInfo] | None = None,
        collation_strategy: Literal[
            "sequence_labeling",
            "sequence_classification",
            "language_modeling",
            "multitask",
        ] = "language_modeling",
        label_dict: dict[str, dict[str, int]] | None = None,
        max_length: int | None = None,
        pad_to_multiple_of: int | None = None,
        return_attention_mask: bool = True,
        return_special_tokens_mask: bool = True,
        padding: PaddingStrategy | bool = True,
        pad_zero_expression_strategy: dict | None = None,
        truncation: TruncationStrategy | bool = True,
        masker: object | None = None,
        sequence_order: str | None = None,
        sequence_dropout_factor: float | int | None = None,
        log_normalize_transform: bool = False,
        median_normalization: bool = False,
        rda_transform: Literal["downsample", "poisson_downsample", "equal"]
        | int
        | None = None,
        map_orthologs: str | None = None,
        tokenize_kwargs: dict | None = None,
        renoise: float | None = 0.6,
        limit_input_tokens: list | None = None,
        label_ontology: dict[str, LabelOntology] | None = None,
    ):
        """
        Multifield Data collator.

        Args:
        ----
            tokenizer (MultiFieldTokenizer): Tokenizer
            fields (list[FieldInfo]): List of FieldInfo
            label_dict (dict[dict[str, int]] | None, optional): Label dictionary. Defaults to None.
            pad_to_multiple_of (int): Pad inputs to multiple of
            max_length (int | None, optional): Maximum length of input sequence. Defaults to None.
            return_attention_mask (bool, optional): Whether to return attention mask. Defaults to True.
            return_special_tokens_mask (bool, optional): Whether to return special tokens mask. Defaults to True.
            padding (PaddingStrategy, optional): Padding strategy. Defaults to PaddingStrategy.LONGEST. Available options are PaddingStrategy.LONGEST, PaddingStrategy.MAX_LENGTH, PaddingStrategy.DO_NOT_PAD.
            truncation (TruncationStrategy, optional): Truncation strategy. Defaults to TruncationStrategy.ONLY_FIRST. Available options are TruncationStrategy.ONLY_FIRST, TruncationStrategy.ONLY_SECOND, True, TruncationStrategy.LONGEST_SECOND, TruncationStrategy.DO_NOT_TRUNCATE.

        Raises:
        ------
            ValueError: If tokenizer does not have a mask token
            ValueError: If input fields are not specified


        """
        self.tokenizer = tokenizer
        self.fields = fields
        self.label_columns = label_columns
        self.padding = padding
        self.max_length = max_length
        self.truncation = truncation
        self.return_attention_mask = return_attention_mask
        self.return_special_tokens_mask = return_special_tokens_mask
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_zero_expression_strategy = pad_zero_expression_strategy
        self.masker = masker
        self.input_fields = [fi for fi in self.fields if fi.is_input]
        self.input_field_names = [fi.field_name for fi in self.input_fields]
        self.label_field_names = [
            fi.field_name for fi in self.fields if not fi.is_input
        ]
        self.sequence_order = sequence_order
        self.sequence_dropout_factor = sequence_dropout_factor
        self.rda_transform = rda_transform
        self.log_normalize_transform = log_normalize_transform
        self.median_normalization = median_normalization
        self.collation_strategy = collation_strategy
        self.label_dict = label_dict
        self.map_orthologs = map_orthologs
        self.tokenize_kwargs = tokenize_kwargs if tokenize_kwargs is not None else {}
        self.renoise = renoise
        self.limit_input_tokens = limit_input_tokens
        self.label_ontology = label_ontology
        self.sample_metadata_keys = ["cell_name", "seq_id", "perturbed_genes"]
        self.__post__init__()

    def __post__init__(self):
        if self.input_field_names is None or len(self.input_field_names) == 0:
            raise ValueError("You need to specify the input fields")

        if self.rda_transform and getattr(self.masker, "switch_ratio", 0) > 0:
            raise ValueError("switch ratio is not allowed with rda down sampling")

        if self.rda_transform in {"downsample", "poisson_downsample", "equal"}:
            assert "label_expressions" in self.label_field_names

    def _mark_rda_special_tokens(self, batch, fields: list[FieldInfo]):
        for field in fields:
            if field.is_input:
                batch[field.field_name]["special_tokens_mask"][:, 1:3] = 1

    def _encode_continuous_value_special_tokens(self, batch, fields: list[FieldInfo]):
        for field in fields:
            if (
                field.is_input
                and field.tokenization_strategy == "continuous_value_encoder"
            ):
                batch[field.field_name]["input_ids"] = torch.where(
                    batch[field.field_name]["special_tokens_mask"].bool(),
                    -(batch[field.field_name]["input_ids"] + 1),
                    batch[field.field_name]["input_ids"],
                )

    def read_label_ids(
        self, examples: list[MultiFieldInstance]
    ) -> dict[str, torch.Tensor]:
        """
        Load labels for MultiFieldInstances.

        Handles regression, classification, and multilabel settings.

        Uses -100 for missing, silent, or unknown labels.
        """
        labels = {}
        if not examples:
            return labels

        for label_info in self.label_columns:
            col = label_info.label_column_name
            label_dict = self.label_dict.get(col, {})
            if not label_dict:
                logger.warning(f"Label dict missing for label_column_name {col}")
            id_list = [
                self._label_id_from_example(
                    example, label_info, label_dict, self.label_ontology
                )
                for example in examples
            ]
            labels[col] = torch.tensor(id_list)

        return labels

    @staticmethod
    def _label_id_from_example(
        example: MultiFieldInstance,
        label_info: LabelColumnInfo,
        single_label_dict: dict[str, int],
        label_ontology: dict[str, LabelOntology] | None = None,
    ) -> int | float | list[float]:
        """
        Return label ID (or one-hot list) for a single example and label column.

        - Handles regression, single-label, and multilabel classification.
        - Returns -100 for missing, silent, or unknown labels.
        """
        col = label_info.label_column_name
        val = example.metadata.get(col, "nan")
        val_str = str(val)

        # Merge standard and custom silent values
        silent_values = {"nan"}
        if label_info.silent_label_values:
            silent_values.update(label_info.silent_label_values)

        if val_str in silent_values:
            return -100

        if label_info.is_regression_label:
            return val

        if label_info.label_ontology:
            ontology = label_ontology[label_info.label_column_name]
            ont_leaves = ontology.find_leaves(val_str)
            # allocate space for all probs + flag to handle no label case
            # in loss
            probs = torch.zeros(len(single_label_dict) + 1)
            indices = [single_label_dict[i] for i in ont_leaves]
            probs[indices] = 1.0
            probs[-1] = bool(len(ont_leaves))
            return probs.tolist()

        if label_info.is_multilabel:
            indices = {
                single_label_dict[label]
                for label in val_str.split(label_info.multilabel_str_sep)
                if label in single_label_dict
            }
            return [1.0 if i in indices else 0.0 for i in range(len(single_label_dict))]

        if val_str in single_label_dict:
            return single_label_dict[val_str]

        logger.warning(f"Unknown label '{val_str}' for column '{col}', using -100.")
        return -100

    @property
    def selective_dropout_weights(self):
        return getattr(self.masker, "selective_dropout_weights", None)

    def __call__(
        self,
        examples: (
            list[MultiFieldInstance]
            | tuple[list[MultiFieldInstance], list[MultiFieldInstance]]
        ),
    ) -> dict[str, torch.Tensor]:
        """
        Creates a batch of inputs and labels for sequence labelling.

        Args:
        ----
            examples (list[MultiFieldInstance]): List of MultiFieldInstance

        Returns:
        -------
            dict[str, torch.Tensor]: Dictionary of tensors for input_ids, labels for each field if masking fields else input_ids. If return_attention_mask is True, attention_mask is also returned. If return_special_tokens_mask is True, special_tokens_mask is also returned.

        """
        examples_pair = None
        pre_transformed_examples = examples
        # to avoid circular import
        from bmfm_targets.training.sample_transforms import transform_inputs

        if isinstance(examples[0], MultiFieldInstance):
            examples = transform_inputs(examples, **self._transform_inputs_kwargs)
        else:
            examples, examples_pair = list(map(list, zip(*examples)))
            examples = transform_inputs(examples, **self._transform_inputs_kwargs)
            examples_pair = transform_inputs(
                examples_pair, **self._transform_inputs_kwargs
            )

        batch = self.tokenize_batch(examples, examples_pair)
        batch["mfi"] = pre_transformed_examples

        if examples[0].metadata is not None:
            for key in self.sample_metadata_keys:
                if key in examples[0].metadata:
                    batch[key] = [mfi.metadata.get(key) for mfi in examples]

        # TODO: Here we assume that the first multi-field instance of the pair is the one that contains the labels
        if (
            self.collation_strategy in ["multitask", "sequence_classification"]
            and self.label_dict is not None
            and self.label_columns is not None
            and len(self.label_columns) > 0
        ):
            batch["label_ids"] = self.read_label_ids(examples)

        self._encode_continuous_value_special_tokens(batch, self.fields)
        if self.rda_transform:
            self._mark_rda_special_tokens(batch, self.fields)

        return_dict = self._prepare_return_dict(batch)
        return return_dict

    @property
    def _transform_inputs_kwargs(self):
        return {
            "fields": self.fields,
            "label_columns": self.label_columns,
            "sequence_order": self.sequence_order,
            "log_normalize_transform": self.log_normalize_transform,
            "median_normalization": self.median_normalization,
            "rda_transform": self.rda_transform,
            "pad_zero_expression_strategy": self.pad_zero_expression_strategy,
            "max_length": self.max_length,
            "sequence_dropout_factor": self.sequence_dropout_factor,
            "map_orthologs": self.map_orthologs,
            "renoise": self.renoise,
            "limit_input_tokens": self.limit_input_tokens,
            "selective_dropout_weights": self.selective_dropout_weights,
        }

    def tokenize_batch(self, examples, examples_pair=None):
        tokenize_kwargs = {
            "return_tensors": "pt",
            "truncation": self.truncation,
            "max_length": self.max_length,
            "padding": self.padding,
            "return_attention_mask": self.return_attention_mask,
            "return_special_tokens_mask": self.return_special_tokens_mask,
            "pad_to_multiple_of": self.pad_to_multiple_of,
        }
        tokenize_kwargs.update(self.tokenize_kwargs)
        batch = self.tokenizer(
            mfi=examples, mfi_pair=examples_pair, fields=self.fields, **tokenize_kwargs
        )
        return batch

    def _prepare_return_dict(self, batch):
        """Prepare the final return dictionary with input_ids, labels, and metadata."""
        attention_mask = batch[self.input_field_names[0]]["attention_mask"]
        input_ids = torch.stack(
            [batch[field]["input_ids"] for field in self.input_field_names], dim=1
        )

        return_dict = {"input_ids": input_ids, "attention_mask": attention_mask}

        # MLM masking / WCED label extraction for language_modeling, multitask
        # and WCED sequence_labeling

        if self.masker is not None:
            input_ids, labels, attention_mask = self.masker.mask_inputs(
                self.fields, batch
            )
            return_dict.update(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": attention_mask,
                }
            )
        # Sequence labeling labels
        elif self.collation_strategy == "sequence_labeling":
            special_tokens_mask = batch[self.input_field_names[0]][
                "special_tokens_mask"
            ].bool()
            labels = {
                field: batch[field]["input_ids"] for field in self.label_field_names
            }
            for field_labels in labels.values():
                field_labels[special_tokens_mask] = -100
            return_dict["labels"] = labels

        # Classification labels for sequence_classification and multitask
        if (
            self.collation_strategy in ["sequence_classification", "multitask"]
            and "label_ids" in batch
        ):
            if "labels" in return_dict:
                return_dict["labels"].update(batch["label_ids"])
            else:
                return_dict["labels"] = batch["label_ids"]

        # Add metadata
        return_dict.update(
            {k: batch[k] for k in self.sample_metadata_keys if k in batch}
        )

        return return_dict
