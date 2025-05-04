"""Multifield data collator for language modeling."""

import logging
from typing import Literal

import torch
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from bmfm_targets.config import FieldInfo, LabelColumnInfo
from bmfm_targets.tokenization.resources import get_ortholog_genes
from bmfm_targets.training import sample_transforms
from bmfm_targets.training.masking import Masker

from .multifield_instance import MultiFieldInstance
from .multifield_tokenizer import MultiFieldTokenizer

logger = logging.getLogger(__name__)
from functools import partial, reduce


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
            "multilabel",
        ] = "language_modeling",
        label_dict: dict[str, dict[str, int]] | None = None,
        max_length: int | None = None,
        pad_to_multiple_of: int | None = None,
        return_attention_mask: bool = True,
        return_special_tokens_mask: bool = True,
        padding: PaddingStrategy | bool = True,
        pad_zero_expression_strategy: dict | None = None,
        truncation: TruncationStrategy | bool = True,
        mlm: bool = True,
        masker: Masker | None = None,
        sequence_order: str | None = None,
        sequence_dropout_factor: float | int | None = None,
        log_normalize_transform: bool = False,
        rda_transform: Literal["downsample"] | Literal["equal"] | int | None = None,
        multilabel_str_sep: str = "|",
        map_orthologs: str | None = None,
    ):
        """
        Multifield Data collator.

        Args:
        ----
            tokenizer (MultiFieldTokenizer): Tokenizer
            fields (list[FieldInfo]): List of FieldInfo
            label_dict (dict[dict[str, int]] | None, optional): Label dictionary. Defaults to None.
            mlm (bool): Whether to use MLM
            pad_to_multiple_of (int): Pad inputs to multiple of
            max_length (int | None, optional): Maximum length of input sequence. Defaults to None.
            return_attention_mask (bool, optional): Whether to return attention mask. Defaults to True.
            return_special_tokens_mask (bool, optional): Whether to return special tokens mask. Defaults to True.
            padding (PaddingStrategy, optional): Padding strategy. Defaults to PaddingStrategy.LONGEST. Available options are PaddingStrategy.LONGEST, PaddingStrategy.MAX_LENGTH, PaddingStrategy.DO_NOT_PAD.
            truncation (TruncationStrategy, optional): Truncation strategy. Defaults to TruncationStrategy.ONLY_FIRST. Available options are TruncationStrategy.ONLY_FIRST, TruncationStrategy.ONLY_SECOND, True, TruncationStrategy.LONGEST_SECOND, TruncationStrategy.DO_NOT_TRUNCATE.
            multilabel_str_sep (str="|"): The string separating lables of a multilabel class label.

        Raises:
        ------
            ValueError: If tokenizer does not have a mask token
            ValueError: If input fields are not specified
            ValueError: If mlm and mask fields are not specified
            ValueError: If mlm and mask fields are not subset of input fields
            ValueError: If mlm and change ratio is not between 0 and 1
            ValueError: If mlm and mask ratio is not between 0 and 1
            ValueError: If mlm and switch ratio is not between 0 and 1
            ValueError: If mlm and mask ratio + switch ratio is greater than 1

        """
        self.tokenizer = tokenizer
        self.mlm = mlm
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
        self.mask_field_names = [
            fi.field_name for fi in self.fields if fi.is_masked and fi.is_input
        ]
        self.label_field_names = [
            fi.field_name for fi in self.fields if not fi.is_input
        ]
        self.sequence_order = sequence_order
        self.sequence_dropout_factor = sequence_dropout_factor
        self.rda_transform = rda_transform
        self.log_normalize_transform = log_normalize_transform
        self.collation_strategy = collation_strategy
        self.label_dict = label_dict
        self.multilabel_str_sep = multilabel_str_sep
        self.map_orthologs = map_orthologs
        self.__post__init__()

    def __post__init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        if self.mlm and self.masker is None:
            raise ValueError("MLM requested but no Masker object provided")

        if self.input_field_names is None or len(self.input_field_names) == 0:
            raise ValueError("You need to specify the input fields")

        if (
            self.mlm
            and (self.mask_field_names is None or len(self.mask_field_names)) == 0
        ):
            raise ValueError("You need to specify the mask fields")

        if self.mlm and not set(self.mask_field_names).issubset(
            set(self.input_field_names)
        ):
            raise ValueError("Mask fields should be subset of input fields")
        if self.mlm and self.rda_transform and self.masker.switch_ratio > 0:
            raise ValueError("switch ratio is not allowed with rda down sampling")

        if self.rda_transform in {"downsample", "equal"}:
            assert "label_expressions" in self.label_field_names

    def _compose_transforms(
        self,
        sequence_order=None,
        log_normalize_transform=False,
        rda_transform=None,
        pad_zero_expression_strategy=None,
        max_length=None,
        sequence_dropout_factor=None,
        fields_to_downcast=None,
        map_orthologs=None,
    ):
        transforms = []
        if sequence_order == "random":
            transforms.append(sample_transforms.randomize)
        elif sequence_order == "sorted":
            transforms.append(
                partial(sample_transforms.sort_by_field, field="expressions")
            )
        if sequence_dropout_factor is not None:
            if sequence_dropout_factor > 1:
                transforms.append(
                    partial(
                        sample_transforms.dropout_chunk_in_range,
                        chunk_size=sequence_dropout_factor,
                        drop_range=(0, max_length - 1),
                    )
                )
            else:
                transforms.append(
                    partial(
                        sample_transforms.dropout_random,
                        dropout_ratio=sequence_dropout_factor,
                    )
                )

        if pad_zero_expression_strategy is not None:
            transforms.append(
                partial(
                    sample_transforms.pad_zero_expressed_genes,
                    pad_zero_expression_strategy=pad_zero_expression_strategy,
                    max_length=max_length,
                )
            )

        if log_normalize_transform:
            transforms.append(
                partial(sample_transforms.log_normalize, max_length=max_length)
            )

        if rda_transform == "downsample":
            transforms.append(
                partial(
                    sample_transforms.rda_downsample,
                    max_length=max_length,
                    downsample_threshold=1000,
                    normalized_sum=10000,
                )
            )
        elif rda_transform == "equal":
            transforms.append(
                partial(
                    sample_transforms.rda_downsample,
                    max_length=max_length,
                    downsample_threshold=torch.inf,
                    normalized_sum=10000,
                )
            )
        elif isinstance(rda_transform, int) and not isinstance(rda_transform, bool):
            transforms.append(
                partial(
                    sample_transforms.rda_align,
                    target_read_resolution=rda_transform,
                    max_length=max_length,
                    normalized_sum=10000,
                )
            )
        if fields_to_downcast:
            transforms.append(
                partial(
                    sample_transforms.downcast_numeric_fields,
                    fields_to_downcast=fields_to_downcast,
                )
            )
        if map_orthologs == "mouse_to_human_orthologs":
            mapping = get_ortholog_genes(
                return_mapping=True,
                from_species="mmusculus",
                to_species="hsapiens",
                id_type="gene_name",
            )
            transforms.append(
                partial(
                    sample_transforms.field_remap,
                    field_to_remap="genes",
                    mapping=mapping,
                )
            )
        elif map_orthologs == "human_to_mouse_orthologs":
            mapping = get_ortholog_genes(
                return_mapping=True,
                from_species="hsapiens",
                to_species="mmusculus",
                id_type="gene_name",
            )
            transforms.append(
                partial(
                    sample_transforms.field_remap,
                    field_to_remap="genes",
                    mapping=mapping,
                )
            )

        return transforms

    def _transform_inputs(self, examples):
        fields_to_downcast = [
            f.field_name
            for f in self.fields
            if "expressions" in f.field_name
            and f.tokenization_strategy != "continuous_value_encoder"
            and f.is_input
        ]
        transforms = self._compose_transforms(
            sequence_order=self.sequence_order,
            log_normalize_transform=self.log_normalize_transform,
            rda_transform=self.rda_transform,
            pad_zero_expression_strategy=self.pad_zero_expression_strategy,
            max_length=self.max_length,
            sequence_dropout_factor=self.sequence_dropout_factor,
            fields_to_downcast=fields_to_downcast,
            map_orthologs=self.map_orthologs,
        )
        if "perturbations" in [i.field_name for i in self.fields]:
            transforms.append(
                partial(sample_transforms.sort_by_field, field="perturbations")
            )
        if len(transforms) > 0:
            expressed_genes_in_batch = set()
            if (
                self.pad_zero_expression_strategy is not None
                and self.pad_zero_expression_strategy["strategy"] == "batch_wise"
            ):
                expressed_genes_in_batch = {
                    gene
                    for mfi in examples
                    for gene, expression in zip(
                        mfi.data["genes"], mfi.data["expressions"]
                    )
                    if expression != 0.0
                }
            combined_func = reduce(
                lambda f, g: lambda x, *a, **k: g(f(x, *a, **k), *a, **k),
                transforms,
            )
            return [
                combined_func(x, expressed_genes_in_batch=expressed_genes_in_batch)
                for x in examples
            ]
        return examples

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

    def _read_label_ids(
        self, examples: list[MultiFieldInstance]
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        labels = {}

        def _lookup_id(mfi, label, single_label_dict):
            # only one label means regression and labels should not be converted
            if len(single_label_dict) <= 1:
                return mfi.metadata[label]
            label_val = str(mfi.metadata[label])
            if self.collation_strategy == "multilabel":
                # label_val is pipe delimeted...
                multilabels = []
                for label in label_val.split(self.multilabel_str_sep):
                    if label in single_label_dict.keys():
                        multilabels.append(single_label_dict[label])
                multilabels = set(multilabels)
                # Now convert the multilabels to one-hot-coding....
                one_hot_coding = [
                    1.0 if i in multilabels else 0.0
                    for i in range(len(single_label_dict))
                ]
                return one_hot_coding
            else:
                # nan is mapped to standard `ignore_index` of -100
                if label_val == "nan":
                    return -100
                return single_label_dict[label_val]

        for label in self.label_columns:
            label_column_name = label.label_column_name
            single_label_dict = self.label_dict[label_column_name]
            if examples[0].metadata and label_column_name in examples[0].metadata:
                id_list = [
                    _lookup_id(mfi, label_column_name, single_label_dict)
                    for mfi in examples
                ]
                labels[label_column_name] = torch.tensor(id_list)

        return labels

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
            dict[str, torch.Tensor]: Dictionary of tensors for input_ids, labels for each field if mlm is True else input_ids. If return_attention_mask is True, attention_mask is also returned. If return_special_tokens_mask is True, special_tokens_mask is also returned.

        """
        examples_pair = None
        if isinstance(examples[0], MultiFieldInstance):
            examples = self._transform_inputs(examples)
        else:
            examples, examples_pair = list(map(list, zip(*examples)))
            examples = self._transform_inputs(examples)
            examples_pair = self._transform_inputs(examples_pair)

        batch = self.tokenize_batch(examples, examples_pair)
        # TODO: This is not generic, this should be addressed
        if examples[0].metadata is not None and "cell_name" in examples[0].metadata:
            batch["cell_names"] = [mfi.metadata.get("cell_name") for mfi in examples]
        # TODO: Here we assume that the first multi-field instance of the pair is the one that contains the labels
        if (
            self.collation_strategy
            in ["multitask", "sequence_classification", "multilabel"]
            and self.label_dict is not None
            and self.label_columns is not None
            and len(self.label_columns) > 0
        ):
            batch["label_ids"] = self._read_label_ids(examples)

        self._encode_continuous_value_special_tokens(batch, self.fields)
        if self.rda_transform:
            self._mark_rda_special_tokens(batch, self.fields)

        return_dict = self._prepare_return_dict(batch)
        return return_dict

    def tokenize_batch(self, examples, examples_pair=None):
        batch = self.tokenizer(
            mfi=examples,
            mfi_pair=examples_pair,
            fields=self.fields,
            return_tensors="pt",
            truncation=self.truncation,
            max_length=self.max_length,
            padding=self.padding,
            return_attention_mask=self.return_attention_mask,
            return_special_tokens_mask=self.return_special_tokens_mask,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        return batch

    def _prepare_return_dict(self, batch):
        attention_mask: torch.Tensor = batch[self.input_field_names[0]][
            "attention_mask"
        ]
        special_tokens_mask = batch[self.input_field_names[0]][
            "special_tokens_mask"
        ].bool()

        field_input_ids = [
            batch[field]["input_ids"] for field in self.input_field_names
        ]
        input_ids = torch.stack(field_input_ids, dim=1)
        return_dict = {}

        if self.collation_strategy == "multitask":
            labels = {}
            if self.mlm and self.mask_field_names is not None:
                input_ids, labels, attention_mask = self.masker.mask_inputs(
                    self.fields, batch
                )
            if "label_ids" in batch:
                for key in batch["label_ids"]:
                    labels[key] = batch["label_ids"][key]
                return_dict = {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": attention_mask,
                }

            if len(labels) == 0:
                return_dict = {"input_ids": input_ids, "attention_mask": attention_mask}

        if self.collation_strategy == "language_modeling":
            if self.mlm and self.mask_field_names is not None:
                input_ids, labels, attention_mask = self.masker.mask_inputs(
                    self.fields, batch
                )

                return_dict = {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": attention_mask,
                }
            else:
                return_dict = {"input_ids": input_ids, "attention_mask": attention_mask}

        elif self.collation_strategy in ["sequence_classification", "multilabel"]:
            return_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

            if "label_ids" in batch:
                return_dict["labels"] = batch["label_ids"]

        elif self.collation_strategy == "sequence_labeling":
            field_label_ids = {
                field: batch[field]["input_ids"] for field in self.label_field_names
            }
            for field in field_label_ids:
                field_label_ids[field][special_tokens_mask] = -100
            return_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": field_label_ids,
            }

        if "cell_names" in batch:
            return_dict["cell_names"] = batch["cell_names"]
        return return_dict
