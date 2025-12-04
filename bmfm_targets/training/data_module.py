import logging
from collections.abc import Mapping
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
import pytorch_lightning as pl
import torch
from scanpy import AnnData, read_h5ad
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from bmfm_targets.config import FieldInfo, LabelColumnInfo
from bmfm_targets.datasets import DatasetTransformer, PerturbationDatasetTransformer
from bmfm_targets.datasets.base_dna_dataset import BaseDNASeqDataset
from bmfm_targets.datasets.base_perturbation_dataset import BasePerturbationDataset
from bmfm_targets.datasets.base_rna_dataset import BaseRNAExpressionDataset
from bmfm_targets.datasets.label_ontology import LabelOntology
from bmfm_targets.tokenization import MultiFieldCollator, MultiFieldTokenizer
from bmfm_targets.tokenization.resources import (
    get_L1000_genes,
    get_ortholog_genes,
    get_protein_coding_genes,
)
from bmfm_targets.training.masking import Masker, MaskingStrategy
from bmfm_targets.training.masking.strategy import WCEDMasker

logger = logging.getLogger(__name__)


class DataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for scRNA datasets."""

    DATASET_FACTORY: type[BaseRNAExpressionDataset] = BaseRNAExpressionDataset
    DATASET_TRANSFORMER_FACTORY: type[DatasetTransformer] = DatasetTransformer

    class LabelOntologyTuple(NamedTuple):
        """Label ontology artifacts."""

        label_dict: dict[str, int]
        label_ontology: dict[str, LabelOntology]

    def __init__(
        self,
        tokenizer: MultiFieldTokenizer,
        fields: list[FieldInfo],
        label_columns: list[LabelColumnInfo] | None = None,
        data_dir: str | Path | None = None,
        processed_name: str = "processed",
        dataset_kwargs: dict[str, Any] | None = None,
        transform_kwargs: dict[str, Any] | None = None,
        transform_datasets: bool = True,
        limit_dataset_samples: int | Mapping[str, int] | None = None,
        limit_genes: Literal["tokenizer"]
        | Literal["protein_coding"]
        | None = "tokenizer",
        balancing_label_column: str | None = None,
        max_length: int | Literal["scheduled"] = 512,
        padding: PaddingStrategy | str | bool = "max_length",
        truncation: TruncationStrategy | bool = True,
        pad_to_multiple_of: int = 16,
        sequence_order: str | None = None,
        sequence_dropout_factor: int | float | None = None,
        pad_zero_expression_strategy: str | dict | None = None,
        collation_strategy: Literal[
            "language_modeling",
            "sequence_classification",
            "sequence_labeling",
            "multitask",
        ] = "language_modeling",
        mlm: bool = False,
        change_ratio: float = 0.15,
        mask_ratio: float = 1.0,
        switch_ratio: float = 0.0,
        masking_strategy: MaskingStrategy | None = None,
        prevent_attention_to_masked: bool = False,
        comask_across_fields: bool = False,
        log_normalize_transform: bool = False,
        median_normalization: bool = False,
        rda_transform: Literal["downsample"]
        | Literal["auto_align"]
        | Literal["equal"]
        | int
        | None = None,
        map_orthologs: str | None = None,
        batch_size: int | Literal["scheduled"] = 32,
        num_workers: int = 0,
        shuffle: bool = False,
        tokenize_kwargs: dict | None = None,
        bootstrap_test_dataloader: bool = False,
        **kwargs,  # to enable stashing hydra config settings
    ):
        """
        Construct a PyTorch Lightning DataModule for single-cell RNA expression datasets.

        This module handles loading, processing, and providing batched data from scRNA-seq datasets
        for training and evaluation of deep learning models.

        Parameters
        ----------
        tokenizer : MultiFieldTokenizer
            Tokenizer used to convert gene expression data into tokens for model input.
        fields : list[FieldInfo]
            List of field information objects defining the structure of the input data.
        label_columns : list[LabelColumnInfo] | None, optional
            Columns to use as labels/targets, by default None.
        data_dir : str | Path | None, optional
            Directory containing raw and processed data files. If provided, raw data is expected at
            `data_dir/h5ad/dataset_name.h5ad` and processed data at `data_dir/{processed_name}.h5ad`.
            If not supplied, dataset_kwargs and transform_kwargs must contain full paths for h5ad files.
        processed_name : str, optional
            Name to use for processed data file (without extension), by default "processed".
            The `.h5ad` suffix will be appended automatically.
        dataset_kwargs : dict[str, Any] | None, optional
            Keyword arguments to pass to the Dataset constructor, by default None.
        transform_kwargs : dict[str, Any] | None, optional
            Keyword arguments to pass to DatasetTransformer, by default None.
            Values for `source_h5ad_file_name` and `stratifying_column` will be derived from
            the relevant Dataset if possible. Either `source_h5ad_file_name` or `data_dir`
            must be included.
        transform_datasets : bool, optional
            Whether to apply transformations to datasets, by default True.
        limit_dataset_samples : int | Mapping[str, int] | None, optional
            Limit number of samples per dataset split, by default None.
            - If int: All datasets will be limited to this size.
            - If dict: Different splits can have different limits, e.g.,
            {"train": 100, "val": 50, "test": 10}. Missing keys won't be limited.
            - If None: No limits applied.
        limit_genes : "tokenizer" | "protein_coding" | None, optional
            Strategy to limit the genes in the dataset, by default "tokenizer".
            - "tokenizer": Uses intersection of genes in tokenizer and dataset (avoids UNKs).
            - "protein_coding": Uses only HGNC protein coding genes.
            - "L1000": use the L1000 landmark genes
            - None: No gene filtering.
        balancing_label_column : str | None, optional
            Column to use for balancing the dataset, by default None.
            Ensures that each label from this column is equally likely to appear in training batches.
        max_length : int | "scheduled", optional
            Maximum length of input sequences, by default 512.
            - "scheduled": then the `batch_size` will be controlled by the callback `BatchSizeScheduler`.
        padding : PaddingStrategy | str | bool, optional
            Padding strategy for input sequences, by default "max_length".
            Options: "max_length", "longest", True, False.
        truncation : TruncationStrategy | bool, optional
            Truncation strategy for input sequences, by default True.
            Options: "only_first", "only_second", "longest_first", True, False.
        pad_to_multiple_of : int, optional
            Pad sequence length to be multiple of this value, by default 16.
            Helps with efficiency on some hardware accelerators.
        sequence_order : str | None, optional
            Determines the order of sequences in the input, by default None.
            Options: "random", "sorted" (sorted by expression value in descending order).
        sequence_dropout_factor : int | float | None, optional
            Factor for dropping out sequences during training, by default None.
            If float: Probability of dropping sequences.
            If int: Number of contiguous samples to drop out.
        pad_zero_expression_strategy (dict) | None: The strategy for handling
            zero-expression genes. Required key "strategy" can have values:
                - 'batch_wise': Prioritizes genes expressed in the batch.
                - 'random': Randomly selects zero-expression genes if needed.
            optional keys:
            - interleave_zero_ratio (float): Interleave a fixed ratio of zeros
              (0 means first include all nonzero values, 1 means put all zeros first)
            If no strategy is supplied, all zeros will be exposed.
        collation_strategy : "language_modeling" | "sequence_classification" | "sequence_labeling" | "multitask", optional
            Strategy for collating samples into batches, by default "language_modeling".
            - "language_modeling": For masked language modeling and causal language modeling tasks.
            - "sequence_classification": For classification tasks on entire sequences.
            - "sequence_labeling": For token-level classification/labeling tasks.
            - "multitask": For multi-task learning with multiple objectives.
        mlm : bool, optional
            Whether to use masked language modeling, by default False.
        change_ratio : float, optional
            Proportion of tokens to modify during masking, by default 0.15.
        mask_ratio : float, optional
            Proportion of changed tokens to mask, by default 1.0.
            Only relevant when mlm=True.
        switch_ratio : float, optional
            Proportion of changed tokens to switch with random tokens, by default 0.0.
            Only relevant when mlm=True.
        masking_strategy : MaskingStrategy | None, optional
            Strategy for masking tokens, by default None.
            Refer to MaskingStrategy documentation for detailed configuration options.
        prevent_attention_to_masked: bool = False
            Controls whether the attention mask blocks attention to masked tokens.
            Based on scGPT's attention strategy and only applies when mlm=True.
            When multiple fields are masked, this requires comask_across_fields=True.
            Will raise an error if True with comask_across_fields=False and multiple
            masked fields. Defaults to False.
        comask_across_fields: bool = False
            Determines if fields are masked at the same token positions in the sequence.
            Only relevant when multiple fields have is_masked=True and mlm=True.
            When True, the model predicts both gene and expression value simultaneously.
            Task may be underdefined with random sequence order but defined with
            consistent sorting. Defaults to False.
        log_normalize_transform : bool, optional
            Whether to apply log normalization to expression data, by default False.
        median_normalization: bool, optional
            Whether to apply median normalization to expression data, by default False.
        rda_transform : "downsample" | "auto_align" | "equal" | int | None, optional
            RNA differential abundance transformation strategy, by default None.
            - "downsample": Applies RDA downsampling augmentation strategy as described in scFoundation.
            - "auto_align": Automatically selects maximum expression and sets target reads [T] value
            to that for all samples. Recommended for inference, not training.
            - "equal": Sets source reads [S] equal to target reads [T], preparing sequences in
            RDA style without upsampling.
            - int: Uses specified value as the target reads [T].
            - None: No transformation applied.
        map_orthologs: str | None, optional
            Mapping genes across species by leveraging orthologs and HUGO gene symbols (external gene names).
            - "mouse_to_human": Maps mouse genes to human genes using orthologs.
            - "human_to_mouse": Maps human genes to mouse genes using orthologs.
        batch_size : int | "scheduled", optional
            Number of samples per batch, by default 32.
            - int: number of samples per batch.
            - "scheduled": then the `batch_size` will be controlled by the callback `BatchSizeScheduler`.
            - None: defaults 32.
        num_workers : int, optional
            Number of worker processes for data loading, by default 0.
            0 means data loading happens in the main process.
        shuffle : bool, optional
            Whether to shuffle the data during training, by default False.
        """
        super().__init__()

        self.data_dir = Path(data_dir) if data_dir is not None else data_dir
        self.processed_name = processed_name + ".h5ad"
        self.dataset_kwargs = dataset_kwargs
        self.transform_kwargs = transform_kwargs
        self.tokenizer = tokenizer
        self.fields = fields
        self.label_columns = label_columns
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of
        self.truncation = truncation
        self.limit_dataset_samples = limit_dataset_samples
        self.shuffle = shuffle
        self.sequence_order = sequence_order
        self.log_normalize_transform = log_normalize_transform
        self.median_normalization = median_normalization
        self.rda_transform = rda_transform
        if isinstance(pad_zero_expression_strategy, str):
            logger.warning(
                "str arguments for `pad_zero_expression_strategy` are deprecated"
            )
            logger.warning(
                f"{pad_zero_expression_strategy} -> { {'strategy': pad_zero_expression_strategy} }"
            )
            pad_zero_expression_strategy = {"strategy": pad_zero_expression_strategy}
        self.pad_zero_expression_strategy = pad_zero_expression_strategy
        self.collation_strategy = collation_strategy
        if isinstance(masking_strategy, partial):
            masking_strategy = masking_strategy(tokenizer=tokenizer)
        self.masking_strategy = masking_strategy
        if isinstance(self.masking_strategy, WCEDMasker):
            self.masker = masking_strategy
        elif mlm:
            self.masker = Masker(
                change_ratio=change_ratio,
                mask_ratio=mask_ratio,
                switch_ratio=switch_ratio,
                tokenizer=tokenizer,
                prevent_attention_to_masked=prevent_attention_to_masked,
                comask_across_fields=comask_across_fields,
                masking_strategy=masking_strategy,
            )
        else:
            self.masker = None
        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        self.transform_datasets = transform_datasets
        self.sequence_dropout_factor = sequence_dropout_factor
        self.balancing_label_column = balancing_label_column
        self.limit_genes = limit_genes
        self.stratifying_label = None
        if self.label_columns:
            self.stratifying_label = next(
                (
                    label.label_column_name
                    for label in self.label_columns
                    if label.is_stratification_label
                ),
                None,
            )
        self.map_orthologs = map_orthologs
        self.tokenize_kwargs = tokenize_kwargs
        self.label_ontology_artifacts = self._setup_label_ontology()
        self.bootstrap_test_dataloader = bootstrap_test_dataloader
        self.__post_init__()

    def __post_init__(self):
        pass

    def get_trainer_callbacks(self) -> list:
        """Here datamodules can add their own callbacks to callback list for PL trainer."""
        if self.masking_strategy and hasattr(
            self.masking_strategy, "get_trainer_callback"
        ):
            return [self.masking_strategy.get_trainer_callback()]
        return []

    def _setup_label_ontology(self) -> LabelOntologyTuple | None:
        """
        Checks all labels for label_ontology, if found, creates a dictionary with label ontologies
        and additional label_dict from ontology for all such labels.
        """
        if not self.label_columns:
            return None

        columns = [i for i in self.label_columns if i.label_ontology]
        if not columns:
            return None

        label_ontology = {
            i.label_column_name: LabelOntology.load_ontology(i.label_ontology)
            for i in columns
        }
        label_dict = {
            label_name: ontology.get_label_dictionary()
            for label_name, ontology in label_ontology.items()
        }

        out = DataModule.LabelOntologyTuple(label_dict, label_ontology)
        return out

    def get_dataset_instance(self) -> BaseRNAExpressionDataset:
        """
        Internal function to return a dataset instance.

        Used to expose the label_dict and other dataset level variables to callers
        in a split independent way.

        Already instantiated datasets are used if available, with a preference for
        train, dev, test and finally predict.
        If this function is called before dm.setup(), and there are no datasets, it will
        instantiate one with split=None.

        Returns
        -------
            BaseRNAExpressionDataset: a dataset from the underlying dm
        """
        potential_datasets = [
            self.train_dataset,
            self.dev_dataset,
            self.test_dataset,
            self.predict_dataset,
        ]
        instantiated_datasets = [d for d in potential_datasets if d is not None]
        if not instantiated_datasets:
            dataset = self.DATASET_FACTORY(**self.dataset_kwargs, split=None)
        else:
            dataset = instantiated_datasets.pop(0)
            if isinstance(dataset, Subset):
                dataset = dataset.dataset
        return dataset

    @property
    def label_dict(self) -> None | dict[str, dict[str, int]]:
        """
        Dictionary of labels supported by dataset.

        If dataset has no labels, returns None
        """
        dataset = self.get_dataset_instance()

        ontology_dict = (
            self.label_ontology_artifacts.label_ontology
            if self.label_ontology_artifacts
            else None
        )
        label_dict = getattr(dataset, "label_dict", ontology_dict)

        if ontology_dict and label_dict is not ontology_dict:
            label_dict.update(self.label_ontology_artifacts.label_dict)

        return label_dict

    @property
    def label_output_size(self) -> dict[str, int] | None:
        """
        Number of labels for predictive task or None if no labels available.

        Returns
        -------
          dict[str,int]: dict with number of label values for each supported label in dataset
             or None if no labels available
        """
        if self.label_dict and self.label_columns:
            return {
                label.label_column_name: len(self.label_dict[label.label_column_name])
                for label in self.label_columns
            }
        return None

    def update_tokenizer(self) -> None:
        vocab_to_load = [
            (
                field,
                (
                    self.get_vocab_for_field(field.field_name)
                    if field.vocab_update_strategy == "dynamic"
                    else list(self.tokenizer.get_field_vocab(field.field_name).keys())
                ),
            )
            for field in self.fields
        ]

        for field in self.fields:
            self.tokenizer.reset_tokenizer_vocab(field_name=field.field_name)

        for field, vocab in vocab_to_load:
            self.tokenizer.add_tokens_for_field(
                field_name=field.field_name,
                token_to_add=vocab,
                ### TODO: not clear what should happen here.
                # add_special_tokens=field.vocab_update_strategy == "dynamic",
            )
        self.tokenizer.sanitize_special_tokens()

    def prepare_data(self) -> None:
        if not self.transform_datasets:
            return
        if self.transform_kwargs is None:
            transform_kwargs = {}
        else:
            transform_kwargs = self.transform_kwargs
        self.processed_data_file = self._read_processed_data_file_from_transform_kwargs(
            transform_kwargs
        )
        source_h5ad_file_name = self._read_source_h5ad_file_names_from_transform_kwargs(
            transform_kwargs
        )

        transforms = transform_kwargs.get("transforms", None)

        if transforms is None:
            transforms = getattr(self.DATASET_FACTORY, "DEFAULT_TRANSFORMS", None)

        transformer = self.DATASET_TRANSFORMER_FACTORY(
            source_h5ad_file_name=source_h5ad_file_name,
            transforms=transforms,
            stratifying_label=self.stratifying_label,
            split_column_name=transform_kwargs.get("split_column_name", None),
            split_weights=transform_kwargs.get("split_weights", None),
            random_state=transform_kwargs.get("random_state", 42),
        )
        processed_data = transformer.process_datasets()

        processed_data.write_h5ad(self.processed_data_file)

        # this process should happen only once
        self.transform_datasets = False
        super().prepare_data()

    def _read_source_h5ad_file_names_from_transform_kwargs(
        self, transform_kwargs: dict
    ):
        if "source_h5ad_file_name" in transform_kwargs:
            return transform_kwargs["source_h5ad_file_name"]

        if self.data_dir is None:
            raise ValueError(
                "You must set `data_dir` or for `transform_kwargs.source_h5ad_file_name`"
            )
        if not hasattr(self.DATASET_FACTORY, "source_h5ad_file_name"):
            raise ValueError(
                "When using `data_dir` option your dataset class must have the `source_h5ad_file_name` defined"
            )
        return self.data_dir / "h5ad" / self.DATASET_FACTORY.source_h5ad_file_name

    def _read_processed_data_file_from_transform_kwargs(self, transform_kwargs: dict):
        if "processed_h5ad_file_name" in transform_kwargs:
            return transform_kwargs["processed_h5ad_file_name"]
        if self.data_dir is None:
            raise ValueError(
                "You must set `data_dir` or for `transform_kwargs.processed_h5ad_file_name`"
            )
        return self.data_dir / self.processed_name

    def _dataset_sample_limit(self, dataset_split):
        if isinstance(self.limit_dataset_samples, Mapping):
            return self.limit_dataset_samples.get(dataset_split, None)
        return self.limit_dataset_samples

    def setup(self, stage=None) -> None:
        """
        Sets up the dataset.


        Args:
        ----
            stage (str, optional): Stage. Defaults to None.

        Raises:
        ------
            FileNotFoundError: If label dict file is missing.

        """
        self.dataset_kwargs = self._prepare_dataset_kwargs()

        if stage == "fit" or stage is None:
            self.train_dataset = self.DATASET_FACTORY(
                **self.dataset_kwargs,
                split="train",
                limit_samples=self._dataset_sample_limit("train"),
            )
            self.dev_dataset = self.DATASET_FACTORY(
                **self.dataset_kwargs,
                split="dev",
                limit_samples=self._dataset_sample_limit("dev"),
            )
        if stage == "validate" or stage is None:
            self.dev_dataset = self.DATASET_FACTORY(
                **self.dataset_kwargs,
                split="dev",
                limit_samples=self._dataset_sample_limit("dev"),
            )
        if stage == "test" or stage is None:
            self.test_dataset = self.DATASET_FACTORY(
                **self.dataset_kwargs,
                split="test",
                limit_samples=self._dataset_sample_limit("test"),
            )

        if stage == "predict" or stage is None:
            self.predict_dataset = self.DATASET_FACTORY(
                **self.dataset_kwargs,
                split=None,
                limit_samples=self._dataset_sample_limit("predict"),
            )

        if any(field.vocab_update_strategy == "dynamic" for field in self.fields):
            self.update_tokenizer()

    def _prepare_dataset_kwargs(self):
        """
        Runs steps necessary before initializing datasets.

        Can be modified in inheriting classes where steps are different.
        """
        final_dataset_kwargs = {}
        if self.dataset_kwargs and self.dataset_kwargs.get("backed"):
            logger.warning(
                "Unable to load dataset in backed mode inside datamodule. "
                "Filtering by split column will force the data to be copied to memory."
                "Backed mode is supported only for h5ad files that are not filtered after loading."
            )
        if self.dataset_kwargs:
            final_dataset_kwargs = {**self.dataset_kwargs}
        final_dataset_kwargs["stratifying_label"] = self.stratifying_label
        final_dataset_kwargs["processed_data_source"] = self.load_processed_data()
        final_dataset_kwargs["limit_samples_shuffle"] = self.shuffle
        if self.limit_genes is not None:
            final_dataset_kwargs["limit_genes"] = self._get_limited_gene_list(
                self.limit_genes
            )
        if self.label_columns:
            final_dataset_kwargs["label_columns"] = [
                label.label_column_name
                for label in self.label_columns
                if not label.is_regression_label
            ]
            final_dataset_kwargs["regression_label_columns"] = [
                label.label_column_name
                for label in self.label_columns
                if label.is_regression_label
            ]
        return final_dataset_kwargs

    def load_processed_data(self):
        if self.dataset_kwargs and "processed_data_source" in self.dataset_kwargs:
            # case: user provides anndata object already prepared in dataset_kwargs
            if isinstance(self.dataset_kwargs["processed_data_source"], AnnData):
                return self.dataset_kwargs["processed_data_source"]
            # case: user provides explicit path in dataset_kwargs
            self.processed_data_file = Path(
                self.dataset_kwargs["processed_data_source"]
            )
        # case: user assumes default processed file location inside data_dir
        elif self.data_dir:
            self.processed_data_file = self.data_dir / self.processed_name
        else:
            raise ValueError(
                "You must set `data_dir` or `dataset_kwargs.processed_data_source`"
            )
        processed_data = read_h5ad(self.processed_data_file)
        return processed_data

    def _get_limited_gene_list(self, limit_genes_keyword) -> list[str] | None:
        if limit_genes_keyword == "tokenizer":
            if "genes" in self.tokenizer.tokenizers:
                return [*self.tokenizer.get_field_vocab("genes")]
            else:
                logger.warning(
                    "Field 'genes' not found in tokenizer. "
                    '`limit_genes="tokenizer" not executed.'
                )
            return None
        if limit_genes_keyword == "protein_coding":
            tokenizer_genes = [*self.tokenizer.get_field_vocab("genes")]
            return [*{*get_protein_coding_genes()} & {*tokenizer_genes}]
        if limit_genes_keyword == "mouse_to_human_orthologs":
            return get_ortholog_genes(
                return_mapping=False, from_species="mmusculus", to_species="hsapiens"
            )
        if limit_genes_keyword == "human_to_mouse_orthologs":
            return get_ortholog_genes(
                return_mapping=False, from_species="hsapiens", to_species="mmusculus"
            )
        if limit_genes_keyword == "L1000":
            return get_L1000_genes()
        raise ValueError("Unsupported option passed for limit_genes")

    def get_vocab_for_field(self, field_name: str) -> list[str]:
        """
        Gets unique tokens for a field.

        Args:
        ----
            field_name (str): Field name

        Returns:
        -------
            list[str]: Unique tokens for field
        """
        dataset = self.get_dataset_instance()
        return [str(i) for i in dataset.get_vocab_for_field(field_name)]

    @staticmethod
    def get_balanced_weights_for_sampler(labels: np.array):
        # create mapping from cell type label to its weight
        unique_labels, labels_counts = np.unique(labels, return_counts=True)
        label_to_weight = {
            label: 1.0 / count for label, count in zip(unique_labels, labels_counts)
        }
        # retrieve the corresponding weight from label_to_weight
        samples_weight = np.array([label_to_weight[label] for label in labels])
        samples_weight = torch.from_numpy(samples_weight)
        return samples_weight

    def train_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for training.

        Returns
        -------
            DataLoader: DataLoader for training
        """
        if self.balancing_label_column is not None:
            shuffle = False
            labels = np.array(self.train_dataset.metadata[self.balancing_label_column])
            samples_weight = self.get_balanced_weights_for_sampler(labels)
            sampler = WeightedRandomSampler(
                samples_weight, len(samples_weight), replacement=True
            )

        else:
            sampler = None
            shuffle = self.shuffle

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            sampler=sampler,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for validation.

        Returns
        -------
            DataLoader: DataLoader for validation
        """
        return DataLoader(
            self.dev_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.val_collate_fn(),
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def val_collate_fn(self):
        # collate_fn is a property so collator is an obj
        collator = self.collate_fn
        if (
            hasattr(self.masking_strategy, "use_for_validation")
            and not self.masking_strategy.use_for_validation
        ):
            logger.info("Removing masking strategy from val/test dataloader")
            collator.masker = deepcopy(collator.masker)
            if hasattr(collator.masker, "masking_strategy"):
                collator.masker.masking_strategy = None
        return collator

    def test_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for validation.

        Returns
        -------
            DataLoader: DataLoader for validation
        """
        if self.bootstrap_test_dataloader:
            return self._bootstrap_test_dataloader()
        return self._test_dataloader()

    def _test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.val_collate_fn(),
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def _bootstrap_test_dataloader(self):
        num_samples = len(self.test_dataset)
        weights = torch.ones(num_samples)
        sampler = WeightedRandomSampler(
            weights, num_samples=num_samples, replacement=True
        )
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self.val_collate_fn(),
        )

    def predict_dataloader(self) -> DataLoader:
        """Returns a list of DataLoaders for prediction."""
        collate_fn = self.collate_fn
        collate_fn.label_columns = None
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    @property
    def collate_fn(self):
        """
        Returns a collate function.

        Returns
        -------
            Callable: Collate function
        """
        if self.rda_transform == "auto_align":
            dataset = self.get_dataset_instance()
            if hasattr(dataset, "max_counts"):
                rda_transform = int(self.get_dataset_instance().max_counts())
            else:
                raise ValueError(
                    f"RDA `auto_align` not supported for dataset of type {type(dataset)}"
                )
        else:
            rda_transform = self.rda_transform
        return MultiFieldCollator(
            tokenizer=self.tokenizer,
            label_dict=self.label_dict,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            fields=self.fields,
            label_columns=self.label_columns,
            truncation=self.truncation,
            collation_strategy=self.collation_strategy,
            max_length=self.max_length,
            masker=self.masker,
            sequence_order=self.sequence_order,
            log_normalize_transform=self.log_normalize_transform,
            median_normalization=self.median_normalization,
            rda_transform=rda_transform,
            pad_zero_expression_strategy=self.pad_zero_expression_strategy,
            sequence_dropout_factor=self.sequence_dropout_factor,
            map_orthologs=self.map_orthologs,
            tokenize_kwargs=self.tokenize_kwargs,
            label_ontology=self.label_ontology_artifacts.label_ontology
            if self.label_ontology_artifacts
            else None,
        )


class PerturbationDataModule(DataModule):
    DATASET_FACTORY: type[BasePerturbationDataset] = ...
    DATASET_TRANSFORMER_FACTORY: type[PerturbationDatasetTransformer] = ...

    def __init__(
        self,
        tokenizer: MultiFieldTokenizer,
        fields: list[FieldInfo],
        data_dir: str | Path | None = None,
        processed_name: str = "processed",
        dataset_kwargs: dict[str, Any] | None = None,
        label_columns: list[LabelColumnInfo] | None = None,
        transform_kwargs: dict[str, Any] | None = None,
        transform_datasets: bool = True,
        batch_size: int = 32,
        num_workers: int = 0,
        max_length: int = 512,
        padding: PaddingStrategy | str | bool = "max_length",
        truncation: TruncationStrategy | bool = True,
        pad_to_multiple_of: int = 16,
        collation_strategy: Literal[
            "language_modeling", "sequence_classification", "sequence_labeling"
        ] = "sequence_labeling",
        limit_dataset_samples: int | Mapping[str, int] | None = None,
        shuffle: bool = False,
        sequence_order: str | None = None,
        sequence_dropout_factor: int | float | None = None,
        log_normalize_transform: bool = False,
        median_normalization: bool = False,
        pad_zero_expression_strategy: str | None = None,
        balancing_label_column: str | None = None,
        perturbation_column_name: str = "perturbation",
        limit_genes: Literal["protein_coding", "tokenizer", None] = "tokenizer",
        sequence_label_extractor: None | WCEDMasker = None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            fields=fields,
            label_columns=label_columns,
            data_dir=data_dir,
            processed_name=processed_name,
            dataset_kwargs=dataset_kwargs,
            transform_kwargs=transform_kwargs,
            transform_datasets=transform_datasets,
            batch_size=batch_size,
            num_workers=num_workers,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            collation_strategy=collation_strategy,
            mlm=False,
            limit_dataset_samples=limit_dataset_samples,
            shuffle=shuffle,
            sequence_order=sequence_order,
            sequence_dropout_factor=sequence_dropout_factor,
            log_normalize_transform=log_normalize_transform,
            median_normalization=median_normalization,
            pad_zero_expression_strategy=pad_zero_expression_strategy,
            balancing_label_column=balancing_label_column,
            limit_genes=limit_genes,
            masking_strategy=sequence_label_extractor,
        )

        self.perturbation_column_name = perturbation_column_name

    def _prepare_dataset_kwargs(self):
        self.dataset_kwargs = super()._prepare_dataset_kwargs()
        unsupported_kwargs = {*self.dataset_kwargs.keys()} - {
            *self.DATASET_FACTORY.__init__.__annotations__.keys()
        }
        self.dataset_kwargs = {
            k: v for k, v in self.dataset_kwargs.items() if k not in unsupported_kwargs
        }
        logger.warning(f"Unsupported dataset kwargs removed: {unsupported_kwargs}")
        self.dataset_kwargs["perturbation_column_name"] = self.perturbation_column_name
        return self.dataset_kwargs

    def prepare_data(self) -> None:
        if not self.transform_datasets:
            return
        if self.transform_kwargs is None:
            transform_kwargs = {}
        else:
            transform_kwargs = self.transform_kwargs
        self.processed_data_file = self._read_processed_data_file_from_transform_kwargs(
            transform_kwargs
        )
        source_h5ad_file_names = (
            self._read_source_h5ad_file_names_from_transform_kwargs(transform_kwargs)
        )

        transformer = self.DATASET_TRANSFORMER_FACTORY(
            tokenizer=self.tokenizer,
            source_h5ad_file_names=source_h5ad_file_names,
            transforms=transform_kwargs.get("transforms", None),
            stratifying_label=self.stratifying_label,
            perturbation_column_name=self.perturbation_column_name,
            split_column_name=transform_kwargs.get("split_column_name", None),
            group_column_name=transform_kwargs.get("group_column_name", None),
            split_weights=transform_kwargs.get("split_weights", None),
            random_state=transform_kwargs.get("random_state", 42),
        )
        processed_data = transformer.process_datasets()

        processed_data.uns.pop(
            "rank_genes_groups", None
        )  # This is huge and we dont use it later in the code
        processed_data.write_h5ad(self.processed_data_file)

        # this process should happen only once
        self.transform_datasets = False
        super().prepare_data()

    def _read_source_h5ad_file_names_from_transform_kwargs(
        self, transform_kwargs: dict
    ):
        if "source_h5ad_file_names" in transform_kwargs:
            return transform_kwargs["source_h5ad_file_names"]

        if self.data_dir is None:
            raise ValueError(
                "You must set `data_dir` or for `transform_kwargs.source_h5ad_file_name`"
            )
        return [
            self.data_dir / "h5ad" / source_h5ad_file_name
            for source_h5ad_file_name in self.DATASET_FACTORY.source_h5ad_file_names
        ]


class DNASeqDataModule(pl.LightningDataModule):
    DATASET_FACTORY: type[BaseDNASeqDataset] = BaseDNASeqDataset

    def __init__(
        self,
        tokenizer: MultiFieldTokenizer,
        fields: list[FieldInfo],
        label_columns: list[LabelColumnInfo],
        data_dir: str | Path | None = None,
        dataset_kwargs: dict[str, Any] | None = None,
        split_ratio: dict | None = None,
        transform_kwargs: dict[str, Any] | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
        max_length: int = 512,
        padding: PaddingStrategy | str | bool = "max_length",
        truncation: TruncationStrategy | bool = True,
        pad_to_multiple_of: int = 16,
        collation_strategy: Literal[
            "language_modeling",
            "sequence_classification",
            "sequence_labeling",
        ] = "sequence_classification",
        limit_dataset_samples: int | Mapping[str, int] | None = None,
        sequence_order: str | None = None,
        shuffle: bool = False,
        sequence_dropout_factor: int | float | None = None,
        balancing_label_column: str | None = None,
    ):
        """
        Construct the data module.

        Args:
        ----
            tokenizer (MultiFieldTokenizer): Tokenizer to use
            fields (list[FieldInfo]): List of FieldInfo objects
            data_dir (str|Path|None): Look for raw and processed data files in `data_dir` using
              default file names: raw data in data_dir/h5ad/dataset_name.h5ad and processsed
              data in data_dir/processed.h5ad. If not supplied, dataset_kwargs and
              transform_kwargs must have full paths for the h5ad files.
            processed_name (str): name to use for processed data file. Defaults to "processed".
            dataset_kwargs: Keyword arguments to pass to Dataset
            transform_kwargs (dict | None): kwargs to pass to DatasetTransformer. Values
              for `source_h5ad_file_name` and `stratifying_column` will be derived from
              the relevant Dataset if possible.
            batch_size (int, optional): Batch size. Defaults to 32.
            num_workers (int, optional): Number of workers for DataLoader. Defaults to 0.
            max_length (int, optional): Maximum length of input sequences. Defaults to 512.
            padding (PaddingStrategy, optional): Padding strategy. Defaults to True. Available options: PaddingStrategy.MAX_LENGTH, PaddingStrategy.LONGEST.
            truncation (TruncationStrategy, optional): Truncation strategy. Defaults to TruncationStrategy.ONLY_FIRST. Available options: TruncationStrategy.ONLY_FIRST, TruncationStrategy.ONLY_SECOND, TruncationStrategy.LONGEST_FIRST, TruncationStrategy.LONGEST_SECOND.
            limit_dataset_samples (int | dict | None, optional): Limit number of training samples.
              If an integer is passed, all internal datasets will be limited to that size.
              If a dict is passed, different internal datasets will be limited as requested,
              such that {"train":100, "dev":50, "test":10, "predict":5} will limit the different
              datasets accordingly. If dataset names are missing from the dict they will not be
              limited. Defaults to None, in which case none of the datasets are limited.

        """
        super().__init__()

        self.data_dir = Path(data_dir) if data_dir is not None else data_dir
        self.dataset_kwargs = dataset_kwargs
        self.split_ratio = split_ratio
        self.transform_kwargs = transform_kwargs
        self.tokenizer = tokenizer
        self.fields = fields
        self.label_columns = label_columns
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of
        self.truncation = truncation
        self.limit_dataset_samples = limit_dataset_samples
        self.sequence_order = sequence_order
        self.collation_strategy = collation_strategy
        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        self.shuffle = shuffle
        self.sequence_dropout_factor = sequence_dropout_factor
        self.balancing_label_column = balancing_label_column

        self.masker = None

        self.stratifying_label = None
        if self.label_columns:
            self.stratifying_label = next(
                (
                    label.label_column_name
                    for label in self.label_columns
                    if label.is_stratification_label
                ),
                None,
            )

    def get_trainer_callbacks(self) -> list:
        """Here datamodules can add their own callbacks to callback list for PL trainer."""
        return []

    def get_dataset_instance(self) -> BaseDNASeqDataset:
        """
        Internal function to return a dataset instance.

        Used to expose the label_dict and other dataset level variables to callers
        in a split independent way.

        Already instantiated datasets are used if available, with a preference for
        train, dev, test and finally predict.
        If this function is called before dm.setup(), and there are no datasets, it will
        instantiate one with split=None.

        Returns
        -------
            BaseDNASeqDataset: a dataset from the underlying dm
        """
        potential_datasets = [
            self.train_dataset,
            self.dev_dataset,
            self.test_dataset,
            self.predict_dataset,
        ]
        instantiated_datasets = [d for d in potential_datasets if d is not None]
        if not instantiated_datasets:
            dataset = self.DATASET_FACTORY(**self.dataset_kwargs, split=None)
        else:
            dataset = instantiated_datasets.pop(0)
            if isinstance(dataset, Subset):
                dataset = dataset.dataset
        return dataset

    @property
    def label_dict(self):
        """
        Dictionary of labels supported by dataset.

        If dataset has no labels, returns None
        """
        dataset = self.get_dataset_instance()

        return getattr(dataset, "label_dict", None)

    @property
    def label_output_size(self) -> dict[str, int] | None:
        """
        Number of labels for predictive task or None if no labels available.

        Returns
        -------
          dict[str,int]: dict with number of label values for each supported label in dataset
             or None if no labels available
        """
        if self.label_dict:
            return {k: len(v) for k, v in self.label_dict.items()}
        return None

    def update_tokenizer(self) -> None:
        vocab_to_load = [
            (
                field,
                (
                    self.get_vocab_for_field(field.field_name)
                    if field.vocab_update_strategy == "dynamic"
                    else list(self.tokenizer.get_field_vocab(field.field_name).keys())
                ),
            )
            for field in self.fields
        ]

        for field in self.fields:
            self.tokenizer.reset_tokenizer_vocab(field_name=field.field_name)

        for field, vocab in vocab_to_load:
            self.tokenizer.add_tokens_for_field(
                field_name=field.field_name,
                token_to_add=vocab,
                ### TODO: not clear what should happen here.
                # add_special_tokens=field.vocab_update_strategy == "dynamic",
            )
        self.tokenizer.sanitize_special_tokens()

    def prepare_data(self) -> None:
        return

    def _prepare_dataset_kwargs(self):
        """
        Runs steps necessary before initializing datasets.

        Can be modified in inheriting classes where steps are different.
        """
        final_dataset_kwargs = {}
        if self.dataset_kwargs:
            final_dataset_kwargs = {**self.dataset_kwargs}
        # This is data_dir for now, if we need transforms then this needs to be updated accordingly.
        if "processed_data_source" not in final_dataset_kwargs:
            if self.data_dir is None:
                raise ValueError(
                    "one of the datadir or processed data source has to be provided"
                )
            final_dataset_kwargs["processed_data_source"] = self.data_dir
        if self.label_columns:
            final_dataset_kwargs["label_columns"] = [
                label.label_column_name
                for label in self.label_columns
                if not label.is_regression_label
            ]
            final_dataset_kwargs["regression_label_columns"] = [
                label.label_column_name
                for label in self.label_columns
                if label.is_regression_label
            ]
        return final_dataset_kwargs

    def _dataset_sample_limit(self, dataset_split):
        if isinstance(self.limit_dataset_samples, Mapping):
            return self.limit_dataset_samples.get(dataset_split, None)
        return self.limit_dataset_samples

    def setup(self, stage=None) -> None:
        """
        Sets up the dataset.


        Args:
        ----
            stage (str, optional): Stage. Defaults to None.

        Raises:
        ------
            FileNotFoundError: If label dict file is missing.

        """
        self.dataset_kwargs = self._prepare_dataset_kwargs()

        if stage == "fit" or stage is None:
            self.train_dataset = self.DATASET_FACTORY(
                **self.dataset_kwargs,
                split="train",
            )
            self.dev_dataset = self.DATASET_FACTORY(
                **self.dataset_kwargs,
                split="dev",
            )
        if stage == "validate" or stage is None:
            self.dev_dataset = self.DATASET_FACTORY(
                **self.dataset_kwargs,
                split="dev",
            )
        if stage == "test" or stage is None:
            self.test_dataset = self.DATASET_FACTORY(
                **self.dataset_kwargs,
                split="test",
            )

        if stage == "predict" or stage is None:
            self.predict_dataset = self.DATASET_FACTORY(
                **self.dataset_kwargs,
                split=None,
            )

        if any(field.vocab_update_strategy == "dynamic" for field in self.fields):
            self.update_tokenizer()

    def get_vocab_for_field(self, field_name: str) -> list[str]:
        """
        Gets unique tokens for a field.

        Args:
        ----
            field_name (str): Field name

        Returns:
        -------
            list[str]: Unique tokens for field
        """
        dataset = self.get_dataset_instance()
        return [str(i) for i in dataset.get_vocab_for_field(field_name)]

    @staticmethod
    def get_balanced_weights_for_sampler(labels: np.array):
        # create mapping from cell type label to its weight
        unique_labels, labels_counts = np.unique(labels, return_counts=True)
        label_to_weight = {
            label: 1.0 / count for label, count in zip(unique_labels, labels_counts)
        }
        # retrieve the corresponding weight from label_to_weight
        samples_weight = np.array([label_to_weight[label] for label in labels])
        samples_weight = torch.from_numpy(samples_weight)
        return samples_weight

    def train_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for training.

        Returns
        -------
            DataLoader: DataLoader for training
        """
        if self.balancing_label_column is not None:
            shuffle = False
            labels = np.array(self.train_dataset.metadata[self.balancing_label_column])
            samples_weight = self.get_balanced_weights_for_sampler(labels)
            sampler = WeightedRandomSampler(
                samples_weight, len(samples_weight), replacement=True
            )

        else:
            sampler = None
            shuffle = self.shuffle

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            sampler=sampler,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for validation.

        Returns
        -------
            DataLoader: DataLoader for validation
        #
        """
        return DataLoader(
            self.dev_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for validation.

        Returns
        -------
            DataLoader: DataLoader for validation
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self) -> DataLoader:
        """Returns a list of DataLoaders for prediction."""
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    @property
    def collate_fn(self):
        """
        Returns a collate function.

        Returns
        -------
            Callable: Collate function
        """
        return MultiFieldCollator(
            tokenizer=self.tokenizer,
            fields=self.fields,
            label_columns=self.label_columns,
            label_dict=self.label_dict,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            sequence_order=self.sequence_order,
            sequence_dropout_factor=self.sequence_dropout_factor,
            collation_strategy=self.collation_strategy,
            masker=self.masker,
        )
