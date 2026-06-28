import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from bmfm_targets.config import FieldInfo, LabelColumnInfo
from bmfm_targets.datasets import DatasetTransformer
from bmfm_targets.datasets.samplers import ConditionHomogeneousBatchSampler
from bmfm_targets.datasets.scRNA2ChIP.base_scrna2chip_dataset import (
    BasescRNA2ChIPDataset,
)
from bmfm_targets.tokenization import MultiFieldTokenizer
from bmfm_targets.training.data_module import DataModule
from bmfm_targets.training.masking.strategy import WCEDMasker

logger = logging.getLogger(__name__)


class _ChipCollator:
    """
    Picklable collate wrapper that appends the ChIP population tensor to each batch.

    Must be a module-level class (not a local closure) so multiprocessing workers
    can pickle it when num_workers > 0.

    The population tensor is placed at both ``batch["chip_population"]`` (backward
    compatibility) and ``batch["labels"]["chip_population"]`` (required so the
    composed loss system, which only receives ``batch["labels"]``, can access it).
    """

    def __init__(self, base_collate, celltype_column, chip_populations, vocab_size):
        self.base_collate = base_collate
        self.celltype_column = celltype_column
        self.chip_populations = chip_populations
        self.vocab_size = vocab_size

    def __call__(self, examples):
        batch = self.base_collate(examples)
        celltype = examples[0].metadata.get(self.celltype_column, "unknown")
        if celltype in self.chip_populations:
            population = self.chip_populations[celltype]
        else:
            logger.warning(
                f"Celltype {celltype!r} not found in chip_populations; "
                "using a zero-filled fallback tensor. "
                "Check that the celltype column and dataset are aligned."
            )
            population = torch.zeros(1, self.vocab_size)
        # Top-level key for backward compatibility.
        batch["chip_population"] = population
        # Under labels so that composed loss tasks can read it via LabelSource.
        if "labels" not in batch:
            batch["labels"] = {}
        batch["labels"]["chip_population"] = population
        return batch


class scRNA2ChIPDataModule(DataModule):
    DATASET_FACTORY: type[BasescRNA2ChIPDataset] = ...
    DATASET_TRANSFORMER_FACTORY: type[DatasetTransformer] = ...

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
        use_ot_batching: bool = False,
        celltype_column: str = "tissue_label",
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
        self.use_ot_batching = use_ot_batching
        self.celltype_column = celltype_column
        self.chip_populations: dict[str, torch.Tensor] = {}  # built in setup()

    def _prepare_dataset_kwargs(self):
        self.dataset_kwargs = super()._prepare_dataset_kwargs()
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
            source_h5ad_file_name=source_h5ad_file_names[0],
            split_weights=transform_kwargs.get("split_weights", None),
            transforms=transform_kwargs.get("transforms", None),
            split_column_name=transform_kwargs.get("split_column_name", None),
            stratifying_label=self.stratifying_label,
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

    def setup(self, stage=None):
        super().setup(stage)
        if self.use_ot_batching and (stage in ("fit", "validate") or stage is None):
            if not self.chip_populations:
                self._build_chip_populations()

    # def _build_chip_populations(self):
    #     """
    #     Build celltype -> ChIP population tensor map, aligned to tokenizer vocab.
    #
    #     Each ChIP cell's gene values are scattered into a full-vocab-sized vector so
    #     the population tensor [M, vocab_size] is in the same space as the WCED logits.
    #     """
    #     chip = self.train_dataset.chipseq_cells  # AnnData slice
    #     col = self.celltype_column
    #     gene_tokenizer = self.tokenizer.get_field_tokenizer("genes")
    #     vocab_size = len(gene_tokenizer.vocab)
    #
    #     # Map chip gene names -> vocab ids (missing genes -> -1, will be skipped)
    #     chip_genes = list(chip.var_names)
    #     gene_ids = [
    #         gene_tokenizer.convert_tokens_to_ids(g) if g in gene_tokenizer.vocab else -1
    #         for g in chip_genes
    #     ]
    #     valid_mask = [i for i, gid in enumerate(gene_ids) if gid >= 0]
    #     valid_vocab_ids = [gene_ids[i] for i in valid_mask]
    #
    #     for celltype, idx in chip.obs.groupby(col, observed=True).groups.items():
    #         mat = chip[idx].X
    #         if hasattr(mat, "toarray"):
    #             mat = mat.toarray()
    #         mat = mat[:, valid_mask]  # [M, n_valid_genes]
    #         # scatter into vocab-sized tensor
    #         pop = torch.zeros(len(idx), vocab_size, dtype=torch.float32)
    #         pop[:, valid_vocab_ids] = torch.tensor(mat, dtype=torch.float32)
    #         self.chip_populations[celltype] = pop
    #
    #     logger.info(
    #         f"Built chip_populations for {len(self.chip_populations)} celltypes, "
    #         f"aligned {len(valid_mask)}/{len(chip_genes)} genes to vocab (size {vocab_size})"
    #     )

    def _build_chip_populations(self):
        """
        Build celltype -> ChIP population tensor map, aligned to tokenizer vocab.
        """
        # 1. Collect all loaded datasets
        chip_datasets = []
        if getattr(self, "train_dataset", None) is not None:
            chip_datasets.append(self.train_dataset.chipseq_cells)
        if getattr(self, "dev_dataset", None) is not None:
            chip_datasets.append(self.dev_dataset.chipseq_cells)
        if getattr(self, "test_dataset", None) is not None:
            chip_datasets.append(self.test_dataset.chipseq_cells)

        if not chip_datasets:
            return

        col = self.celltype_column
        gene_tokenizer = self.tokenizer.get_field_tokenizer("genes")
        vocab_size = len(gene_tokenizer.vocab)

        # 2. Extract and assign populations
        for chip in chip_datasets:
            chip_genes = list(chip.var_names)
            gene_ids = [
                gene_tokenizer.convert_tokens_to_ids(g) if g in gene_tokenizer.vocab else -1
                for g in chip_genes
            ]
            valid_mask = [i for i, gid in enumerate(gene_ids) if gid >= 0]
            valid_vocab_ids = [gene_ids[i] for i in valid_mask]

            for celltype, idx in chip.obs.groupby(col, observed=True).groups.items():
                mat = chip[idx].X
                if hasattr(mat, "toarray"):
                    mat = mat.toarray()
                mat = mat[:, valid_mask]

                pop = torch.zeros(len(idx), vocab_size, dtype=torch.float32)
                pop[:, valid_vocab_ids] = torch.tensor(mat, dtype=torch.float32)

                # Directly assign since there's no overlap across splits
                self.chip_populations[celltype] = pop
        logger.info(
            f"Built chip_populations for {len(self.chip_populations)} celltypes, "
            f"aligned {len(valid_mask)}/{len(chip_genes)} genes to vocab (size {vocab_size})"
        )

    def train_dataloader(self):
        if not self.use_ot_batching:
            return super().train_dataloader()

        obs_conditions = self.train_dataset.metadata[self.celltype_column].reset_index(
            drop=True
        )
        batch_sampler = ConditionHomogeneousBatchSampler(
            obs_conditions=obs_conditions,
            batch_size=self.batch_size,
        )
        vocab_size = len(self.tokenizer.get_field_tokenizer("genes").vocab)
        collate_fn = _ChipCollator(
            base_collate=self.collate_fn,
            celltype_column=self.celltype_column,
            chip_populations=self.chip_populations,
            vocab_size=vocab_size,
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """
        Returns a DataLoader for validation.

        When ``use_ot_batching`` is True, mirrors the training loader: batches are
        cell-type homogeneous (via ``ConditionHomogeneousBatchSampler``) and the
        ``_ChipCollator`` injects ``batch["labels"]["chip_population"]`` so the
        composed OT loss can access the reference population during validation.

        When ``use_ot_batching`` is False, delegates to the base implementation.

        Returns
        -------
            DataLoader: DataLoader for validation.
        """
        if not self.use_ot_batching:
            return super().val_dataloader()

        obs_conditions = self.dev_dataset.metadata[self.celltype_column].reset_index(
            drop=True
        )
        batch_sampler = ConditionHomogeneousBatchSampler(
            obs_conditions=obs_conditions,
            batch_size=self.batch_size,
        )
        vocab_size = len(self.tokenizer.get_field_tokenizer("genes").vocab)
        collate_fn = _ChipCollator(
            base_collate=self.val_collate_fn(),
            celltype_column=self.celltype_column,
            chip_populations=self.chip_populations,
            vocab_size=vocab_size,
        )

        return DataLoader(
            self.dev_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )

    def _read_source_h5ad_file_names_from_transform_kwargs(
        self, transform_kwargs: dict
    ):
        if "source_h5ad_file_name" in transform_kwargs:
            return [
                self.data_dir / (transform_kwargs["source_h5ad_file_name"] + ".h5ad")
            ]
        raise ValueError(
            "You must set `transform_kwargs.source_h5ad_file_name` or use an already transformed data"
        )


class ConcatDataset(BasescRNA2ChIPDataset):
    DATASET_NAME = "concat_chip_subsampled_scrna"


class ConcatDataModule(scRNA2ChIPDataModule):
    """Lightning DataModule for the concatenated scRNA + ChIP translation dataset."""

    DATASET_FACTORY = ConcatDataset
    DATASET_TRANSFORMER_FACTORY = DatasetTransformer
