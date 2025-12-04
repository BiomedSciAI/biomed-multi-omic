import inspect
import logging

from litdata import CombinedStreamingDataset, StreamingDataLoader, StreamingDataset

from bmfm_targets.training.data_module import DataModule

logger = logging.getLogger(__name__)


class StreamingDataModule(DataModule):
    """
    PyTorch Lightning DataModule for scRNA datasets with option to save dataloader state.

    Attributes
    ----------
        dataset_kwargs: Keyword arguments for dataset
        tokenizer (MultiFieldTokenizer): Tokenizer to use
        fields (list[FieldInfo]): List of FieldInfo objects
        batch_size (int, optional): Batch size. Defaults to 32.
        num_workers (int, optional): Number of workers for DataLoader. Defaults to 1.
        max_length (int, optional): Maximum length of input sequences. Defaults to 512.
        padding (PaddingStrategy, optional): Padding strategy. Defaults to True. Available options: PaddingStrategy.MAX_LENGTH, PaddingStrategy.LONGEST.
        truncation (TruncationStrategy, optional): Truncation strategy. Defaults to TruncationStrategy.ONLY_FIRST. Available options: TruncationStrategy.ONLY_FIRST, TruncationStrategy.ONLY_SECOND, TruncationStrategy.LONGEST_FIRST, TruncationStrategy.LONGEST_SECOND.
    """

    state_entry_name = "datamodule_state"
    DATASET_FACTORY: type[StreamingDataset] = None

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        if "limit_dataset_samples" in kwargs:
            raise ValueError(
                "Streaming data module does not support limit_dataset_samples, use task's limit_{train,val,test}_batches"
            )

        super().__init__(*args, **kwargs)

    def prepare_data(self) -> None:
        return

    def _prepare_dataset_kwargs(self):
        final_dataset_kwargs = {}
        if self.dataset_kwargs:
            final_dataset_kwargs = {**self.dataset_kwargs}
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
        if self.limit_genes is not None:
            if "limit_genes" in inspect.signature(self.DATASET_FACTORY).parameters:
                final_dataset_kwargs["limit_genes"] = self._get_limited_gene_list(
                    self.limit_genes
                )
            else:
                logger.warning(
                    "`limit_genes` requested but not supported "
                    f"for dataset {self.DATASET_FACTORY.__name__}."
                )

        return final_dataset_kwargs

    def setup(self, stage: str = None) -> None:
        self.dataset_kwargs = self._prepare_dataset_kwargs()
        if any(field.vocab_update_strategy == "dynamic" for field in self.fields):
            raise NotImplementedError(
                "Dynamic vocabulary updates is not supported for streaming dataset."
            )

        if stage == "fit" or stage is None:
            # stage predict should be removed here.
            # This is a temporary fix for how the current predict dataloaders are implemented
            self.train_dataset = self.__class__.DATASET_FACTORY(
                **self.dataset_kwargs,
                split="train",
                shuffle=self.shuffle,
            )
            self.dev_dataset = self.__class__.DATASET_FACTORY(
                **self.dataset_kwargs,
                split="dev",
                shuffle=self.shuffle,
            )
        if stage == "validate":
            self.dev_dataset = self.__class__.DATASET_FACTORY(
                **self.dataset_kwargs,
                split="dev",
                shuffle=self.shuffle,
            )
        if stage == "test":
            self.test_dataset = self.__class__.DATASET_FACTORY(
                **self.dataset_kwargs,
                split="test",
                shuffle=self.shuffle,
            )
        if stage == "predict":
            if self.train_dataset is None or self.dev_dataset is None:
                self.setup("fit")
            self.predict_dataset = CombinedStreamingDataset(
                [self.train_dataset, self.dev_dataset]
            )

    def train_dataloader(self) -> StreamingDataLoader:
        """
        Returns a DataLoader for training.

        Returns
        -------
            DataLoader: DataLoader for training
        """
        self._train_dataloader = StreamingDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        return self._train_dataloader

    def val_dataloader(self) -> StreamingDataLoader:
        """
        Returns a DataLoader for validation.

        Returns
        -------
            DataLoader: DataLoader for validation
        """
        dataloader = StreamingDataLoader(
            self.dev_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.val_collate_fn(),
        )
        return dataloader

    def test_dataloader(self) -> StreamingDataLoader:
        """
        Returns a DataLoader for testing.

        Returns
        -------
            DataLoader: DataLoader for testing
        """
        return StreamingDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.val_collate_fn(),
        )
