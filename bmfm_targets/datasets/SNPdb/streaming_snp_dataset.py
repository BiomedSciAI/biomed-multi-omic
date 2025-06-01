import json
import logging
from pathlib import Path

from litdata import StreamingDataset
from litdata.streaming.combined import CombinedStreamingDataset

from bmfm_targets.datasets.data_conversion import get_user_serializer
from bmfm_targets.tokenization import (
    MultiFieldCollator,
    MultiFieldInstance,
)
from bmfm_targets.training.masking import Masker
from bmfm_targets.training.streaming_datamodule import StreamingDataModule

serializer = get_user_serializer(list[str])
logger = logging.getLogger(__name__)
import os


class StreamingSNPdbDataset(StreamingDataset):
    DATASET_NAME = "SNPdb"

    def __init__(
        self,
        input_dir: str,
        split: str,
        shuffle: bool = False,
        drop_last: bool | None = None,
        seed: int = 42,
        max_cache_size: int | str = "1GB",
    ):
        input_dir = os.path.join(input_dir, split)
        super().__init__(
            input_dir=input_dir,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
            max_cache_size=max_cache_size,
        )

    def __getitem__(self, idx: int) -> MultiFieldInstance:
        obj = super().__getitem__(idx)
        return MultiFieldInstance(
            data={
                "dna_chunks": serializer.deserialize(obj),
            },
        )


class StreamingSNPdbDataModule(StreamingDataModule):
    DATASET_FACTORY: StreamingDataset = StreamingSNPdbDataset


class StreamingHiCDataset(StreamingDataset):
    DATASET_NAME = "HiC"

    def __init__(
        self,
        input_dir: str,
        split: str,
        shuffle: bool = False,
        drop_last: bool | None = None,
        label_columns: list[str] | None = None,
        regression_label_columns: list[str] | None = None,
        masker: Masker | None = None,
        seed: int = 42,
        max_cache_size: int | str = "1GB",
    ):
        input_dir = os.path.join(input_dir, split)
        super().__init__(
            input_dir=input_dir,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
            max_cache_size=max_cache_size,
        )
        self.label_columns = label_columns
        self.regression_label_columns = regression_label_columns
        self.masker = masker
        self.label_dict_path = Path(__file__).parent / "hic_binary_labels.json"
        self.label_dict = self.get_label_dict(self.label_dict_path)

    def load_all_labels_json(self, output_path, label_column_name):
        if Path(output_path).exists():
            with open(output_path) as json_file:
                all_labels_dict = json.load(json_file)
                for item in all_labels_dict:
                    if label_column_name == item["label_name"]:
                        return item["label_values"]

    def get_sub_label_dict(self, label_column_name, label_dict_path):
        return self.load_all_labels_json(label_dict_path, label_column_name)

    def get_label_dict(self, label_dict_path=None):
        if label_dict_path is None:
            categorical_labels = {}
        else:
            categorical_labels = {
                l: self.get_sub_label_dict(l, label_dict_path)
                for l in self.label_columns
            }
        if self.regression_label_columns is None:
            regression_labels = {}
        else:
            regression_labels = {l: {0: 0} for l in self.regression_label_columns}
        return {**categorical_labels, **regression_labels}

    def __getitem__(self, idx: int) -> tuple[MultiFieldInstance, MultiFieldInstance]:
        obj = super().__getitem__(idx)
        dna_chunk1, dna_chunk2, hic_contact = (serializer.deserialize(x) for x in obj)
        ## original hic_contact is deserialized into a list ['1', '.', '5', '5'] if serializer input is ['1.55'] with 1 element - bug?
        # hic_contact = float("".join(hic_contact))
        ## TODO work for both regression and classification
        hic_contact = str("".join(hic_contact))
        return (
            MultiFieldInstance(
                data={
                    "dna_chunks": dna_chunk1,
                    "dna_chunk_ids": ["0" for _ in range(len(dna_chunk1))],
                },
                metadata={"hic_contact": hic_contact},
            ),
            MultiFieldInstance(
                data={
                    "dna_chunks": dna_chunk2,
                    "dna_chunk_ids": ["1" for _ in range(len(dna_chunk2))],
                },
            ),
        )


class StreamingHiCDataModule(StreamingDataModule):
    """
    PyTorch Lightning DataModule for DNA sequence datasets.

    Attributes
    ----------
        dataset_kwargs: Keyword arguments for dataset
        tokenizer (MultiFieldTokenizer): Tokenizer to use
        fields (list[FieldInfo]): List of FieldInfo objects
        batch_size (int, optional): Batch size. Defaults to 32.
        num_workers (int, optional): Number of workers for DataLoader. Defaults to 1.
        max_length (int, optional): Maximum length of input sequences. Defaults to 512.
        padding (PaddingStrategy, optional): Padding strategy. Defaults to PaddingStrategy.MAX_LENGTH. Available options: PaddingStrategy.MAX_LENGTH, PaddingStrategy.LONGEST.
        truncation (TruncationStrategy, optional): Truncation strategy. Defaults to TruncationStrategy.ONLY_FIRST. Available options: TruncationStrategy.ONLY_FIRST, TruncationStrategy.ONLY_SECOND, TruncationStrategy.LONGEST_FIRST, TruncationStrategy.LONGEST_SECOND.
        save_state (bool): flag to save state of the dataloader into checkpoint
        checkpoint (str: None): File name of checkpoint to laod the state from
    """

    state_entry_name = "datamodule_state"
    DATASET_FACTORY: type[StreamingDataset] = StreamingHiCDataset

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
            masker=self.masker,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            collation_strategy=self.collation_strategy,
        )


class StreamingInsulationDataset(StreamingDataset):
    DATASET_NAME = "Insulation"

    def __init__(
        self,
        input_dir: str,
        split: str,
        shuffle: bool = False,
        drop_last: bool | None = None,
        label_columns: list[str] | None = None,
        regression_label_columns: list[str] | None = None,
        masker: Masker | None = None,
        seed: int = 42,
        max_cache_size: int | str = "1GB",
    ):
        input_dir = os.path.join(input_dir, split)
        super().__init__(
            input_dir=input_dir,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
            max_cache_size=max_cache_size,
        )
        self.label_columns = label_columns
        self.regression_label_columns = regression_label_columns
        self.masker = masker
        self.label_dict_path = None
        self.label_dict = self.get_label_dict(self.label_dict_path)

    def load_all_labels_json(self, output_path, label_column_name):
        if Path(output_path).exists():
            with open(output_path) as json_file:
                all_labels_dict = json.load(json_file)
                for item in all_labels_dict:
                    if label_column_name == item["label_name"]:
                        return item["label_values"]

    def get_sub_label_dict(self, label_column_name, label_dict_path):
        return self.load_all_labels_json(label_dict_path, label_column_name)

    def get_label_dict(self, label_dict_path=None):
        if label_dict_path is None:
            categorical_labels = {}
        else:
            categorical_labels = {
                l: self.get_sub_label_dict(l, label_dict_path)
                for l in self.label_columns
            }
        if self.regression_label_columns is None:
            regression_labels = {}
        else:
            regression_labels = {l: {0: 0} for l in self.regression_label_columns}
        return {**categorical_labels, **regression_labels}

    def __getitem__(self, idx: int) -> MultiFieldInstance:
        obj = super().__getitem__(idx)
        dna_chunks, insulation = (serializer.deserialize(x) for x in obj)
        ## Here regression labels need to be float (classification labels are str!!)
        ## TODO should update multifield collator convert it to float
        insulation = float("".join(insulation))
        return MultiFieldInstance(
            data={"dna_chunks": dna_chunks},
            metadata={"insulation": insulation},
        )


class StreamingInsulationDataModule(StreamingDataModule):
    """
    PyTorch Lightning DataModule for DNA sequence datasets.

    Attributes
    ----------
        dataset_kwargs: Keyword arguments for dataset
        tokenizer (MultiFieldTokenizer): Tokenizer to use
        fields (list[FieldInfo]): List of FieldInfo objects
        batch_size (int, optional): Batch size. Defaults to 32.
        num_workers (int, optional): Number of workers for DataLoader. Defaults to 1.
        max_length (int, optional): Maximum length of input sequences. Defaults to 512.
        padding (PaddingStrategy, optional): Padding strategy. Defaults to PaddingStrategy.MAX_LENGTH. Available options: PaddingStrategy.MAX_LENGTH, PaddingStrategy.LONGEST.
        truncation (TruncationStrategy, optional): Truncation strategy. Defaults to TruncationStrategy.ONLY_FIRST. Available options: TruncationStrategy.ONLY_FIRST, TruncationStrategy.ONLY_SECOND, TruncationStrategy.LONGEST_FIRST, TruncationStrategy.LONGEST_SECOND.
        save_state (bool): flag to save state of the dataloader into checkpoint
        checkpoint (str: None): File name of checkpoint to laod the state from
    """

    state_entry_name = "datamodule_state"
    DATASET_FACTORY: type[StreamingDataset] = StreamingInsulationDataset

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
            masker=self.masker,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            collation_strategy=self.collation_strategy,
        )


class StreamingTokenLabelDataset(StreamingDataset):
    DATASET_NAME = "TokenLabel"

    def __init__(
        self,
        input_dir: str,
        split: str,
        shuffle: bool = False,
        drop_last: bool | None = None,
        seed: int = 42,
        max_cache_size: int | str = "1GB",
    ):
        input_dir = os.path.join(input_dir, split)
        super().__init__(
            input_dir=input_dir,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
            max_cache_size=max_cache_size,
        )

    def __getitem__(self, idx: int) -> MultiFieldInstance:
        obj = super().__getitem__(idx)
        dna_chunks, token_labels = (serializer.deserialize(x) for x in obj)
        return MultiFieldInstance(
            data={
                "dna_chunks": dna_chunks,
                "token_labels": token_labels,
            },
        )


class StreamingTokenLabelDataModule(StreamingDataModule):
    DATASET_FACTORY: StreamingDataset = StreamingTokenLabelDataset


class CombinedStreamingSNPdbDataset(CombinedStreamingDataset):
    def __init__(
        self,
        input_dirs: list[str],
        split: str,
        **kwargs,
    ):
        self.input_dirs = [os.path.join(x, split) for x in input_dirs]
        self.split = split
        datasets = [
            StreamingSNPdbDataset(input_dir=x, split=self.split) for x in input_dirs
        ]
        super().__init__(
            datasets=datasets,
            **kwargs,
        )

    # TODO it has __iter__ do we still need __getitem__ ?
    # def __getitem__(self, idx: int) -> MultiFieldInstance:
    #    ## CombinedStreamingDataset doesn't define __getitem__
    #    return super().__iter__()._get_sample(idx)


class CombinedStreamingSNPdbDataModule(StreamingDataModule):
    DATASET_FACTORY: StreamingDataset = CombinedStreamingSNPdbDataset

    ## redefine setup since it's different in CombinedStreamingDataset
    ## TODO: CombinedStreamingDataset support shuffle during setup?
    def setup(self, stage: str = None) -> None:
        if any(field.vocab_update_strategy == "dynamic" for field in self.fields):
            raise NotImplementedError(
                "Dynamic vocabulary updates is not supported for streaming dataset."
            )
        if stage == "fit" or stage is None:
            # stage predict should be removed here.
            # This is a temporary fix for how the current predict dataloaders are implemented
            self.train_dataset = self.DATASET_FACTORY(
                **self.dataset_kwargs,
                split="train",
            )
            self.dev_dataset = self.DATASET_FACTORY(
                **self.dataset_kwargs,
                split="dev",
            )
        if stage == "validate":
            self.dev_dataset = self.DATASET_FACTORY(
                **self.dataset_kwargs,
                split="dev",
            )
        if stage == "test":
            self.test_dataset = self.DATASET_FACTORY(
                **self.dataset_kwargs,
                split="test",
            )
