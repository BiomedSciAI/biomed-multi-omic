import logging
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

from bmfm_targets.datasets.datasets_utils import (
    load_and_update_all_labels_json,
)
from bmfm_targets.tokenization import (
    MultiFieldInstance,
)

logger = logging.getLogger(__name__)


class BaseDNASeqDataset(Dataset):
    default_label_dict_path: str | None = None
    DATASET_NAME: str | None = None

    def __init__(
        self,
        processed_data_source: str | Path,
        dataset_name: str | None = None,
        split: str | None = None,
        label_dict_path: str | None = None,
        label_columns: list[str] | None = None,
        regression_label_columns: list[str] | None = None,
        num_workers: int = 1,
    ) -> None:
        """
        Initializes the dataset. It processes the train, test and dev dataset and tokenize them and save them in a .pt file.

        Args:
        ----
        split (str): Split to use. Must be one of train, dev, test.
        num_workers (int): Number of workers to use for parallel processing.
        label_dict_path: str | None = None, Contining the name of label_dict file in json format.
        label_columns: list[str] | None = None, The categorical label names.
        regression_label_columns: list[str] | None = None, The regression label names.

        Assumptions:
        -----------
        On the data is that the first column should contain the dna_chunks with any header name followed by the labels specified in header either by
        label_columns or regression_label_columns or both.

        Raises:
        ------
            FileNotFoundError: If the data file does not exist.
            ValueError: If the split is not one of train, dev, test.
            NotImplementedError: If the pre-tokenizer is not implemented.

        """
        if dataset_name:
            self.DATASET_NAME = dataset_name
        if split not in ["train", "dev", "test", None]:
            raise ValueError(
                "Split must be one of train, dev, test; or None to get full dataset"
            )
        self.split = split
        self.label_dict_path = label_dict_path
        self.label_dict = None  # will be set only if dataset has labels
        self.label_columns = label_columns
        self.regression_label_columns = regression_label_columns
        if Path(processed_data_source).is_file():
            self.processed_file = Path(processed_data_source)
            self.processed_data = pd.read_csv(
                self.processed_file, sep=",", header=0, dtype=str
            )
        elif Path(processed_data_source).is_dir():
            if self.split is None:
                self.processed_files = [
                    Path(processed_data_source) / f"{split}.csv"
                    for split in ["train", "dev", "test"]
                ]
                split_data_frames = []
                for file in self.processed_files:
                    split_data_frames.append(
                        pd.read_csv(file, sep=",", header=0, dtype=str)
                    )
                self.processed_data = pd.concat(split_data_frames, ignore_index=True)
            else:
                self.processed_file = Path(processed_data_source) / f"{self.split}.csv"
                self.processed_data = pd.read_csv(
                    self.processed_file, sep=",", header=0, dtype=str
                )
        else:
            raise FileNotFoundError(
                f"Processed data source {processed_data_source} does not exist."
            )

        # Force the first column to be named "dna_chunks"
        self.processed_data = self.processed_data.rename(
            columns={self.processed_data.columns[0]: "dna_chunks"}
        )

        if self.label_columns or self.regression_label_columns:
            self.labels_requested = True
        else:
            self.labels_requested = False

        if self.labels_requested:
            if self.label_dict_path is None:
                self.label_dict_path = self.default_label_dict_path
            self.label_dict = self.get_label_dict(self.label_dict_path)

    def get_label_dict(self, label_dict_path):
        label_dict = {}
        if self.label_columns:
            label_dict.update(
                {
                    l: self.get_sub_label_dict(l, label_dict_path)
                    for l in self.label_columns
                }
            )
        if self.regression_label_columns:
            label_dict.update({l: {0: 0} for l in self.regression_label_columns})
        return label_dict

    def get_sub_label_dict(self, label_column_name, label_dict_path):
        return load_and_update_all_labels_json(
            self.processed_data, label_dict_path, label_column_name
        )

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns
        -------
            int: Number of samples.

        """
        return self.processed_data.shape[0]

    def __getitem__(self, idx: int) -> MultiFieldInstance:
        """
        Returns a single sample of chunk segments.

        Args:
        ----
            idx (int): Index of the chunk.

        Returns:
        -------
            MultiFieldInstance: A single sample of chunk.

        """
        metadata = self.get_sample_metadata(idx)

        return MultiFieldInstance(
            metadata=metadata,
            data={
                "dna_chunks": self.processed_data["dna_chunks"].iloc[idx],
            },
        )

    def get_sample_metadata(self, idx):
        metadata = {"seq_id": str(idx)}
        if self.label_columns:
            metadata.update(
                {
                    label_column: self.processed_data[label_column].iloc[idx]
                    for label_column in self.label_columns
                }
            )

        if self.regression_label_columns:
            metadata.update(
                {
                    regression_label_column: float(
                        self.processed_data[regression_label_column].iloc[idx]
                    )
                    for regression_label_column in self.regression_label_columns
                }
            )

        return metadata
