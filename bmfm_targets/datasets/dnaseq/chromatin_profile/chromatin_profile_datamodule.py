import logging
from pathlib import Path

from torch.utils.data import Dataset

from bmfm_targets.datasets.base_dna_dataset import BaseDNASeqDataset
from bmfm_targets.training.data_module import DNASeqDataModule

logging.basicConfig(
    level=logging.INFO,
    filename="snpdb_dataset.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from litdata import StreamingDataset

from bmfm_targets.datasets.data_conversion import get_user_serializer
from bmfm_targets.datasets.datasets_utils import (
    load_and_update_all_labels_json,
)
from bmfm_targets.tokenization import (
    MultiFieldInstance,
)
from bmfm_targets.training.streaming_datamodule import StreamingDataModule

serializer = get_user_serializer(list[str])
logging.basicConfig(
    level=logging.INFO,
    filename="snpdb_dataset.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class StreamingDNASeqChromatinProfileDataset(StreamingDataset):
    DATASET_NAME = "chromatin_profile"
    default_label_dict_path = (
        Path(__file__).parent / f"{DATASET_NAME.lower()}_all_labels.json"
    )
    default_label_columns = (
        ["dnase_" + str(i) for i in range(125)]
        + ["tf_" + str(i) for i in range(125, 815)]
        + ["histone_" + str(i) for i in range(815, 919)]
    )
    HAS_LABELS = True

    def __init__(
        self,
        processed_data_source: str | Path,
        split: str | None = None,
        shuffle: bool = False,
        drop_last: bool | None = None,
        label_dict_path: str | None = None,
        label_columns: list[str] | None = None,
        regression_label_columns: list[str] | None = None,
        seed: int = 42,
        max_cache_size: int | str = "1GB",
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
        if split not in ["train", "dev", "test", None]:
            raise ValueError(
                "Split must be one of train, dev, test; or None to get full dataset"
            )
        self.split = split
        self.label_dict_path = label_dict_path
        self.label_dict = None  # will be set only if dataset has labels
        self.label_columns = label_columns
        if self.label_columns is None:
            self.label_columns = self.default_label_columns
        self.regression_label_columns = regression_label_columns

        input_dir = Path(processed_data_source) / split
        super().__init__(
            input_dir=str(input_dir),
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
            max_cache_size=max_cache_size,
        )
        self.label_dict_path = self.default_label_dict_path
        self.label_dict = self.get_label_dict(self.label_dict_path)

    def get_label_dict(self, label_dict_path=None):
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
        return load_and_update_all_labels_json(None, label_dict_path, label_column_name)

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
        obj = super().__getitem__(idx)
        objs = [serializer.deserialize(x) for x in obj]
        dna_chunks = objs[0]
        labels = objs[1:]
        labels = [l[0] for l in labels]  # delisting the labels...

        if len(self.label_columns) < len(labels):
            label_indices = [
                self.default_label_columns.index(lc) for lc in self.label_columns
            ]
            labels = [labels[ind] for ind in label_indices]
        assert len(self.label_columns) == len(labels)
        metadata = dict(zip(self.label_columns, labels))
        return MultiFieldInstance(
            metadata=metadata,
            data={"dna_chunks": dna_chunks},
        )


class StreamingDNASeqChromatinProfileDataModule(StreamingDataModule):
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

    DATASET_FACTORY: type[StreamingDataset] = StreamingDNASeqChromatinProfileDataset

    # @property
    # def collate_fn(self):
    #     """
    #     Returns a collate function.

    #     Returns
    #     -------
    #         Callable: Collate function
    #     """
    #     return MultiFieldCollator(
    #         tokenizer=self.tokenizer,
    #         fields=self.fields,
    #         label_columns=self.label_columns,
    #         label_dict=self.label_dict,
    #         pad_to_multiple_of=self.pad_to_multiple_of,
    #         max_length=self.max_length,
    #         padding=self.padding,
    #         truncation=self.truncation,
    #         collation_strategy=self.collation_strategy,
    #     )


class DNASeqChromatinProfileDataset(BaseDNASeqDataset):
    DATASET_NAME = "chromatin_profile"
    default_label_dict_path = (
        Path(__file__).parent / f"{DATASET_NAME.lower()}_all_labels.json"
    )


class DNASeqChromatinProfileDataModule(DNASeqDataModule):
    DATASET_FACTORY: Dataset = DNASeqChromatinProfileDataset
