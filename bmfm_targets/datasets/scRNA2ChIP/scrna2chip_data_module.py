import logging

from bmfm_targets.datasets import DatasetTransformer
from bmfm_targets.datasets.base_scrna2chip_dataset import BasescRNA2ChIPDataset
from bmfm_targets.training.data_module import scRNA2ChIPDataModule

logger = logging.getLogger(__name__)


class ConcatDataset(BasescRNA2ChIPDataset):
    DATASET_NAME = "concat_chip_subsampled_scrna"


class ConcatDataModule(scRNA2ChIPDataModule):
    """Lightning DataModule for the concatenated scRNA + ChIP translation dataset."""

    DATASET_FACTORY = ConcatDataset
    DATASET_TRANSFORMER_FACTORY = DatasetTransformer
