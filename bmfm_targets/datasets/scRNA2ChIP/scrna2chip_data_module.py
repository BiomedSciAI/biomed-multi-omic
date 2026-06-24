import logging

from bmfm_targets.datasets import (
    DatasetTransformer
)
from bmfm_targets.datasets.base_scrna2chip_dataset import BasescRNA2ChIPDataset

from bmfm_targets.training.data_module import scRNA2ChIPDataModule

logging.basicConfig(
    level=logging.INFO,
    filename="scrna2chip_dataset.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ConcatDataset(BasescRNA2ChIPDataset):
    DATASET_NAME = "concat_chip_subsampled_scrna"
    # source_h5ad_file_names = ["/Users/elafallik/Library/CloudStorage/GoogleDrive-ela.fallik@mail.huji.ac.il/Shared drives/Friedman Lab Shared Drive/ElaFallik/IBM/biomed-multi-omic/retrained_models/perturb_like_concat/concat_chip_subsampled_scrna_X_cv0_train_dev_by_tissues.h5ad"]


class ConcatDataModule(scRNA2ChIPDataModule):
    """PyTorch Lightning DataModule for perturbation datasets."""
    DATASET_FACTORY = ConcatDataset
    DATASET_TRANSFORMER_FACTORY = DatasetTransformer

