import logging

from bmfm_targets.datasets import (
    GearsPerturbationDatasetTransformer,
    ScperturbPerturbationDatasetTransformer,
)
from bmfm_targets.datasets.base_perturbation_dataset import BasePerturbationDataset
from bmfm_targets.datasets.dataset_transformer import (
    ReploglePerturbationDatasetTransformer,
    VCCPerturbationDatasetTransformer,
)
from bmfm_targets.training.data_module import PerturbationDataModule

logging.basicConfig(
    level=logging.INFO,
    filename="perturbation_dataset.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ScperturbDataset(BasePerturbationDataset):
    DATASET_NAME = "adamson_weissman"
    source_h5ad_file_names = ["AdamsonWeissman2016_GSM2406675_10X001.h5ad"]


class ScperturbDataModule(PerturbationDataModule):
    """PyTorch Lightning DataModule for perturbation datasets."""

    DATASET_FACTORY = ScperturbDataset
    DATASET_TRANSFORMER_FACTORY = ScperturbPerturbationDatasetTransformer


class GearsDataset(BasePerturbationDataset):
    DATASET_NAME = "norman"
    source_h5ad_file_names = [
        "/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/finetune/Perturbation/Norman_GSE133344/preprocessed_GEARS/perturb_processed.h5ad"
    ]


class GearsDataModule(PerturbationDataModule):
    """PyTorch Lightning DataModule for perturbation datasets."""

    DATASET_FACTORY = GearsDataset
    DATASET_TRANSFORMER_FACTORY = GearsPerturbationDatasetTransformer


class ReplogleDataset(BasePerturbationDataset):
    DATASET_NAME = "replogle"
    source_h5ad_file_names = [
        "/$BMFM_TARGETS_REPLOGLE_DATA/k562_gwps_raw_singlecell_01_processed_with_scgpt_split.h5ad"
    ]


class ReplogleDataModule(PerturbationDataModule):
    """PyTorch Lightning DataModule for perturbation datasets."""

    DATASET_FACTORY = ReplogleDataset
    DATASET_TRANSFORMER_FACTORY = ReploglePerturbationDatasetTransformer


class VCCDataset(BasePerturbationDataset):
    DATASET_NAME = "vcc"
    source_h5ad_file_names = ["train_and_test.h5ad"]


class VCCDataModule(PerturbationDataModule):
    """PyTorch Lightning DataModule for perturbation datasets."""

    DATASET_FACTORY = VCCDataset
    DATASET_TRANSFORMER_FACTORY = VCCPerturbationDatasetTransformer
