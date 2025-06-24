import logging
from pathlib import Path

from bmfm_targets.datasets.base_rna_dataset import BaseRNAExpressionDataset
from bmfm_targets.datasets.dataset_transformer import DatasetTransformer
from bmfm_targets.training.data_module import DataModule

logging.basicConfig(
    level=logging.INFO,
    filename="multiple_sclerosis_dataset.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MultipleSclerosisDataset(BaseRNAExpressionDataset):
    """
    A PyTorch Dataset for Multiple Sclerosis (MS) h5ad files.

    Attributes
    ----------
    data_dir (str | Path): Path to the directory containing the data.
            split (str): Split to use. Must be one of train, dev, test.
            split_params (dict, optional): _description_. Defaults to default_split.
            transforms (list[dict] | None, optional): List of transforms to be applied to the datasets.

    Dataset Description
    ----------
            The MS dataset was accessed from EMBL-EBI (https://www.ebi.ac.uk/gxa/sc/experiments/E-HCAD-35). Nine healthy control samples and 12 MS samples are included in the dataset.
            Here we use the split as described in scGPT:
            Reference set includes the Nine healthy control samples. The MS samples are included in the query set for evaluation.
            This setting serves as an example of out-of-distribution data.
            Three cell types are excluded: B cells, T cells and oligodendrocyte B cells, which only existed in the query dataset.
            The final cell counts were 7,844 in the training reference set and 13,468 in the query set.
            The provided cell type labels from the original publication were used as ground truth labels for evaluation.
            The data-processing protocol involved selecting HVGs to retain 3,000 genes. The data is already log-normalized.
    """

    DATASET_NAME = "multiple_sclerosis"
    source_h5ad_file_name = "multiple_sclerosis_scgpt_split.h5ad"

    default_label_dict_path = Path(__file__).parent / f"{DATASET_NAME}_all_labels.json"


class MultipleSclerosisDataModule(DataModule):
    """PyTorch Lightning DataModule for MS dataset."""

    DATASET_FACTORY = MultipleSclerosisDataset
    DATASET_TRANSFORMER_FACTORY = DatasetTransformer
