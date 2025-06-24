import logging
from pathlib import Path

from bmfm_targets.datasets.base_rna_dataset import BaseRNAExpressionDataset
from bmfm_targets.datasets.dataset_transformer import DatasetTransformer
from bmfm_targets.training.data_module import DataModule

logging.basicConfig(
    level=logging.INFO,
    filename="myeloid_dataset.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MyeloidDataset(BaseRNAExpressionDataset):
    """
    A PyTorch Dataset for Myeloid h5ad files.

    Attributes
    ----------
    data_dir (str | Path): Path to the directory containing the data.
            split (str): Split to use. Must be one of train, dev, test.
            split_params (dict, optional): _description_. Defaults to default_split.
            transforms (list[dict] | None, optional): List of transforms to be applied to the datasets.

    Dataset Description
    ----------
            Here we use the split as described in scGPT:
            The myeloid dataset30 can be accessed from the Gene Expression
            Omnibus (GEO) database using accession number GSE154763. The
            dataset consists of nine different cancer types, but, for the purpose of
            training and evaluating the model, six cancer types were selected in
            the reference set for training, while three cancer types were used for
            the query set. The reference set contains myeloid cancer types UCEC,
            PAAD, THCA, LYM, cDC2 and kidney, while the query set contains MYE,
            OV-FTC and ESCA. The dataset was also randomly subsampled. The final
            cell counts were 9,748 in the reference set and 3,430 in the query set.
            Three thousand HVGs were selected during data processing.
            scGPT share their processed datasets at:
            https://figshare.com/articles/dataset/Processed_datasets_used_in_the_scGPT_foundation_model/24954519/1?file=43939560
    """

    DATASET_NAME = "myeloid"
    source_h5ad_file_name = "myeloid_scgpt_split.h5ad"
    default_label_dict_path = Path(__file__).parent / f"{DATASET_NAME}_all_labels.json"


class MyeloidDataModule(DataModule):
    """PyTorch Lightning DataModule for MS dataset."""

    DATASET_FACTORY = MyeloidDataset
    DATASET_TRANSFORMER_FACTORY = DatasetTransformer
