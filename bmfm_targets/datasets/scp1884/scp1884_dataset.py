import logging
from pathlib import Path

from bmfm_targets.datasets.base_rna_dataset import BaseRNAExpressionDataset

logging.basicConfig(
    level=logging.INFO,
    filename="scp1884_dataset.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SCP1884Dataset(BaseRNAExpressionDataset):
    """
    A PyTorch Dataset for scp1884 h5ad.
    https://singlecell.broadinstitute.org/single_cell/study/SCP1884/human-cd-atlas-study-between-colon-and-terminal-ileum.

    h5ad is generated downloaded tree from above URL using this notebook.
    https://github.ibm.com/A1KOSUGI/gsetests/blob/main/scp1884.ipynb

    Attributes
    ----------
            data_dir (str | Path): Path to the directory containing the data.
            split (str): Split to use. Must be one of train, dev, test.
            split_params (dict, optional): _description_. Defaults to default_split.
            transforms (list[dict] | None, optional): List of transforms to be applied to the datasets.
    Dataset Description
    ----------
            We obtained full dataset from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156728, and converted using functions written in gse156729_util.py
            Notebooks to generate pre-processed files using gse156729_util.py are stored in the separate repository.

    """

    URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156728"

    DATASET_NAME = "SCP1884"
    source_h5ad_file_name = "scp1884.h5ad"
    default_label_dict_path = Path(__file__).parent / f"{DATASET_NAME}_all_labels.json"
