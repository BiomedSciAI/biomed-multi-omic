import logging
from pathlib import Path

from bmfm_targets.datasets.dataset_transformer import DatasetTransformer
from bmfm_targets.datasets.scp1884 import SCP1884Dataset
from bmfm_targets.training.data_module import DataModule

logging.basicConfig(
    level=logging.INFO,
    filename="scp1884_datamodule.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SCP1884DataModule(DataModule):
    """PyTorch Lightning DataModule for SCP1884 dataset."""

    DATASET_FACTORY = SCP1884Dataset
    DATASET_TRANSFORMER_FACTORY = DatasetTransformer
    default_label_dict_path = Path(__file__).parent / "scp1884_all_labels.json"
