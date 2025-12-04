import logging
from pathlib import Path

from bmfm_targets.datasets.base_rna_dataset import BaseRNAExpressionDataset
from bmfm_targets.datasets.dataset_transformer import DatasetTransformer
from bmfm_targets.training.data_module import DataModule

logging.basicConfig(
    level=logging.INFO,
    filename="hbones_dataset.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class HBonesDataset(BaseRNAExpressionDataset):
    """
    A PyTorch Dataset for hBones h5ad files.

    Attributes
    ----------
    data_dir (str | Path): Path to the directory containing the data.
            split (str): Split to use. Must be one of train, dev, test.
            split_params (dict, optional): _description_. Defaults to default_split.
            transforms (list[dict] | None, optional): List of transforms to be applied to the datasets.

    Dataset Description
    ----------
            HBones Dataset (GSE152805) provides scRNA seq data described in Chou, Ching-Heng, et al. "Synovial cell cross-talk with cartilage plays a major role in the pathogenesis of osteoarthritis." Scientific Reports 10.1 (2020): 10868

            The study aims at performing single-cell transcriptomic analysis on osteoarthritis knee joint tissues to systematically identify cell types and states within human osteoarthritic synovium and matched cartilage. The study also aims at determining regulators of these articular chondrocyte phenotypes.
            https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE152805

            Here, we use the train/test split used in Chen, J., Xu, H., Tao, W., Chen, Z., Zhao, Y., & Han, J. D. J. (2023). Transformer for one stop interpretable cell type annotation. Nature Communications, 14(1), 223.
            In this split, the hBone datasets use healthy samples as training data and predict disease samples.
    """

    DATASET_NAME = "hBones"
    source_h5ad_file_name = "hbones.h5ad"
    DEFAULT_TRANSFORMS = [
        {
            "transform_name": "RenameGenesTransform",
            "transform_args": {
                "gene_map": None,
            },
        },
        {
            "transform_name": "KeepGenesTransform",
            "transform_args": {"genes_to_keep": None},
        },
        {
            "transform_name": "QcMetricsTransform",
            "transform_args": {"pct_counts_mt": 8},
        },
        {
            "transform_name": "FilterCellsTransform",
            "transform_args": {"min_counts": 450},
        },
        {
            "transform_name": "FilterCellsTransform",
            "transform_args": {"max_counts": 700},
        },
        {
            "transform_name": "FilterCellsTransform",
            "transform_args": {"min_genes": 300},
        },
        {
            "transform_name": "FilterCellsTransform",
            "transform_args": {"max_genes": 800},
        },
        {
            "transform_name": "NormalizeTotalTransform",
            "transform_args": {
                "exclude_highly_expressed": False,
                "max_fraction": 0.05,
                "target_sum": 10000.0,
            },
        },
        {
            "transform_name": "LogTransform",
            "transform_args": {"base": 2, "chunk_size": None, "chunked": None},
        },
        {
            "transform_name": "BinTransform",
            "transform_args": {"num_bins": 10, "binning_method": "int_cast"},
        },
    ]
    default_label_dict_path = Path(__file__).parent / f"{DATASET_NAME}_all_labels.json"


class HBonesDataModule(DataModule):
    """PyTorch Lightning DataModule for hBones dataset."""

    DATASET_FACTORY = HBonesDataset
    DATASET_TRANSFORMER_FACTORY = DatasetTransformer
