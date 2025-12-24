import logging
from pathlib import Path

from bmfm_targets.datasets.base_rna_dataset import BaseRNAExpressionDataset
from bmfm_targets.datasets.dataset_transformer import DatasetTransformer
from bmfm_targets.training.data_module import DataModule

logging.basicConfig(
    level=logging.INFO,
    filename="pancreascross_dataset.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PancreasCrossDataset(BaseRNAExpressionDataset):
    """
    A PyTorch Dataset for PancreasCross h5ad files.

    Attributes
    ----------
    data_dir (str | Path): Path to the directory containing the data.
            split (str): Split to use. Must be one of train, dev, test.
            split_params (dict, optional): _description_. Defaults to default_split.
            transforms (list[dict] | None, optional): List of transforms to be applied to the datasets.

    Dataset Description
    ----------
            PancreasCross Dataset is aiming at providing scRNA seq datasets described in Zhao, Hongyu, et al. "Evaluating the Utilities of Large Language Models in Single-cell Data Analysis." (2023). The PancreasCross dataset is an aggregation of scRNA seq datasets discussed several papers.

            Currently we are using a dataset referred as "hPancreas" in Chen, J., Xu, H., Tao, W., Chen, Z., Zhao, Y., & Han, J. D. J. (2023). Transformer for one stop interpretable cell type annotation. Nature Communications, 14(1), 223.

            This dataset is a concatenation of various datasets from studies who reported individual pancreatic cell transcriptomes in human. We follow the splits determined by Zhao, Hongyu, et al, as follows:

            Training dataset contains data (restricted to humans) from the following studies:

            Baron, Maayan, et al. "A single-cell transcriptomic map of the human and mouse pancreas reveals inter-and intra-cell population structure." Cell systems 3.4 (2016): 346-360.
            Muraro, Mauro J., et al. "A single-cell transcriptome atlas of the human pancreas." Cell systems 3.4 (2016): 385-394.

            Test dataset contains data from the following studies:

            Xin, Yurong, et al. "RNA sequencing of single human islet cells reveals type 2 diabetes genes." Cell metabolism 24.4 (2016): 608-615.
            Lawlor, Nathan, et al. "Single-cell transcriptomes identify human islet cell signatures and reveal cell-type–specific expression changes in type 2 diabetes." Genome research 27.2 (2017): 208-222.
            Segerstolpe, Åsa, et al. "Single-cell transcriptome profiling of human pancreatic islets in health and type 2 diabetes." Cell metabolism 24.4 (2016): 593-607.

            The dataset we used follows the same train/test split as Zhao, Hongyu, et al. It was downloaded from:
            https://figshare.com/articles/dataset/Pre-processed_data_for_benchmarking/24637044?file=43295985

            Of note, the authors renamed "stellate cells" to "PSC" (Pancreas stellate cell) for consistency. Aditionally, annotations such as "PP contaminated" or "Beta activated" have been removed due to their ambiguous meaning.

    """

    DATASET_NAME = "pancreascross"
    source_h5ad_file_name = "pancreascross_sceval_split.h5ad"
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
            "transform_name": "FilterCellsTransform",
            "transform_args": {"min_counts": 170},
        },
        {
            "transform_name": "FilterCellsTransform",
            "transform_args": {"max_counts": 850},
        },
        {
            "transform_name": "FilterCellsTransform",
            "transform_args": {"min_genes": 150},
        },
        {
            "transform_name": "FilterCellsTransform",
            "transform_args": {"max_genes": 1250},
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


class PancreasCrossDataModule(DataModule):
    """PyTorch Lightning DataModule for Pancreas Cross dataset."""

    DATASET_FACTORY = PancreasCrossDataset
    DATASET_TRANSFORMER_FACTORY = DatasetTransformer
