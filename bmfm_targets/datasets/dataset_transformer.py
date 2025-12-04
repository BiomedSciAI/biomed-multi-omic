import logging
import os

import numpy as np
from anndata import AnnData, read_h5ad
from scipy.sparse import csr_matrix

from bmfm_targets.datasets import datasets_utils
from bmfm_targets.transforms.compose import Compose
from bmfm_targets.transforms.sc_transforms import make_transform

logger = logging.getLogger(__name__)


default_transforms = [
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
    {"transform_name": "FilterCellsTransform", "transform_args": {"min_counts": 2}},
    {"transform_name": "FilterGenesTransform", "transform_args": {"min_counts": 1}},
]

default_perturbation_transforms = [
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
]


class DatasetTransformer:
    def __init__(
        self,
        source_h5ad_file_name: str,
        split_weights: dict | None = None,
        transforms: list[dict] | None = None,
        split_column_name: str | None = None,
        stratifying_label: str | None = None,
        random_state: int = 42,
    ) -> None:
        """
        Initializes the dataset.

        Args:
        ----
            split_weights: dict | None = None,
            split_balancing_label: str = "celltype",
            transforms (list[dict] | None, optional): List of transforms to be applied to the datasets.
            split_column_name: str | None = None, column name where split is stored. If None,
                a split column will be created. If `stratifying_label` is present, a stratified
                split will be created with name `split_stratified_{stratifying_label}`.
                If `stratifying_label` is not present, a random split will be created. Defaults to None.
            stratifying_label: str | None. Label to use for stratified split, must be a column
               in `source_h5ad_file_name` obs DataFrame.
            random_state: int. For reproducibility, defaults to 42.

        Raises:
        ------
            ValueError: If the split is not one of train, dev, test.
            FileNotFoundError: If the input data file (in h5ad format) does not exist.
            FileNotFoundError: If the processed data file does not exist and transform_datasets=False

        """
        if not os.path.exists(source_h5ad_file_name):
            raise FileNotFoundError(
                str(source_h5ad_file_name) + " input data file does not exist",
            )

        self.source_h5ad_file_name = source_h5ad_file_name

        if split_weights is None:
            split_weights = {"train": 0.8, "dev": 0.1, "test": 0.1}
        self.split_weights = split_weights

        if split_column_name is None:
            if stratifying_label:
                split_column_name = f"split_stratified_{stratifying_label}"
            else:
                split_column_name = "split_random"
        self.split_column_name = split_column_name
        self.stratifying_label = stratifying_label
        self.random_state = random_state
        if transforms is None:
            transforms = default_transforms
        if transforms:
            self.transforms = Compose([make_transform(**d) for d in transforms])
        else:  # transforms explicitly set to empty list []
            self.transforms = None

    def process_datasets(self) -> AnnData:
        """
        Processes the datasets by applying the pre-transforms and
        concatenating the datasets.

        Returns
        -------
            AnnData: Processed data.
        """
        raw_data = read_h5ad(self.source_h5ad_file_name)
        if not isinstance(raw_data.X, csr_matrix):
            if isinstance(raw_data.X, np.ndarray):
                raw_data.X = csr_matrix(raw_data.X)
            else:
                raw_data.X = raw_data.X.tocsr()

        if self.split_column_name not in raw_data.obs.columns:
            split_column = datasets_utils.get_split_column(
                raw_data.obs,
                self.split_weights,
                stratifying_label=self.stratifying_label,
                random_state=self.random_state,
            )
            raw_data.obs[self.split_column_name] = split_column

        if self.transforms is not None:
            processed_data = self.transforms(adata=raw_data)["adata"]
        else:
            processed_data = raw_data

        return processed_data


class PerturbationDatasetTransformer:
    def __init__(
        self,
        tokenizer,
        source_h5ad_file_names: list[str],
        split_weights: dict | None = None,
        transforms: list[dict] | None = None,
        split_column_name: str | None = None,
        group_column_name: str = "group",
        stratifying_label: str | None = "perturbation",
        perturbation_column_name: str | None = "perturbation",
        stratification_type: str = "single_only",
        random_state: int = 42,
    ) -> None:
        """
        Initializes the dataset.

        Args:
        ----
            source_h5ad_file_names: list[str]. List of h5ad files to be concatenated.
            split_weights: dict | None = None,
            transforms (list[dict] | None, optional): List of transforms to be applied to the datasets.
            split_column_name: str | None = None, column name where split is stored. If None,
                a split column will be created. If `stratifying_label` is present, a stratified
                split will be created with name `split_stratified_{stratifying_label}`.
                If `stratifying_label` is not present, a random split will be created. Defaults to None.
            stratifying_label: str | None. Label to use for stratified split, must be a column
               in `source_h5ad_file_name` obs DataFrame.
            random_state: int. For reproducibility, defaults to 42.

        Raises:
        ------
            ValueError: If the split is not one of train, dev, test.
            FileNotFoundError: If the input data file (in h5ad format) does not exist.
            FileNotFoundError: If the processed data file does not exist and transform_datasets=False

        """
        for source_h5ad_file_name in source_h5ad_file_names:
            if not os.path.exists(source_h5ad_file_name):
                raise FileNotFoundError(
                    str(source_h5ad_file_name) + "input data file does not exist",
                )

        self.source_h5ad_file_names = source_h5ad_file_names
        self.stratification_type = stratification_type
        if split_weights is None:
            split_weights = {"train": 0.8, "dev": 0.1, "test": 0.1}
        self.split_weights = split_weights

        if split_column_name is None:
            split_column_name = f"split_stratified_{stratifying_label}"

        self.split_column_name = split_column_name
        self.group_column_name = group_column_name
        self.stratifying_label = stratifying_label
        self.perturbation_column_name = perturbation_column_name
        self.random_state = random_state
        if transforms is None:
            transforms = default_perturbation_transforms
        if transforms:
            self.transforms = Compose([make_transform(**d) for d in transforms])
        else:  # transforms explicitly set to empty list []
            self.transforms = None
        self.tokenizer = tokenizer

    def _clean_dataset(self, ann_data: AnnData):
        """
        Cleans perturbation dataset.


        Args:
        ----
            ann_data (sc.AnnData): AnnData object containing perturbation data
        """
        # remove all perturbations that are not in the tokenizer vocabulary but keep the control
        vocab_genes = list(self.tokenizer.get_field_vocab("genes").keys()) + ["Control"]
        ann_data = ann_data[
            ann_data.obs[self.perturbation_column_name]
            .str.split("_")
            .apply(lambda x: all(gene in vocab_genes for gene in x))
        ]

        # remove all the perturbation that are not in the genes of the dataset. Not sure why this can happen.
        gene_list = list(ann_data.var_names) + ["Control"]
        ann_data = ann_data[
            ann_data.obs[self.perturbation_column_name]
            .str.split("_")
            .apply(lambda x: all(gene in gene_list for gene in x))
        ]

        ann_data.obs[self.perturbation_column_name] = ann_data.obs[
            self.perturbation_column_name
        ].apply(lambda x: "_".join(sorted(x.split("_"))))

        return ann_data

    def process_datasets(self) -> AnnData:
        """
        Processes the datasets by applying the pre-transforms and
        concatenating the datasets.

        Returns
        -------
            AnnData: Processed data.
        """
        cleaned_datasets = []
        for source_h5ad_file_name in self.source_h5ad_file_names:
            raw_data = read_h5ad(source_h5ad_file_name)
            if not isinstance(raw_data.X, csr_matrix):
                if isinstance(raw_data.X, np.ndarray):
                    raw_data.X = csr_matrix(raw_data.X)
                else:
                    raw_data.X = raw_data.X.tocsr()
            cleaned_datasets.append(self._clean_dataset(raw_data))

        merged_data = AnnData.concatenate(*cleaned_datasets)

        if self.split_column_name not in raw_data.obs.columns:
            if self.stratification_type == "simulation":
                (
                    split_column,
                    group_column,
                ) = datasets_utils.get_perturbation_split_column(
                    merged_data.obs,
                    self.split_weights,
                    perturbation_column_name=self.perturbation_column_name,
                    stratification_type=self.stratification_type,
                    random_state=self.random_state,
                )

                merged_data.obs[self.split_column_name] = split_column
                merged_data.obs[self.group_column_name] = group_column
            else:
                split_column = datasets_utils.get_perturbation_split_column(
                    merged_data.obs,
                    self.split_weights,
                    perturbation_column_name=self.perturbation_column_name,
                    stratification_type=self.stratification_type,
                    random_state=self.random_state,
                )
                merged_data.obs[self.split_column_name] = split_column

        if self.transforms is not None:
            processed_data = self.transforms(adata=merged_data)["adata"]
        else:
            processed_data = merged_data
        self.add_de_genes_to_uns(processed_data)
        return processed_data

    def add_de_genes_to_uns(self, processed_data):
        datasets_utils.add_gene_ranking(
            processed_data,
            perturbation_column_name=self.perturbation_column_name,
            control_name="Control",
            n_genes=50,
        )
        datasets_utils.add_non_zero_non_dropout_de_gene_ranking(
            processed_data, self.perturbation_column_name
        )


class GearsPerturbationDatasetTransformer(PerturbationDatasetTransformer):
    def _clean_dataset(self, ann_data: AnnData):
        """
        Cleans perturbation dataset.


        Args:
        ----
            ann_data (sc.AnnData): AnnData object containing perturbation data
        """
        ann_data = ann_data[~ann_data.obs[self.perturbation_column_name].isna()]
        ann_data.obs[self.perturbation_column_name] = (
            ann_data.obs[self.perturbation_column_name]
            .str.replace("+ctrl", "")
            .str.replace("ctrl+", "")
            .str.replace("+", "_")
            .str.replace("ctrl", "Control")
        )

        ann_data.var_names = ann_data.var["gene_name"].values

        ann_data = super()._clean_dataset(ann_data)
        return ann_data


class ScperturbPerturbationDatasetTransformer(PerturbationDatasetTransformer):
    def _clean_dataset(self, ann_data: AnnData):
        """
        Cleans perturbation dataset.


        Args:
        ----
            ann_data (sc.AnnData): AnnData object containing perturbation data
        """
        ann_data = ann_data[~ann_data.obs[self.perturbation_column_name].isna()]

        ann_data.obs[self.perturbation_column_name] = (
            ann_data.obs[self.perturbation_column_name].str.rsplit("_", n=1).str[0]
        )

        ann_data.obs[self.perturbation_column_name] = (
            ann_data.obs[self.perturbation_column_name]
            .replace(r".+\(mod\)", "Control", regex=True)
            .replace(r".+ctrl", "Control", regex=True)
        )
        ann_data.obs[self.perturbation_column_name] = ann_data.obs[
            self.perturbation_column_name
        ].str.replace("_only", "")
        # remove all the rows where clean_perturbation column has "*" value
        ann_data = ann_data[ann_data.obs[self.perturbation_column_name] != "*"]

        ann_data = super()._clean_dataset(ann_data)
        return ann_data


class ReploglePerturbationDatasetTransformer(PerturbationDatasetTransformer):
    def _clean_dataset(self, ann_data: AnnData):
        r"""
        Cleans Replogle ScGPT processed perturbation dataset:
        1) rename control perturbation to "Control"
        2) make var_name = perturbation name.

        Assumes split columns was already added (column scgpt_split).

        For the pre-processing see datasets\perturbation]replogle.ipynb

        Args:
        ----
            ann_data (sc.AnnData): AnnData object containing perturbation data
        """
        # rename control column
        ann_data.obs[self.perturbation_column_name] = ann_data.obs[
            self.perturbation_column_name
        ].replace("non-targeting", "Control")

        # set var_names to be gene_names
        ann_data.var["ensemble_id"] = ann_data.var.index
        ann_data.var["gene_name"] = ann_data.var["gene_name"].astype(str)
        ann_data.var = ann_data.var.set_index("gene_name")
        # print("Original var_names:", len(ann_data.var_names))

        ann_data.var_names_make_unique()
        # print("Original var_names:", len(ann_data.var_names))

        ann_data = super()._clean_dataset(ann_data)
        return ann_data


class VCCPerturbationDatasetTransformer(PerturbationDatasetTransformer):
    def _clean_dataset(self, ann_data: AnnData):
        """
        Cleans Replogle ScGPT processed perturbation dataset:
        1) rename control perturbation to "Control".

        Assumes split columns was already added.
        For pre-processing see vcc_preprocess_and_descstats.ipynb

        Args:
        ----
            ann_data (sc.AnnData): AnnData object containing perturbation data
        """
        # rename control column
        ann_data.obs[self.perturbation_column_name] = ann_data.obs[
            self.perturbation_column_name
        ].replace("non-targeting", "Control")

        ann_data = super()._clean_dataset(ann_data)
        return ann_data
