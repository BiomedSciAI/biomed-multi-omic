import abc
import random
from typing import Literal

import numpy as np
import scipy.sparse as ss
from anndata import AnnData, read_h5ad

try:
    # anndata >= 0.11
    from anndata.abc import CSRDataset as SparseDataset
except ImportError:
    # anndata >= 0.10
    from anndata.experimental import CSRDataset as SparseDataset


from torch.utils.data import Dataset

from bmfm_targets.config.dataset_config import ExposeZerosMode
from bmfm_targets.datasets.base_rna_dataset import BaseRNAExpressionDataset, logger
from bmfm_targets.datasets.datasets_utils import (
    guess_if_raw,
    make_group_means,
    random_subsampling,
)
from bmfm_targets.tokenization import MultiFieldInstance


class BasePerturbationDataset(Dataset, abc.ABC):
    """A PyTorch Dataset for the  perturbation dataset."""

    source_h5ad_file_name: list[str] = ...

    def __init__(
        self,
        processed_data_source: AnnData | str = "processed.h5ad",
        split: str | None = None,
        split_column_name: str | None = None,
        perturbation_column_name: str | None = None,
        stratifying_label: str | None = None,
        backed: Literal["r", "r+"] | None = None,
        limit_samples: int | None = None,
        limit_samples_shuffle: bool = True,
        limit_genes: list[str] | None = None,
        filter_query: str | None = None,
        expose_zeros: ExposeZerosMode = ExposeZerosMode.LABEL_NONZEROS,
        label_columns: list[str] | None = None,
        regression_label_columns: list[str] | None = None,
        sort_genes_var: str | None = None,
    ) -> None:
        """
        Initializes the dataset.

        Args:
        ----
            processed_data_source (AnnData | str): either an AnnData object that has been processed or the path to such an h5ad file.
            split (str): Split to use. Must be one of train, dev, test or None to get all data.
            split_column_name (str): The column name where split is stored. If None, all of the data will be used as test.
            perturbation_column_name (str): The column name in the AnnData object that contains the perturbation information.
            stratifying_label (str): The column name in the AnnData object that contains the stratifying label.
            backed (Literal["r", "r+"] | None): Whether to read the data in backed mode. If None, the data will be read in memory.
            limit_samples (int | None): The number of samples to limit the dataset to.
            limit_samples_shuffle (bool): Whether to shuffle the samples before limiting the dataset.
            limit_genes (list[str] | None): The list of genes to limit the dataset to.
            filter_query (str | None): The query to filter the data. If None, no filtering will be applied.
            expose_zeros (str|None): How to handle exposing zeros
            sort_genes_var (str | None) : A var column name according to which genes will be sorted per sample

        Raises:
        ------
            ValueError: If the split is not one of train, dev, test.
            ValueError: If the data is not sparse
        """
        if split not in ["train", "dev", "test", None]:
            raise ValueError("The split must be one of train, dev, test or None.")

        self.split = split
        self.backed = backed
        self.filter_query = filter_query
        self.limit_genes = limit_genes
        self.expose_zeros = ExposeZerosMode(expose_zeros)
        self.sort_genes_var = sort_genes_var
        self.processed_data_source = processed_data_source
        if perturbation_column_name is None:
            raise ValueError("A `perturbation_column_name` must be defined")
        self.perturbation_column_name = perturbation_column_name
        if split_column_name is None:
            split_column_name = f"split_stratified_{stratifying_label}"
        self.split_column_name = split_column_name

        if isinstance(processed_data_source, AnnData):
            self.processed_data = processed_data_source.copy()
        else:
            self.processed_data = read_h5ad(processed_data_source, backed=self.backed)

        if not (
            ss.issparse(self.processed_data.X)
            or isinstance(self.processed_data.X, SparseDataset)
        ):
            raise ValueError("Data is not sparse")
        if guess_if_raw(self.processed_data.X.data):
            exp_before_mean = False
        else:
            exp_before_mean = True

        self.group_means = make_group_means(
            self.processed_data,
            self.perturbation_column_name,
            self.split_column_name,
            exp_before_mean=exp_before_mean,
        )
        if "Average_Perturbation_Train" in self.group_means.obs_names:
            self.avg_pert_expressions = (
                self.group_means["Average_Perturbation_Train"].to_df().squeeze()
            )
        else:
            self.avg_pert_expressions = None

        self.processed_data = self.filter_data(self.processed_data)
        if self.sort_genes_var is not None:
            self.processed_data = BaseRNAExpressionDataset.sort_by_var_column(
                self.processed_data, self.sort_genes_var
            )
        self.all_genes = self.processed_data.var_names
        is_control = self.processed_data.obs[perturbation_column_name] == "Control"

        # all splits get all the control cells
        self.control_cells = self.processed_data[is_control].copy()
        self.perturbation_cells = self.processed_data[~is_control].copy()

        self.perturbation_cells = self.limit_to_split(self.perturbation_cells)

        if limit_samples:
            self.perturbation_cells = random_subsampling(
                self.perturbation_cells,
                limit_samples,
                shuffle=limit_samples_shuffle,
            )

        logger.info(f"Number of Control cells: {self.control_cells.shape[0]}")
        logger.info(
            f"Number of Pertubations: {self.perturbation_cells.obs[self.perturbation_column_name].nunique()}"
        )
        logger.info(f"Number of Perturbation cells: {self.perturbation_cells.shape[0]}")
        self.label_columns = label_columns
        self.regression_label_columns = regression_label_columns
        assert not self.perturbation_cells.is_view, "Must use a copy, not a view"
        assert not self.control_cells.is_view, "Must use a copy, not a view"

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns
        -------
            int: The length of the dataset.
        """
        return len(self.perturbation_cells)

    def limit_to_split(self, data: AnnData):
        if self.split is not None:
            is_this_split = data.obs[self.split_column_name] == self.split
            data = data[is_this_split].copy()
        return data

    def filter_data(self, data: AnnData):
        if self.filter_query is not None:
            filtered_idx = data.obs.query(self.filter_query).index
            data = data[filtered_idx]
        if self.limit_genes:
            data = BaseRNAExpressionDataset.limit_data_to_gene_list(
                data, self.limit_genes
            )
            de_genes_keys = [
                "rank_genes_groups_cov_all",
                "top_non_dropout_de_20",
                "top_non_zero_de_20",
            ]
            limit_genes_set = set(self.limit_genes)
            for key in {*de_genes_keys} & {*data.uns_keys()}:
                limit_de = {}
                for pert, de_genes in data.uns[key].items():
                    # limit_de[pert] = [g for g in de_genes if g in self.limit_genes]
                    limit_de[pert] = [g for g in de_genes if g in limit_genes_set]
                data.uns[key] = limit_de
        return data

    def __getitem__(self, index: int) -> MultiFieldInstance:
        """
        Returns a dictionary containing the data for the given index.

        Args:
        ----
            index (int): The index of the data to return.

        Returns:
        -------
            dict: A dictionary containing the data for the given index.
        """
        perturbed_cell_expressions = self.perturbation_cells.X[index].toarray()[0]
        random_cc_index = random.randint(0, self.control_cells.X.shape[0] - 1)
        control_cell_expressions = self.control_cells.X[random_cc_index].toarray()[0]

        perturb_genes = sorted(
            self.perturbation_cells.obs[self.perturbation_column_name]
            .iloc[index]
            .split("_")
        )
        perturb_gene_indices = [self.all_genes.get_loc(gene) for gene in perturb_genes]

        if self.expose_zeros == ExposeZerosMode.ALL:
            gene_indices = np.arange(len(self.all_genes))
        elif self.expose_zeros == ExposeZerosMode.LABEL_NONZEROS:
            # get non-zero indices from control and perturbation cells
            nz_control_indices = control_cell_expressions.nonzero()[0]
            nz_perturb_indices = perturbed_cell_expressions.nonzero()[0]

            # combine the indices
            gene_indices = sorted(
                set(nz_perturb_indices)
                | set(nz_control_indices)
                | set(perturb_gene_indices)
            )
        elif self.expose_zeros == ExposeZerosMode.NO_ZEROS:
            nz_control_indices = control_cell_expressions.nonzero()[0]
            gene_indices = sorted(set(nz_control_indices) | set(perturb_gene_indices))
        else:
            raise ValueError("Unknown expose_zeros strategy")
        # TODO: this may we wrong.
        genes = self.all_genes[gene_indices].to_list()
        perturbation_vector = ["0"] * len(gene_indices)
        for i, gene in enumerate(genes):
            if gene in perturb_genes:
                perturbation_vector[i] = "1"

        # convert pertrubation_cell sparse array to list of str
        metadata = {
            "cell_name": self.perturbation_cells.obs.index[index],
            "control_cell_name": self.control_cells.obs.index[random_cc_index],
            "perturbed_genes": "_".join(perturb_genes),
        }
        if self.label_columns is not None:
            for label_col in self.label_columns:
                metadata[label_col] = self.perturbation_cells.obs[label_col].iloc[index]
        if self.regression_label_columns is not None:
            for label_col in self.regression_label_columns:
                metadata[label_col] = self.perturbation_cells.obs[label_col].iloc[index]

        perturbed_expressions = perturbed_cell_expressions[gene_indices]
        control_expressions = control_cell_expressions[gene_indices]

        if self.avg_pert_expressions is not None:
            baseline_prediction_expressions = self.avg_pert_expressions.loc[
                genes
            ].to_numpy()
        else:
            # so that it comes out zero and doesn't push training any way
            baseline_prediction_expressions = perturbed_expressions

        return MultiFieldInstance(
            data={
                "genes": genes,
                "expressions": list(control_expressions),
                "perturbations": perturbation_vector,
                "label_expressions": list(perturbed_expressions),
                "delta_expressions": list(perturbed_expressions - control_expressions),
                "delta_baseline_expressions": list(
                    perturbed_expressions - baseline_prediction_expressions
                ),
            },
            metadata=metadata,
        )
