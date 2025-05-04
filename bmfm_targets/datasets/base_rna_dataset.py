import logging
from collections.abc import Callable
from typing import Literal

import numpy as np
import scanpy as sc
import scipy
from anndata import AnnData, read_h5ad
from torch.utils.data import Dataset

from bmfm_targets.datasets.datasets_utils import (
    load_and_update_all_labels_json,
    random_subsampling,
)
from bmfm_targets.tokenization import MultiFieldInstance

logger = logging.getLogger(__name__)

try:
    # anndata >= 0.11
    from anndata.abc import CSRDataset as SparseDataset
except ImportError:
    # anndata >= 0.10
    from anndata.experimental import CSRDataset as SparseDataset


def multifield_instance_wrapper(genes, expressions, metadata):
    return MultiFieldInstance(
        metadata=metadata,
        data={
            "genes": list(genes),
            "expressions": list(expressions),
        },
    )


class BaseRNAExpressionDataset(Dataset):
    default_label_dict_path: str | None = None

    def __init__(
        self,
        processed_data_source: AnnData | str = "processed.h5ad",
        split: str | None = None,
        label_dict_path: str | None = None,
        split_column_name: str | None = None,
        label_columns: list[str] | None = None,
        regression_label_columns: list[str] | None = None,
        stratifying_label: str | None = None,
        backed: Literal["r", "r+"] | None = None,
        limit_samples: int | None = None,
        limit_samples_shuffle: bool = True,
        filter_query: str | None = None,
        sort_genes_var: str | None = None,
        expose_zeros: str | None = None,
        limit_genes: list[str] | None = None,
        output_wrapper: Callable | None = None,
    ) -> None:
        """
        Initializes the dataset.

        Args:
        ----
            processed_data_source: AnnData | str = either an AnnData object that has been processed
                or the path to such an h5ad file
            split (str): Split to use. Must be one of train, dev, test or None to get all data.
            transform_datasets (bool, optional): Whether to apply the transforms to the datasets.
            transforms (list[dict] | None, optional): List of transforms to be applied to the datasets.
            label_dict_path: str | None = None
            split_column_name: str | None = None, column name where split is stored. If None,
                a new split column with name `split_stratified_{label}` where label is the
                first value in label_columns will be created with samples stratified by label
            label_columns: list[str] | None = None, the columns from the h5ad file to use as labels
                in the dataset. If None, will use the column defined in the `default_label_column_name`
                of the dataset.
            regression_label_columns: list[str] | None
            backed (str | None, optional) = will be passed to `read_h5ad(..., backed=backed)`.
                If `backed='r'`, load underlying h5ad file without loading data into memory
                see scanpy docs for explanation.
                Only used for loading already processed data from disk (ie, transform_datasets=False)
            limit_samples: (int | None) limit the number of samples, None to load all
            limit_samples_shuffle (bool | None) : shuffle the limited samples (random sub sampling)
            filter_query (str | None) : A query to filter the data by
            sort_genes_var (str | None) : A var column name according to which genes will be sorted per sample
            expose_zeros (str | None) : whether to expose zeros, ie genes that had zero reads for a
              given sample. If "all", all zeros are exposed, if None, no zeros are exposed. Default, None.

        Raises:
        ------
            ValueError: If the data is not a sparse matrix.
            ValueError: If the split is not one of train, dev, test, or None.
            ValueError: If the sort_genes_var cannot be found in the var columns.
            FileNotFoundError: If the input data file (in h5ad format) does not exist.
            FileNotFoundError: If the processed data file does not exist and transform_datasets=False
        """
        if split not in ["train", "dev", "test", None]:
            raise ValueError(
                "Split must be one of train, dev, test; or None to get full dataset"
            )
        if expose_zeros not in ["all", None]:
            raise NotImplementedError(
                f"Unsupported option for exposing zeros: {expose_zeros}"
            )

        self.split = split
        self.backed = backed
        self.stratifying_label = stratifying_label
        self.filter_query = filter_query
        self.sort_genes_var = sort_genes_var
        self.expose_zeros = expose_zeros
        self.label_dict_path = label_dict_path
        self.label_dict = None  # will be set only if dataset has labels
        self.label_columns = label_columns
        self.regression_label_columns = regression_label_columns
        self.limit_genes = limit_genes
        if isinstance(processed_data_source, AnnData):
            self.processed_data = processed_data_source.copy()
        else:
            self.processed_data = read_h5ad(processed_data_source, backed=self.backed)

        if not (
            scipy.sparse.issparse(self.processed_data.X)
            or isinstance(self.processed_data.X, SparseDataset)
        ):
            raise ValueError("Data is not sparse")

        if self.label_columns or self.regression_label_columns:
            # need instance var, writing class variable changes all instances of class
            self.labels_requested = True
        else:
            self.labels_requested = False

        if self.labels_requested:
            if self.label_dict_path is None:
                self.label_dict_path = self.default_label_dict_path
            self.label_dict = self.get_label_dict(self.label_dict_path)

        if split_column_name is None:
            if self.stratifying_label:
                split_column_name = f"split_stratified_{self.stratifying_label}"
            else:
                split_column_name = "split_random"
        self.split_column_name = split_column_name

        self.processed_data = self.filter_data(self.processed_data)
        if self.sort_genes_var is not None:
            self.processed_data = self.sort_data(self.processed_data)

        self.metadata = self.processed_data.obs
        self.binned_data = self.processed_data.X
        self.all_genes = self.processed_data.var_names
        self.cell_names = np.array(self.processed_data.obs_names)

        if output_wrapper is None:
            self.output_wrapper = multifield_instance_wrapper
        else:
            self.output_wrapper = output_wrapper

        if limit_samples is not None:
            self.processed_data = random_subsampling(
                adata=self.processed_data,
                n_samples=limit_samples,
                shuffle=limit_samples_shuffle,
            )
        self.__post_init__()

    def __post_init__(self):
        pass

    @property
    def max_expression(self):
        if isinstance(self.binned_data, SparseDataset):
            return np.max(self.binned_data.group["data"])
        else:
            return np.max(self.binned_data.data)

    def max_counts(self, max_length: int | None = None):
        if isinstance(self.binned_data, SparseDataset):
            data = self.binned_data.to_memory()
        else:
            data = self.binned_data
        if max_length is not None:
            sums = []
            for i in range(len(data.indptr) - 1):
                expression_values = data.data[data.indptr[i] : data.indptr[i + 1]].data
                sorted_expression_values = np.sort(expression_values)[-max_length:]
                expression_values_sum = sorted_expression_values.sum()
                sums.append(expression_values_sum)
            return max(np.array(sums))
        else:
            return np.max(data.sum(axis=1))

    def get_vocab_for_field(self, field):
        if field == "expressions":
            return np.arange(0, self.max_expression + 1).astype(str)
        if field == "genes":
            return self.processed_data.var_names

    def get_label_dict(self, label_dict_path):
        label_dict = {}
        if self.label_columns:
            label_dict.update(
                {
                    l: self.get_sub_label_dict(l, label_dict_path)
                    for l in self.label_columns
                }
            )
        if self.regression_label_columns:
            label_dict.update({l: {0: 0} for l in self.regression_label_columns})
        return label_dict

    def get_sub_label_dict(self, label_column_name, label_dict_path):
        return load_and_update_all_labels_json(
            self.processed_data, label_dict_path, label_column_name
        )

    def get_genes_and_nonzero_expressions(self, idx):
        genes, expression_values = self.read_expressions_and_genes_from_csr(
            self.binned_data, self.all_genes, idx
        )
        return genes, expression_values

    def get_genes_and_expressions(self, idx):
        return self.all_genes, self.binned_data[idx].toarray().tolist()[0]

    def sort_data(self, data: AnnData):
        if self.sort_genes_var not in data.var.columns:
            raise ValueError(f"{self.sort_genes_var} not in data vars")
        ordering = data.var[self.sort_genes_var].values
        order_indices = ordering.argsort()
        data = data[:, order_indices]
        return data

    def filter_data(self, data: AnnData):
        if self.split is not None:
            is_this_split = data.obs[self.split_column_name] == self.split
            data = data[is_this_split]
        if self.filter_query is not None:
            filtered_idx = data.obs.query(self.filter_query).index
            data = data[filtered_idx]
        if self.limit_genes:
            data = self.limit_data_to_gene_list(data, self.limit_genes)
        return data

    @staticmethod
    def limit_data_to_gene_list(data: AnnData, limit_genes: list[str]) -> AnnData:
        initial_gene_count = len(data.var_names)
        filtered_vars = data.var.index.isin(limit_genes)
        data = data[:, filtered_vars]
        logger.info(
            f"Reduced dataset genes from {initial_gene_count} to {len(data.var_names)} which overlap with the {len(limit_genes)} in `limit_genes`"
        )
        # cells with no reads must be removed
        cm, cc = sc.pp.filter_cells(data, min_genes=1, inplace=False)
        data = data[cm, :]
        logger.info(f"Removed {sum(~cm)} cells which no longer have counts.")
        data = data.copy()
        return data

    def get_sample_metadata(self, idx):
        cell_name = str(self.cell_names[idx])
        cell_metadata = self.metadata.loc[cell_name]
        categorical_columns = self.label_columns if self.label_columns else []
        regression_columns = (
            self.regression_label_columns if self.regression_label_columns else []
        )
        all_label_columns = [*categorical_columns, *regression_columns]
        metadata = {l: cell_metadata[l] for l in all_label_columns}

        metadata["cell_name"] = cell_name
        return metadata

    @staticmethod
    def read_expressions_and_genes_from_csr(
        binned_data_csr,
        all_genes,
        idx,
    ):
        if isinstance(binned_data_csr, SparseDataset):
            assert binned_data_csr.format == "csr"
            index_item = binned_data_csr[idx]
            expression_values = index_item.data
            nz_rows = index_item.indices
            genes = all_genes[nz_rows]

        else:
            start_index = binned_data_csr.indptr[idx]
            end_index = binned_data_csr.indptr[idx + 1]

            expression_values = binned_data_csr.data[start_index:end_index].data
            nz_rows = binned_data_csr.indices[start_index:end_index]
            genes = all_genes[nz_rows]
        genes = [*genes]
        expression_values = [*expression_values]
        return genes, expression_values

    def __len__(self) -> int:
        """
        Returns the number of cell samples in the dataset.

        Returns
        -------
            int: Number of samples.
        """
        return self.processed_data.X.shape[0]

    def _get_item_by_index(self, idx: int) -> MultiFieldInstance:
        """
        Returns a single cell sample.

        Args:
        ----
            idx (int): Index of the cell sample.

        Returns:
        -------
            MultiFieldInstance: A single cell sample.
        """
        if self.expose_zeros:
            genes, expressions = self.get_genes_and_expressions(idx)
        else:
            genes, expressions = self.get_genes_and_nonzero_expressions(idx)

        metadata = self.get_sample_metadata(idx)

        return self.output_wrapper(genes, expressions, metadata)

    def __getitem__(
        self, idx: int | slice
    ) -> MultiFieldInstance | list[MultiFieldInstance]:
        """
        Returns one or more cell samples based on the index or slice.

        Args:
        ----
            idx (Union[int, slice]): Index or slice of the cell samples.

        Returns:
        -------
            Union[MultiFieldInstance, List[MultiFieldInstance]]: A single or multiple cell samples.
        """
        if isinstance(idx, int):  # Single index
            if idx < 0 or idx >= len(self):  # Ensure index is in range
                raise IndexError("Index out of range")
            return self._get_item_by_index(idx)
        elif isinstance(idx, slice):  # Slice
            start, stop, step = idx.indices(
                len(self)
            )  # Adjust slice for range and step
            return [self._get_item_by_index(i) for i in range(start, stop, step)]
        else:
            raise TypeError(f"Invalid argument type: {type(idx)}")
