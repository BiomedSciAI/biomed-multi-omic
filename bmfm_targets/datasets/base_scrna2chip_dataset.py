import numpy as np

try:
    # anndata >= 0.11
    from anndata.abc import CSRDataset as SparseDataset  # noqa: F401
except ImportError:
    # anndata >= 0.10
    pass


from bmfm_targets.datasets.base_rna_dataset import BaseRNAExpressionDataset
from bmfm_targets.tokenization import MultiFieldInstance

# class BasescRNA2ChIPDataset(Dataset, abc.ABC):
#     """A PyTorch Dataset for the  perturbation dataset."""
#
#     source_h5ad_file_name: list[str] = ...
#
#     def __init__(
#         self,
#         processed_data_source: AnnData | str = "processed.h5ad",
#         split: str | None = None,
#         split_column_name: str | None = None,
#         perturbation_column_name: str | None = None,
#         stratifying_label: str | None = None,
#         backed: Literal["r", "r+"] | None = None,
#         limit_samples: int | None = None,
#         limit_samples_shuffle: bool = True,
#         limit_genes: list[str] | None = None,
#         filter_query: str | None = None,
#     ) -> None:
#         """
#         Initializes the dataset.
#
#         Args:
#         ----
#             processed_data_source (AnnData | str): either an AnnData object that has been processed or the path to such an h5ad file.
#             split (str): Split to use. Must be one of train, dev, test or None to get all data.
#             split_column_name (str): The column name where split is stored. If None, all of the data will be used as test.
#             perturbation_column_name (str): The column name in the AnnData object that contains the perturbation information.
#             stratifying_label (str): The column name in the AnnData object that contains the stratifying label.
#             backed (Literal["r", "r+"] | None): Whether to read the data in backed mode. If None, the data will be read in memory.
#             limit_samples (int | None): The number of samples to limit the dataset to.
#             limit_samples_shuffle (bool): Whether to shuffle the samples before limiting the dataset.
#             limit_genes (list[str] | None): The list of genes to limit the dataset to.
#             filter_query (str | None): The query to filter the data. If None, no filtering will be applied.
#
#         Raises:
#         ------
#             ValueError: If the split is not one of train, dev, test.
#             ValueError: If the data is not sparse
#         """
#         if split not in ["train", "dev", "test", None]:
#             raise ValueError("The split must be one of train, dev, test or None.")
#
#         self.split = split
#         self.backed = backed
#         self.filter_query = filter_query
#         self.limit_genes = limit_genes
#
#         self.processed_data_source = processed_data_source
#         self.perturbation_column_name = perturbation_column_name
#         if split_column_name is None:
#             split_column_name = f"split_stratified_{stratifying_label}"
#         self.split_column_name = split_column_name
#
#         if isinstance(processed_data_source, AnnData):
#             self.processed_data = processed_data_source.copy()
#         else:
#             self.processed_data = read_h5ad(processed_data_source, backed=self.backed)
#
#         if not (
#             scipy.sparse.issparse(self.processed_data.X)
#             or isinstance(self.processed_data.X, SparseDataset)
#         ):
#             raise ValueError("Data is not sparse")
#
#         self.processed_data = self.filter_data(self.processed_data)
#
#         self.all_genes = self.processed_data.var_names
#         is_control = self.processed_data.obs[perturbation_column_name] == "Control"
#
#         # all splits get all the control cells
#         self.control_cells = self.processed_data[is_control]
#
#         self.perturbation_cells = self.processed_data[~is_control]
#         self.perturbation_cells = self.limit_to_split(self.perturbation_cells)
#
#         if limit_samples:
#             self.perturbation_cells = random_subsampling(
#                 self.perturbation_cells,
#                 limit_samples,
#                 shuffle=limit_samples_shuffle,
#             )
#
#         logger.info(f"Number of Control cells: {self.control_cells.shape[0]}")
#         logger.info(f"Number of Perturbation cells: {self.perturbation_cells.shape[0]}")
#
#         self.group_means = self.make_group_means(self.processed_data)
#
#     def make_group_means(self, ad: AnnData):
#         import numpy as np
#         import scipy.sparse as ss
#
#         grouped = ad.obs.groupby(self.perturbation_column_name)
#
#         mean_expressions = [
#             ad[grouped.groups[group]].X.mean(axis=0) for group in grouped.groups
#         ]
#         mean_adata = AnnData(X=ss.csr_matrix(np.vstack(mean_expressions)))
#         mean_adata.var_names = ad.var_names
#         mean_adata.obs[self.perturbation_column_name] = list(grouped.groups.keys())
#         mean_adata.obs = mean_adata.obs.set_index(self.perturbation_column_name)
#
#         return BaseRNAExpressionDataset(processed_data_source=mean_adata, split=None)
#
#     def __len__(self) -> int:
#         """
#         Returns the length of the dataset.
#
#         Returns
#         -------
#             int: The length of the dataset.
#         """
#         return len(self.perturbation_cells)
#
#     def limit_to_split(self, data: AnnData):
#         if self.split is not None:
#             is_this_split = data.obs[self.split_column_name] == self.split
#             data = data[is_this_split]
#         return data
#
#     def filter_data(self, data: AnnData):
#         if self.filter_query is not None:
#             filtered_idx = data.obs.query(self.filter_query).index
#             data = data[filtered_idx]
#         if self.limit_genes:
#             data = BaseRNAExpressionDataset.limit_data_to_gene_list(
#                 data, self.limit_genes
#             )
#             de_genes_keys = [
#                 "rank_genes_groups_cov_all",
#                 "top_non_dropout_de_20",
#                 "top_non_zero_de_20",
#             ]
#             limit_genes_set = set(self.limit_genes)
#             for key in {*de_genes_keys} & {*data.uns_keys()}:
#                 limit_de = {}
#                 for pert, de_genes in data.uns[key].items():
#                     # limit_de[pert] = [g for g in de_genes if g in self.limit_genes]
#                     limit_de[pert] = [g for g in de_genes if g in limit_genes_set]
#                 data.uns[key] = limit_de
#         return data
#
#     def __getitem__(self, index: int) -> MultiFieldInstance:
#         """
#         Returns a dictionary containing the data for the given index.
#
#         Args:
#         ----
#             index (int): The index of the data to return.
#
#         Returns:
#         -------
#             dict: A dictionary containing the data for the given index.
#         """
#         perturbed_cell_expressions = self.perturbation_cells.X[index].toarray()[0]
#         random_cc_index = random.randint(0, self.control_cells.X.shape[0] - 1)
#         control_cell_expressions = self.control_cells.X[random_cc_index].toarray()[0]
#
#         # get non-zero indices from control and perturbation cells
#         nz_control_indices = control_cell_expressions.nonzero()[0]
#         nz_perturb_indices = perturbed_cell_expressions.nonzero()[0]
#         perturb_genes = sorted(
#             self.perturbation_cells.obs[self.perturbation_column_name]
#             .iloc[index]
#             .split("_")
#         )
#         perturb_gene_indices = [self.all_genes.get_loc(gene) for gene in perturb_genes]
#         # combine the indices
#         all_indices = sorted(
#             set(nz_perturb_indices)
#             | set(nz_control_indices)
#             | set(perturb_gene_indices)
#         )
#
#         # TODO: this may we wrong.
#         genes = self.all_genes[all_indices].to_list()
#         perturbation_vector = ["0"] * len(all_indices)
#         for i, gene in enumerate(genes):
#             if gene in perturb_genes:
#                 perturbation_vector[i] = "1"
#
#         # convert pertrubation_cell sparse array to list of str
#         return MultiFieldInstance(
#             data={
#                 "genes": genes,
#                 "expressions": list(control_cell_expressions[all_indices]),
#                 "perturbations": perturbation_vector,
#                 "label_expressions": list(perturbed_cell_expressions[all_indices]),
#             },
#             metadata={
#                 "cell_name": self.perturbation_cells.obs.index[index],
#                 "control_cell_name": self.control_cells.obs.index[random_cc_index],
#                 "perturbed_genes": "_".join(perturb_genes),
#             },
#         )


class BasescRNA2ChIPDataset(BaseRNAExpressionDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.new_field = kwargs["new_field"]
        self.chipseq_cells = self.processed_data[
            self.processed_data.obs.query("data_type == 'ChIP'").index
        ]
        self.processed_data = self.processed_data[
            self.processed_data.obs.query("data_type == 'scRNA'").index
        ]
        self.binned_data = self.processed_data.X
        self.cell_names = np.array(self.processed_data.obs_names)
        self.metadata = self.processed_data.obs

    def get_sample_metadata(self, idx):
        cell_name = str(self.cell_names[idx])
        cell_metadata = self.metadata.loc[cell_name]
        categorical_columns = (
            self.label_columns if self.label_columns else ["tissue_label"]
        )
        regression_columns = (
            self.regression_label_columns if self.regression_label_columns else []
        )
        all_label_columns = [*categorical_columns, *regression_columns]
        metadata = {
            l: cell_metadata[l] for l in all_label_columns if l != self.new_field
        }

        metadata["cell_name"] = cell_name
        return metadata

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
        if idx > len(self.processed_data) - 1 or idx < 0:
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )
        if self.expose_zeros:
            genes, expressions = self.get_genes_and_expressions(idx)
        else:
            genes, expressions = self.get_genes_and_nonzero_expressions(idx)

        metadata = self.get_sample_metadata(idx)

        mfi = MultiFieldInstance(
            metadata=metadata,
            data={
                "genes": list(genes),
                "expressions": list(expressions),
            },
        )

        # find matching ChIP-seq samples
        candidate_matches = self.chipseq_cells.obs.index
        for col_label in self.label_columns if self.label_columns else ["tissue_label"]:
            col_label_for_sample = metadata[col_label]
            candidate_matches = np.intersect1d(
                self.chipseq_cells.obs.query(
                    f"{col_label} == @col_label_for_sample"
                ).index,
                candidate_matches,
            )
        candidate_matches = self.chipseq_cells[candidate_matches]

        if len(candidate_matches) == 0:
            # return mfi  # TODO does this work?
            raise ValueError(
                f"No matching ChIP-seq samples found for cell {metadata['cell_name']} with label {metadata['tissue_label']}"
            )
        cell_idx = np.random.randint(len(candidate_matches))
        paired_sample_genes = self.all_genes
        paired_sample_values = candidate_matches[cell_idx].X.toarray().tolist()[0]
        paired_metadata = dict(candidate_matches.obs.iloc[cell_idx])
        paired_metadata["sample_name"] = str(candidate_matches.obs.index[cell_idx])

        paired_mfi = MultiFieldInstance(
            metadata=paired_metadata,
            data={
                "genes": list(paired_sample_genes),
                self.new_field: list(paired_sample_values),
            },
        )

        merged_mfi = merge_mfis(mfi, paired_mfi, self.new_field)
        return merged_mfi


def merge_mfis(mfi: MultiFieldInstance, paired_mfi: MultiFieldInstance, new_field: str):
    first = dict(zip(mfi["genes"], mfi["expressions"]))
    second = dict(zip(paired_mfi["genes"], paired_mfi[new_field]))
    merged_genes = sorted({*mfi["genes"]} | {*paired_mfi["genes"]})
    merged_expressions = [first.get(g, 0) for g in merged_genes]
    merged_new_field = [second.get(g, 0) for g in merged_genes]
    return MultiFieldInstance(
        metadata={**mfi.metadata, "paired_name": paired_mfi.metadata["sample_name"]},
        data={
            "genes": merged_genes,
            "expressions": merged_expressions,
            new_field: merged_new_field,
        },
    )
