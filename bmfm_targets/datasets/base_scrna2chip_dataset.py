import numpy as np

try:
    # anndata >= 0.11
    from anndata.abc import CSRDataset as SparseDataset  # noqa: F401
except ImportError:
    # anndata >= 0.10
    pass


from bmfm_targets.datasets.base_rna_dataset import BaseRNAExpressionDataset
from bmfm_targets.tokenization import MultiFieldInstance


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
