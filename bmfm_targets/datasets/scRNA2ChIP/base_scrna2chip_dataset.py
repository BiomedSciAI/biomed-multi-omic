import inspect

import numpy as np

try:
    # anndata >= 0.11
    from anndata.abc import CSRDataset as SparseDataset  # noqa: F401
except ImportError:
    # anndata >= 0.10
    pass


from bmfm_targets.datasets.base_rna_dataset import BaseRNAExpressionDataset
from bmfm_targets.datasets.datasets_utils import make_group_means
from bmfm_targets.tokenization import MultiFieldInstance


class BasescRNA2ChIPDataset(BaseRNAExpressionDataset):
    def __init__(self, *args, **kwargs):
        # Forward only kwargs that BaseRNAExpressionDataset accepts. scRNA2ChIP-specific
        # kwargs (new_field, chip_pairing_strategy) and unused extras
        # (perturbation_column_name) are filtered out here so the upstream base dataset
        # doesn't need a **kwargs catch-all.
        super_params = inspect.signature(BaseRNAExpressionDataset.__init__).parameters
        super_kwargs = {k: v for k, v in kwargs.items() if k in super_params}
        super().__init__(*args, **super_kwargs)
        self.new_field = kwargs["new_field"]
        # chip_pairing_strategy is extracted after the super() filter (same pattern as
        # new_field): "random" keeps the existing random-draw behaviour; "pseudobulk"
        # pairs each scRNA cell to the deterministic mean ChIP profile of its group.
        self.chip_pairing_strategy = kwargs.get("chip_pairing_strategy", "random")
        self.chipseq_cells = self.processed_data[
            self.processed_data.obs.query("data_type == 'ChIP'").index
        ]
        self.processed_data = self.processed_data[
            self.processed_data.obs.query("data_type == 'scRNA'").index
        ]
        self.binned_data = self.processed_data.X
        self.cell_names = np.array(self.processed_data.obs_names)
        self.metadata = self.processed_data.obs

        # Always build group_means for per-group metrics (used by the module via
        # prepare_extra_training_module_kwargs → self.kwargs["group_means"]).
        self.group_means = self._build_chip_group_means()
        # Reuse the same AnnData for pseudobulk pairing; the random strategy
        # doesn't need _chip_group_means at item-fetch time.
        if self.chip_pairing_strategy == "pseudobulk":
            self._chip_group_means = self.group_means
        else:
            self._chip_group_means = None

    def _build_chip_group_means(self):
        """
        Precompute per-group pseudobulk means over the ChIP cells.

        For a single label column (the common case) the column name is used
        directly as the grouping key.  For multiple label columns a composite
        key column is built by joining the column values with ``"__"`` so that
        ``make_group_means`` receives a single string column.

        Returns
        -------
        sc.AnnData
            One pseudobulk row per unique label value (or composite label).
            ``obs_names`` are the label values used for lookup in
            ``_get_label_key``.
        """
        label_cols = self.label_columns if self.label_columns else ["tissue_label"]
        if len(label_cols) == 1:
            return make_group_means(
                self.chipseq_cells,
                perturbation_column_name=label_cols[0],
                split_column_name=None,
                avg_row_label=None,
                exp_before_mean=False,
            )
        # Multiple label columns: build a composite key so make_group_means
        # receives a single grouping column.
        sep = "__"
        composite_col = sep.join(label_cols)
        chipseq_copy = self.chipseq_cells.copy()
        chipseq_copy.obs[composite_col] = chipseq_copy.obs[label_cols].apply(
            lambda row: sep.join(str(row[col]) for col in label_cols), axis=1
        )
        return make_group_means(
            chipseq_copy,
            perturbation_column_name=composite_col,
            split_column_name=None,
            avg_row_label=None,
            exp_before_mean=False,
        )

    def _get_label_key(self, metadata: dict) -> str:
        """
        Return the obs-name key used to look up a pseudobulk row.

        Parameters
        ----------
        metadata : dict
            Sample metadata dict as returned by ``get_sample_metadata``.

        Returns
        -------
        str
            Single label value for a single label column, or composite
            ``"val1__val2"`` for multiple label columns.
        """
        label_cols = self.label_columns if self.label_columns else ["tissue_label"]
        if len(label_cols) == 1:
            return str(metadata[label_cols[0]])
        sep = "__"
        return sep.join(str(metadata[col]) for col in label_cols)

    def _select_chip_target(
        self,
        metadata: dict,
        candidate_matches,
    ) -> tuple[list[float], dict]:
        """
        Select a ChIP expression profile to pair with an scRNA cell.

        Dispatches on ``self.chip_pairing_strategy``:

        - ``"random"``: draw one ChIP cell at random from ``candidate_matches``
          (original behaviour, non-deterministic).
        - ``"pseudobulk"``: return the precomputed mean expression profile for
          the cell's group label (deterministic).

        Parameters
        ----------
        metadata : dict
            Sample metadata for the scRNA cell (from ``get_sample_metadata``).
        candidate_matches : sc.AnnData
            ChIP cells that match the scRNA cell's label(s).

        Returns
        -------
        tuple[list[float], dict]
            ``(paired_sample_values, paired_metadata)`` where
            ``paired_metadata`` contains at least a ``"sample_name"`` key
            (consumed by ``merge_mfis`` to set ``paired_name``).
        """
        if self.chip_pairing_strategy == "pseudobulk":
            label_value = self._get_label_key(metadata)
            values = self._chip_group_means[label_value].X.toarray().tolist()[0]
            paired_metadata = {"sample_name": f"pseudobulk::{label_value}"}
            return values, paired_metadata
        # "random": original behaviour — draw one ChIP cell at random.
        cell_idx = np.random.randint(len(candidate_matches))
        values = candidate_matches[cell_idx].X.toarray().tolist()[0]
        paired_metadata = dict(candidate_matches.obs.iloc[cell_idx])
        paired_metadata["sample_name"] = str(candidate_matches.obs.index[cell_idx])
        return values, paired_metadata

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
        paired_sample_genes = self.all_genes
        paired_sample_values, paired_metadata = self._select_chip_target(
            metadata, candidate_matches
        )

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
