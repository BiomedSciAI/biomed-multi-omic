import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
from litdata import StreamingDataset
from litdata.streaming.item_loader import BaseItemLoader, Interval
from scipy.sparse import csr_matrix

from bmfm_targets.datasets.data_conversion.serializers import IndexSerializer
from bmfm_targets.tokenization import MultiFieldInstance
from bmfm_targets.training.streaming_datamodule import StreamingDataModule

logger = logging.getLogger(__name__)


class CellXGeneNexusDataset(StreamingDataset):
    """
    Dataset object built on top of CellXGene's SOMA TileDB database.
    For details see https://chanzuckerberg.github.io/cellxgene-census/index.html.
    """

    def __init__(
        self,
        index_dir: str,
        uri: (
            str | Path | None
        ) = "/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/pretrain/cellxgene/soma-2023-12-15",
        census_version: str | None = "2023-12-15",
        experiment: str = "homo_sapiens",
        raw_counts: bool = False,
        split: str = "train",
        label_columns: list[str] | None = None,
        regression_label_columns: list[str] | None = None,
        expose_zeros: Literal["all"] | None = None,
        limit_genes: list[str] | None = None,
        **kwargs,
    ):
        """
        CellxGene nexus dataset.

        Args:
        ----
        uri (str  |  Path, optional): Path to soma database. If `None`, will access the hosted
            version on AWS based on `census_version`, which may be slow but can run from . Defaults to the downloaded copy
            on CCC at "/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/pretrain/cellxgene/soma-2023-12-15".
        raw_counts (bool): select "raw" or "normalized" X in SOMA schema
        split (str | None, optional): Split to use, must match a split in the split file,
            otherwise the complete dataset is used. Defaults to "train".
        label_columns (list(str), optional): Label columns for MultiFieldInstance metadata.
            Defaults to ["cell_type"].

        """
        self.label_columns = label_columns
        self.regression_label_columns = regression_label_columns

        item_loader = TileDBItemLoader(
            uri,
            census_version,
            experiment,
            raw_counts,
            label_columns,
            regression_label_columns,
            expose_zeros,
            limit_genes,
        )

        index_dir = os.path.join(index_dir, split)

        with open(os.path.join(index_dir, "index.json")) as file:
            self.label_dict = json.load(file)["config"]["tiledb"]["label_dict"]

        super().__init__(index_dir, item_loader=item_loader, **kwargs)


class CellXGeneNexusDataModule(StreamingDataModule):
    DATASET_FACTORY = CellXGeneNexusDataset


class TileDBItemLoader(BaseItemLoader):
    """The base item loader is responsible to decide how the items within a chunk are loaded."""

    def __init__(
        self,
        db_dir,
        census_version: str,
        experiment: str,
        raw_counts: bool,
        label_columns: list[str] | None,
        regression_label_columns: list[str] | None,
        expose_zeros: Literal["all"] | None,
        limit_genes: list[str] | None,
        *args,
        **kwargs,
    ):
        self.census_version = census_version
        self.experiment = experiment
        self.raw_counts = raw_counts
        self.db_dir = db_dir
        self.label_columns = label_columns
        self.regression_label_columns = regression_label_columns
        self.expose_zeros = expose_zeros
        self.limit_genes = limit_genes

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)
        self.chunk_size = self._config["chunk_size"]
        self.buffer = None
        self.buffer_chunk_index = None

    def setup_tiledb(self, *args, **kwargs):
        import cellxgene_census as cc

        census_db = cc.open_soma(uri=self.db_dir, census_version=self.census_version)
        self.soma_experiment = census_db["census_data"][self.experiment]

        self.all_genes = (
            self.soma_experiment.ms["RNA"]
            .var.read(column_names=["feature_name"])
            .concat()
            .to_pandas()
            .squeeze()
            .to_numpy()
        )
        self.genes = self.all_genes
        self.gene_index = None
        if self.limit_genes is not None:
            mask = np.isin(self.all_genes, self.limit_genes)
            self.gene_index = np.where(mask)[0]
            self.genes = self.all_genes[mask]

    def state_dict(self) -> dict:
        return {}

    def generate_intervals(self) -> list[tuple[int, int]]:
        """Returns a list of tuple describing the indexes intervals of the chunks."""
        intervals = []
        offset = 0
        for chunk in self._chunks:
            chunk_size = chunk["chunk_size"]
            start_idx, end_idx = offset, offset + chunk_size
            intervals.append(Interval(start_idx, start_idx, end_idx, end_idx))
            offset += chunk_size
        return intervals

    def load_chunk(self, chunk_filepath: str):
        with open(chunk_filepath, "rb") as file:
            index = IndexSerializer.deserialize(file.read())

        obs = None
        if self.label_columns:
            obs = self.soma_experiment.obs.read(
                coords=(index,),
                result_order="row-major",
                column_names=self.label_columns,
            )
            obs = next(iter(obs))
            obs = list(zip(*[obs[i].to_pylist() for i in self.label_columns]))
        if self.limit_genes is not None:
            coords = (index, self.gene_index)
        else:
            coords = (index,)
        csr_express_matrix, (_, col_index) = next(
            iter(
                self.soma_experiment.ms["RNA"]
                .X["raw" if self.raw_counts else "normalized"]
                .read(coords=coords)
                .blockwise(axis=0)
                .scipy()
            )
        )
        assert len(col_index) == len(self.genes)
        assert isinstance(csr_express_matrix, csr_matrix)
        return (index, csr_express_matrix, obs)

    def pre_load_chunk(self, chunk_index: int, chunk_filepath: str) -> None:
        """Logic to load the chunk in background to gain some time."""
        pass

    def load_item_from_chunk(
        self,
        index: int,
        chunk_index: int,
        chunk_filepath: str,
        begin: int,
        chunk_bytes: int,
    ) -> MultiFieldInstance:
        """Returns an item loaded from a chunk."""
        if self.buffer_chunk_index is None:
            self.setup_tiledb()
        if self.buffer_chunk_index != chunk_index:
            self.buffer_chunk_index = chunk_index
            self.buffer = self.load_chunk(chunk_filepath)

        cell_index, csr_express_matrix, obs = self.buffer

        item_index = index - begin
        mfi_metadata = (
            {f: b for b, f in zip(obs[item_index], self.label_columns)} if obs else {}
        )
        mfi_metadata["cell_name"] = cell_index[item_index]
        if self.expose_zeros == "all":
            genes, expressions = get_genes_and_expressions(
                item_index, csr_express_matrix, self.genes
            )
        else:
            genes, expressions = get_non_zero_genes_and_expressions(
                item_index, csr_express_matrix, self.genes
            )

        mfi_data = {"genes": [*genes], "expressions": [*expressions]}
        item = MultiFieldInstance(data=mfi_data, metadata=mfi_metadata)
        return item

    def delete(self, chunk_index: int, chunk_filepath: str) -> None:
        """Delete a chunk from the local filesystem."""
        pass

    def encode_data(
        self, data: list[bytes], sizes: list[int], flattened: list[Any]
    ) -> Any:
        raise RuntimeError("The method encode_data is not implemented.")


def get_genes_and_expressions(
    item_index: int, csr_express_matrix: csr_matrix, genes: list[str]
):
    return genes, csr_express_matrix[item_index].toarray().tolist()[0]


def get_non_zero_genes_and_expressions(
    item_index: int, csr_express_matrix: csr_matrix, genes
):
    indptr = csr_express_matrix.indptr
    expression_tokens = csr_express_matrix.data[
        indptr[item_index] : indptr[item_index + 1]
    ]
    col_index = csr_express_matrix.indices[indptr[item_index] : indptr[item_index + 1]]
    gene_tokens = genes[col_index]
    return gene_tokens, expression_tokens
