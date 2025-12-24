import json
import logging
import os
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
from litdata import CombinedStreamingDataset, StreamingDataset
from litdata.streaming.item_loader import BaseItemLoader, Interval
from pydantic import BaseModel, validate_call
from scipy.sparse import csr_matrix

from bmfm_targets.config import SplitEnum
from bmfm_targets.config.dataset_config import ExposeZerosMode
from bmfm_targets.datasets.data_conversion.serializers import IndexSerializer
from bmfm_targets.tokenization import MultiFieldInstance
from bmfm_targets.training.streaming_datamodule import StreamingDataModule

logger = logging.getLogger(__name__)


class PerturbxPars(BaseModel):
    """
    Paths of Perturbx dataset.

    Attributes
    ----------
    dataset_path:
        Path to h5ad anndata file
    index_dir:
        litdata index dir
    weight:
        weight for this dataset
    split:
        dataset split
    """

    dataset_path: str | Path
    index_dir: str | Path
    split: SplitEnum
    weight: float | None = None


@validate_call
def build_perturbx_dataset(
    dataset_pars: list[PerturbxPars] | PerturbxPars,
    iterate_over_all: bool,
    split: SplitEnum,
    limit_genes: list[str] | None = None,
    expose_zeros: ExposeZerosMode = ExposeZerosMode.LABEL_NONZEROS,
    alternate_prob: float = 0.5,
    aggregate_file_path: str | Path | None = None,
    **kwargs,
):
    if split == SplitEnum.TEST:
        kwargs = kwargs.copy()
        kwargs["drop_last"] = False

    if not isinstance(dataset_pars, list):
        dataset = PerturbxDataset(
            dataset_pars,
            limit_genes,
            expose_zeros,
            alternate_prob,
            **kwargs,
        )
    else:
        pars = [p for p in dataset_pars if p.split == split]
        weights = (
            [p.weight if p.weight is not None else 1.0 for p in pars]
            if split == SplitEnum.TRAIN and not iterate_over_all
            else None
        )
        datasets = [
            PerturbxDataset(
                p,
                limit_genes,
                expose_zeros,
                alternate_prob,
                **kwargs,
            )
            for p in pars
        ]
        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            dataset = CombinedStreamingDataset(
                datasets, weights=weights, iterate_over_all=iterate_over_all
            )

    if split == SplitEnum.TRAIN and aggregate_file_path is not None:
        adata = ad.read_h5ad(aggregate_file_path)
        dataset.group_means = adata
        dataset.processed_data = adata

    return dataset


class PerturbxDataset(StreamingDataset):
    """
    Reading cell expressions from multiple h5ad files using LitData as frontend.

    Args:
    ----
    dataset_pars(PerturbxPars): Paths to folder with h5ad files and index, weights, split
    """

    def __init__(
        self,
        dataset_pars: PerturbxPars,
        limit_genes: list[str] | None,
        expose_zeros: ExposeZerosMode,
        alternate_prob: float,
        **kwargs,
    ):
        """
        Args:
        ----
            paths (PerturbxPars): Path to folder with h5ad files.
            index_dir (str | Path) : Path to litdata index folder.
        """
        if limit_genes and len(set(limit_genes)) != len(limit_genes):
            raise ValueError("limit_genes parameter has duplicated entries.")

        with open(os.path.join(dataset_pars.index_dir, "index.json")) as file:
            target_gene_column = json.load(file)["config"]["ptb"]["target_gene_column"]

        item_loader = PerturbxItemLoader(
            dataset_pars,
            target_gene_column,
            limit_genes,
            expose_zeros,
            alternate_prob,
        )
        super().__init__(str(dataset_pars.index_dir), item_loader=item_loader, **kwargs)


def _copy_anndata_view(adata: ad.AnnData) -> ad.AnnData:
    if adata.is_view:
        return adata.copy()
    else:
        return adata


class PerturbxItemLoader(BaseItemLoader):
    def __init__(
        self,
        dataset_pars: PerturbxPars,
        target_gene_column: str,
        limit_genes: list[str] | None,
        expose_zeros: ExposeZerosMode,
        alternate_prob: float,
    ):
        self.paths = dataset_pars
        self.buffer_chunk_index = None
        self.adata = None
        self.target_gene_column = target_gene_column
        self.limit_genes = limit_genes
        self.expose_zeros = expose_zeros
        if self.expose_zeros == ExposeZerosMode.ALTERNATE:
            self.alternate_prob = alternate_prob

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)

    def setup_anndata(self, *args, **kwargs):
        self.adata = ad.read_h5ad(self.paths.dataset_path, backed="r")
        var_names = self.adata.var_names

        if self.limit_genes:
            limit_gene_inds = np.nonzero(
                np.isin(var_names, self.limit_genes, assume_unique=True)
            )[0]
            var_names = var_names[limit_gene_inds]
            self.limit_gene_inds = limit_gene_inds

        self.var_names = var_names
        self.var_name_map = {i: index for index, i in enumerate(var_names.tolist())}

        self.control_index, self.control_chunk_size = read_control_index(
            os.path.join(self.paths.index_dir, "control_index.bin")
        )

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

    def load_chunk(self, chunk_filepath: str) -> csr_matrix:
        with open(chunk_filepath, "rb") as file:
            index = IndexSerializer.deserialize(file.read())

        control_chunk_size = min(self.control_chunk_size, len(index))
        ptb_subset = self.adata[index, :].to_memory()
        if self.limit_genes:
            ptb_subset = ptb_subset[:, self.limit_gene_inds].copy()

        ptb_X = csr_matrix(ptb_subset.X)
        ptb_obs_names = ptb_subset.obs_names
        target_gene = ptb_subset.obs[self.target_gene_column]

        rng = np.random.default_rng()
        n = len(self.control_index)
        permutation = rng.permutation(control_chunk_size)
        start = rng.integers(0, n)
        if n - start >= control_chunk_size:
            control_index = self.control_index[start : start + control_chunk_size]
        else:
            control_index = np.hstack(
                [
                    self.control_index[start:],
                    self.control_index[: control_chunk_size - (n - start)],
                ]
            )
        controls = self.adata[control_index, :].to_memory()
        if self.limit_genes:
            controls = controls[:, self.limit_gene_inds].copy()
        controls = controls[permutation]
        control_X = csr_matrix(controls.X)
        control_obs_names = controls.obs_names

        return (ptb_X, ptb_obs_names, control_X, control_obs_names, target_gene)

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
            self.setup_anndata()
        if self.buffer_chunk_index != chunk_index:
            self.buffer_chunk_index = chunk_index
            self.buffer = self.load_chunk(chunk_filepath)

        item_index = index - begin
        ptb_X, ptb_obs_names, control_X, control_obs_names, target_gene = self.buffer

        control_chunk_size = len(control_obs_names)

        ptb_X, control_X = ptb_X[item_index], control_X[item_index % control_chunk_size]
        ptb_nz, control_nz = ptb_X.nonzero()[1], control_X.nonzero()[1]
        target_genes = target_gene.iloc[item_index].split("_")
        target_gene_indices = [self.var_name_map[gene] for gene in target_genes]

        target_gene_indices_set = set(target_gene_indices)
        control_nz_set = set(control_nz)
        ptb_nz_set = set(ptb_nz)

        if self.expose_zeros == ExposeZerosMode.LABEL_NONZEROS:
            gene_indices = ptb_nz_set | control_nz_set
        elif self.expose_zeros == ExposeZerosMode.NO_ZEROS:
            gene_indices = set(control_nz_set)
        elif self.expose_zeros == ExposeZerosMode.ALL:
            gene_indices = set(np.arange(control_X.shape[1]))
        else:
            control_only = np.random.choice(
                [True, False], p=[self.alternate_prob, 1 - self.alternate_prob]
            )
            gene_indices = (
                control_nz_set if control_only else ptb_nz_set | control_nz_set
            )
        gene_indices |= target_gene_indices_set

        gene_indices = list(gene_indices)
        gene_indices.sort()
        genes = self.var_names[gene_indices]
        ptb_vector = np.zeros(len(genes), dtype=np.int_)
        ptb_vector[np.isin(genes, target_genes, assume_unique=True)] = 1
        ptb_vector = ptb_vector.astype("str").tolist()

        return MultiFieldInstance(
            data={
                "genes": genes.tolist(),
                "expressions": control_X.toarray()[0][gene_indices].tolist(),
                "perturbations": ptb_vector,
                "label_expressions": ptb_X.toarray()[0][gene_indices].tolist(),
            },
            metadata={
                "cell_name": ptb_obs_names[item_index],
                "control_cell_name": control_obs_names[item_index % control_chunk_size],
                "perturbed_genes": "_".join(target_genes),
            },
        )

    def delete(self, chunk_index: int, chunk_filepath: str) -> None:
        """Delete a chunk from the local filesystem."""
        pass

    def encode_data(
        self, data: list[bytes], sizes: list[int], flattened: list[Any]
    ) -> Any:
        raise RuntimeError("The method encode_data is not implemented.")


class PerturbxDataModule(StreamingDataModule):
    DATASET_FACTORY = build_perturbx_dataset


def is_hdf5_file(path):
    with open(path, "rb") as f:
        signature = f.read(8)
    return signature == b"\x89HDF\r\n\x1a\n"


def read_control_index(path: str):
    """Reads idex for control samples, check format of index."""
    if is_hdf5_file(path):
        control_index = h5py.File(path, "r")["control_index"]
        control_chunk_size = control_index.chunks[0]
    else:
        with open(path, "rb") as file:
            data = file.read()
        data = IndexSerializer.deserialize(data)
        control_chunk_size = data[-1]  # chunk size is written at the end
        control_index = data[:-1]
    return control_index, control_chunk_size
