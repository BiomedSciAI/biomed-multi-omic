import logging
import os
import shutil
from pathlib import Path
from typing import Any

import scanpy as sc
from anndata.experimental.multi_files import AnnCollection
from litdata import StreamingDataset
from litdata.streaming.item_loader import BaseItemLoader, Interval
from scipy.sparse import csr_matrix

from bmfm_targets.datasets.data_conversion.litdata_indexing import build_index
from bmfm_targets.datasets.data_conversion.serializers import IndexSerializer
from bmfm_targets.datasets.datasets_utils import load_and_update_all_labels_json
from bmfm_targets.tokenization import MultiFieldInstance
from bmfm_targets.training.streaming_datamodule import StreamingDataModule

logger = logging.getLogger(__name__)


class AnnCollectionDataset(StreamingDataset):
    """Reading cell expressions from multiple h5ad files using LitData as frontend."""

    label_dict_path: str | None = None

    def __init__(
        self,
        index_dir: str | Path,
        split: str,
        dataset_dir: str | Path | None = None,
        dataset_paths: list[str] | list[Path] | None = None,
        expose_zeros: str | None = None,
        label_dict_path: str | None = None,
        label_columns: list[str] | None = None,
        regression_label_columns: list[str] | None = None,
        **kwargs,
    ):
        """
        Args:
        ----
            dataset_dir (str  |  Path): Path to folder with h5ad files.
            index_dir (str | Path) : Path to litdata index folder.
        """
        if split not in ["train", "dev", "test", None]:
            raise ValueError(
                "Split must be one of train, dev, test; or None to get full dataset"
            )
        if expose_zeros not in ["all", None]:
            raise NotImplementedError(
                f"Unsupported option for exposing zeros: {expose_zeros}"
            )
        index_dir = Path(index_dir) / split
        _no_labels = label_columns is None and regression_label_columns is None

        src_paths = [
            *([] if dataset_dir is None else [dataset_dir]),
            *([] if dataset_paths is None else dataset_paths),
        ]

        item_loader = (
            AnnDataItemLoader(src_paths, str(index_dir), expose_zeros)
            if _no_labels
            else AnnDataItemLoaderWithLabels(
                src_paths,
                str(index_dir),
                expose_zeros,
                label_dict_path=label_dict_path,
                label_columns=label_columns,
                regression_label_columns=regression_label_columns,
            )
        )
        super().__init__(str(index_dir), item_loader=item_loader, **kwargs)
        self.HAS_LABELS = not _no_labels
        if self.HAS_LABELS:
            self.label_dict_path = label_dict_path
            self.label_dict = self.item_loader.get_label_dict()

    def __getitem__(self, idx: int) -> MultiFieldInstance:
        cell, genes, expressions = super().__getitem__(idx)
        return MultiFieldInstance(
            metadata={"cell": cell}
            if not self.HAS_LABELS
            else cell,  # AnnDataItemLoaderWithLabels returns metadata in 1st return value
            data={
                "genes": [*genes],
                "expressions": [*expressions],
            },
        )


class AnnCollectionDataModule(StreamingDataModule):
    DATASET_FACTORY = AnnCollectionDataset

    def prepare_data(self) -> None:
        """
        Supposed config parameters:
        label_columns:(for sequence_classification)
            - _target_: bmfm_targets.config.LabelColumnInfo
                label_column_name: "celltype"
                is_stratification_label: true
            - _target_: bmfm_targets.config.LabelColumnInfo
                label_column_name: "Final_CellType"
                is_stratification_label: true
        data_module
            dataset_kwargs:
                dataset_dir: /dccstor/bmfm-targets/data/omics/transcriptome/scRNA/finetune/AnnCollection/changepi_scp259/data
                dataset_paths:
                    - /dccstor/bmfm-targets/data/omics/transcriptome/scRNA/finetune/scIBD300K/processed_42.h5ad
                    - /dccstor/bmfm-targets/data/omics/transcriptome/scRNA/finetune/ChangAll/processed_42.h5ad
                index_dir: /dccstor/bmfm-targets/data/omics/transcriptome/scRNA/finetune/AnnCollection/changepi_cp259/index
            transform_datasets: true (if true, the contents in index_dir are erased and overwriten).
        """
        if not self.transform_datasets:
            return
        if self.transform_kwargs is None:
            transform_kwargs = {}
        else:
            transform_kwargs = self.transform_kwargs

        if self.dataset_kwargs is None:
            dataset_kwargs = {}
        else:
            dataset_kwargs = self.dataset_kwargs

        source_paths = [  # do we unify dataset_dir into dataset_paths?
            *(
                [dataset_kwargs["dataset_dir"]]
                if "dataset_dir" in dataset_kwargs
                else []
            ),
            *(
                dataset_kwargs["dataset_paths"]
                if "dataset_paths" in dataset_kwargs
                else []
            ),
        ]

        if "stratifying_labels" in transform_kwargs or has_stratifying_labels(
            source_paths
        ):
            prepare_index_with_label(
                source_paths=source_paths,
                index_dir=Path(dataset_kwargs["index_dir"]),
                stratifying_labels=transform_kwargs["stratifying_labels"]
                if "stratifying_labels" in transform_kwargs
                else None,
            )
        else:
            prepare_index(
                source_paths=source_paths, index_dir=Path(dataset_kwargs["index_dir"])
            )


class AnnDataItemLoader(BaseItemLoader):
    def __init__(
        self,
        dataset_paths: list[str] | list[Path],
        index_dir: str,
        expose_zeros: str | None,
        *args,
        **kwargs,
    ):
        self.dataset_paths = dataset_paths
        self.index_dir = index_dir
        self.buffer_chunk_index = None
        self.expose_zeros = expose_zeros
        self.ann_collection = None

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)

    def setup_anndata(self, *args, **kwargs):
        self.ann_collection = get_ann_collection(self.dataset_paths)
        self.var_names = self.ann_collection.var_names

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
        subset = self.ann_collection[index]
        X = csr_matrix(subset.X)
        obs_names = subset.obs_names
        return (X, obs_names)

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
        X, obs_names = self.buffer
        indptr = X.indptr
        start_index, end_index = indptr[item_index], indptr[item_index + 1]
        if self.expose_zeros == "all":
            genes = self.var_names
            expressions = X[item_index].astype("float").toarray().tolist()[0]
        elif self.expose_zeros is None:
            expressions = [float(i) for i in X.data[start_index:end_index]]
            genes = self.var_names[X.indices[start_index:end_index]]
        else:
            raise ValueError("expose_zeros must be 'all' or None")
        cell = obs_names[item_index]
        return cell, genes, expressions

    def delete(self, chunk_index: int, chunk_filepath: str) -> None:
        """Delete a chunk from the local filesystem."""
        pass

    def encode_data(
        self, data: list[bytes], sizes: list[int], flattened: list[Any]
    ) -> Any:
        raise RuntimeError("The method encode_data is not implemented.")


class AnnDataItemLoaderWithLabels(AnnDataItemLoader):
    all_label_columns: list[str] | None = None
    label_dict_path: str | None = None
    label_columns: list[str] | None = None
    regression_label_columns: list[str] | None = None

    def __init__(
        self,
        dataset_paths: list[str] | list[Path],
        index_dir: str,
        expose_zeros: str | None,
        label_dict_path: str | None = None,
        label_columns: list[str] | None = None,
        regression_label_columns: list[str] | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            dataset_paths=dataset_paths,
            index_dir=index_dir,
            expose_zeros=expose_zeros,
            *args,
            **kwargs,
        )
        self.label_dict_path = label_dict_path
        self.label_columns = label_columns if label_columns else []
        self.regression_label_columns = (
            regression_label_columns if regression_label_columns else []
        )
        self.all_label_columns = [*self.label_columns, *self.regression_label_columns]

    def get_label_dict(self):
        default_path = Path(self.index_dir) / "all_labels.json"
        if self.label_dict_path is None:
            self.label_dict_path = default_path
        categorical_labels = {l: self.get_sub_label_dict(l) for l in self.label_columns}
        regression_labels = {l: {0: 0} for l in self.regression_label_columns}
        return {**categorical_labels, **regression_labels}

    def get_sub_label_dict(self, label_column_name):
        self.setup_anndata()
        return load_and_update_all_labels_json(
            self.ann_collection,
            self.label_dict_path,
            label_column_name,
        )

    def setup_anndata(self, *args, **kwargs):
        if self.ann_collection is None:
            self.ann_collection = get_ann_collection(
                self.dataset_paths, join_obs="outer"
            )
            self.var_names = self.ann_collection.var_names

    def load_chunk(self, chunk_filepath: str) -> csr_matrix:
        with open(chunk_filepath, "rb") as file:
            index = IndexSerializer.deserialize(file.read())
        subset = self.ann_collection[index]
        X = csr_matrix(subset.X)
        obs_names = subset.obs_names
        obs = subset.obs
        return (X, obs_names, obs)

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
        X, obs_names, obs = self.buffer
        indptr = X.indptr
        start_index, end_index = indptr[item_index], indptr[item_index + 1]
        if self.expose_zeros == "all":
            genes = self.var_names
            expressions = X[item_index].astype("float").toarray().tolist()[0]
        elif self.expose_zeros is None:
            expressions = [float(i) for i in X.data[start_index:end_index]]
            genes = self.var_names[X.indices[start_index:end_index]]
        else:
            raise ValueError("expose_zeros must be 'all' or None")
        cellname = obs_names[item_index]
        metadata = {l: obs[l][cellname] for l in self.all_label_columns}
        metadata["cell_name"] = cellname
        return metadata, genes, expressions


def get_ann_collection(input_dir: list[str] | list[Path] | str, join_obs="inner"):
    """Getting AnnCollection instance from the h5ad files in the directory."""
    """
    Args:
    ----
        input_dir (str  |  Path): Path to folder in which h5ad files reside. or list
        join_obs (str) : Method about how obs metadata should be joined. (default: 'inner')
    """
    if isinstance(input_dir, list):
        input_dirs = [Path(i) for i in input_dir]
    else:
        input_dirs = [Path(input_dir)]
    files = [
        str(i)
        for input_dir in input_dirs
        for i in ([input_dir] if input_dir.is_file() else input_dir.glob("*.h5ad"))
    ]
    files.sort()  ##TBD ask Michael: do we really need this step??
    files = [sc.read(i, backed="r") for i in files]
    dataset = AnnCollection(
        files, join_obs=join_obs, join_vars="inner", label="dataset"
    )
    return dataset


def extract_stratifying_labels(collection):
    return [
        next(
            (col for col in ad.obs.columns if col.startswith("split_stratified_")),
            "split_stratified_",
        )
        for ad in collection.adatas
    ]


def get_split_index(collection, stratifying_labels=None):
    index_base = 0
    train_index_all = []
    dev_index_all = []
    test_index_all = []
    s_labels = (
        extract_stratifying_labels(collection=collection)
        if stratifying_labels is None
        else [f"split_stratified_{s}" for s in stratifying_labels]
    )
    for i in range(len(collection.adatas)):
        logger.info(f"length of {i}th anndata={len(collection.adatas[i])}")
        adata = collection.adatas[i].obs.reset_index()
        train_index_ = list(adata[adata[s_labels[i]] == "train"].index)
        train_index = [x + index_base for x in train_index_]
        dev_index_ = list(adata[adata[s_labels[i]] == "dev"].index)
        dev_index = [x + index_base for x in dev_index_]
        test_index_ = list(adata[adata[s_labels[i]] == "test"].index)
        test_index = [x + index_base for x in test_index_]
        logger.info(
            f"length of split (train, dev, test) {len(train_index)}, {len(dev_index)}, {len(test_index)}"
        )

        index_base = index_base + len(adata)

        train_index_all = train_index_all + train_index
        dev_index_all = dev_index_all + dev_index
        test_index_all = test_index_all + test_index
    return {
        "train": train_index_all,
        "dev": dev_index_all,
        "test": test_index_all,
    }


def has_stratifying_labels(source_paths):
    collection = get_ann_collection(input_dir=source_paths, join_obs="outer")
    return any(s for s in collection.obs.columns if s.startswith("split_stratified_"))


def prepare_index_with_label(source_paths, index_dir, stratifying_labels=None):
    collection = get_ann_collection(input_dir=source_paths, join_obs="outer")
    indexes = get_split_index(
        collection=collection, stratifying_labels=stratifying_labels
    )
    for split in ["train", "dev", "test"]:
        # contents are updated, if transform_data = True
        if os.path.exists(index_dir / split):
            shutil.rmtree(index_dir / split)
        build_index(output_dir=index_dir / split, index=indexes[split], chunk_size=5000)


def prepare_index(source_paths, index_dir):
    collection = get_ann_collection(input_dir=source_paths)
    n_cells = collection.n_obs
    n_train_split = int(n_cells * 0.9)
    # contents are updated, if transform_data = True
    if os.path.exists(index_dir / "train"):
        shutil.rmtree(index_dir / "train")
    build_index(
        output_dir=index_dir / "train",
        index=range(0, n_train_split),
        chunk_size=5000,
    )
    # contents are updated, if transform_data = True
    if os.path.exists(index_dir / "dev"):
        shutil.rmtree(index_dir / "dev")
    build_index(
        output_dir=index_dir / "dev",
        index=range(n_train_split, n_cells),
        chunk_size=5000,
    )
