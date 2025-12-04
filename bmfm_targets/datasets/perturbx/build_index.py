import os
from collections.abc import Iterable
from pathlib import Path

import anndata as ad
import h5py
import numpy as np

from bmfm_targets.datasets.data_conversion.litdata_indexing import build_index
from bmfm_targets.datasets.data_conversion.serializers import IndexSerializer

SHUFFLE_PREFIX = "shuffled_"


def get_shuffled_filename(h5ad_path):
    in_path = Path(h5ad_path)
    return in_path.with_name(f"{SHUFFLE_PREFIX}{in_path.name}")


def shuffle_anndata(
    h5ad_path: str,
    target_gene_column: str,
    control_label: str,
    split_column: str | None = None,
) -> str:
    """
    Load an AnnData .h5ad file, shuffle rows so that 'control' rows
    from obs[target_gene_column] come first in random order, followed by
    all other rows in random order. If split_label is defined, splits are grouped into sections.
    Save result to a new file with 'shuffled_' prefix.

    Agrs
    ----------
    h5ad_path : str
        Path to input .h5ad file.

    target_gene_column: str
    control_label: str
    split_label: str
        Label of obs with split string (e.g., "train" or "dev")

    Returns
    -------
    str
        Path to the saved shuffled .h5ad file.
    """
    adata = ad.read_h5ad(h5ad_path)

    is_control = adata.obs[target_gene_column] == control_label

    if not np.any(is_control):
        raise ValueError(
            f"Invalid control label '{control_label}' or no control samples in the file."
        )

    rng = np.random.default_rng()
    control_idx = np.where(is_control)[0]
    rng.shuffle(control_idx)

    if not split_column:
        other_idx = np.where(~is_control)[0]
        rng.shuffle(other_idx)
        other_idx = [other_idx]
    else:
        labels = adata.obs[split_column].dropna().unique()
        other_idx = []
        for l in labels:
            idx = np.where((~is_control) & (adata.obs[split_column] == l))[0]
            rng.shuffle(idx)
            other_idx.append(idx)

    new_order = np.concatenate([control_idx] + other_idx)
    adata_shuffled = adata[new_order].copy()

    out_path = get_shuffled_filename(h5ad_path)
    adata_shuffled.write_h5ad(out_path)
    return str(out_path)


def build_perturbx_index(
    output_dir: str,
    target_gene_column: str,
    control_label: str,
    control_index: Iterable,
    split_index: Iterable,
    chunk_size: int = 3000,
    use_hdf_control_index: bool = True,
):
    """
    Args:
    ----
    _______
    use_hdf_control_index (bool): if true, use hdf format to save index of control, custom serialization otherwise.
    """
    extra_config = {
        "ptb": {
            "control": {"chunk_size": chunk_size},
            "target_gene_column": target_gene_column,
        }
    }

    build_index(
        output_dir, split_index, chunk_size=chunk_size, custom_config=extra_config
    )
    control_index = list(control_index)
    chunk_size = min(chunk_size, len(control_index))
    control_index_filename = "control_index.bin"
    if use_hdf_control_index:
        with h5py.File(os.path.join(output_dir, control_index_filename), "w") as f:
            dataset = f.create_dataset(
                "control_index",
                data=np.array(list(control_index), dtype=np.int32),
                chunks=(chunk_size,),
                compression="gzip",
            )
    else:
        data = IndexSerializer.serialize(control_index + [chunk_size])
        with open(os.path.join(output_dir, control_index_filename), "wb") as file:
            file.write(data)
