import json
import os
from collections.abc import Iterable
from typing import Any

from bmfm_targets.datasets.data_conversion.serializers import IndexSerializer


def push_index(
    output_dir: str,
    filename: str,
    index: list,
):
    data = IndexSerializer.serialize(index)
    with open(os.path.join(output_dir, filename), "wb") as file:
        file.write(data)


def get_chunk_filename(file_index):
    return f"chunk_index{file_index:08}.bin"


def create_chunk_index(
    output_dir: str,
    index: Iterable,
    chunk_size: int,
):
    """
    Original litdata chunks have serialized items in files with bin extension.
    We save indices of the items instead to the bin files and read data from the database.
    The function creates bin files for litdata that have indices of items in the chunks.
    """
    os.makedirs(output_dir, exist_ok=False)

    chunk_index = []
    file_index = 0
    for i in index:
        chunk_index.append(i)
        if len(chunk_index) == chunk_size:
            push_index(
                output_dir, filename=get_chunk_filename(file_index), index=chunk_index
            )
            file_index += 1
            chunk_index = []

    if chunk_index:
        push_index(
            output_dir, filename=get_chunk_filename(file_index), index=chunk_index
        )

    n_full_files = file_index
    last_file_size = len(chunk_index)
    return n_full_files, last_file_size


def create_litdata_index(
    output_dir: str,
    chunk_size: int,
    n_full_files: int,
    last_file_size: int,
    label_dict: dict[str, int],
    custom_config: dict[str, Any],
):
    chunks = []
    for i in range(n_full_files):
        chunk = {
            "chunk_bytes": 1,
            "chunk_size": chunk_size,
            "dim": None,
            "filename": get_chunk_filename(i),
        }
        chunks.append(chunk)

    if last_file_size:
        chunk = {
            "chunk_bytes": 1,
            "chunk_size": last_file_size,
            "dim": None,
            "filename": get_chunk_filename(file_index=n_full_files),
        }
        chunks.append(chunk)

    config = {
        "chunk_bytes": None,
        "chunk_size": chunk_size,
        "compression": None,
        "data_format": ["bytes"],
        "data_spec": '[1, {"type": null, "context": null, "children_spec": []}]',
        "tiledb": {},
    }

    if label_dict:
        config["tiledb"]["label_dict"] = label_dict

    if custom_config:
        config.update(custom_config)
    specs = {"chunks": chunks, "config": config}

    with open(os.path.join(output_dir, "index.json"), "w") as file:
        json.dump(specs, file)


def build_index(
    output_dir: str,
    index: Iterable,
    label_dict: dict[str, int] | None = None,
    chunk_size: int = 5000,
    custom_config: dict[str, Any] = None,
):
    n_full_files, last_file_size = create_chunk_index(output_dir, index, chunk_size)
    create_litdata_index(
        output_dir, chunk_size, n_full_files, last_file_size, label_dict, custom_config
    )
