#!/usr/bin/env python

import argparse
import os
import re
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from litdata import optimize
from transformers import AutoTokenizer

from bmfm_targets.datasets.data_conversion import get_user_serializer

snp_probability_matrix_path = (
    "/dccstor/bmfm-targets/data/omics/genome/snpdb/raw/matrix_snp_probability/"
)
chr_to_snp_flag = {}
for i in range(1, 23):
    chr_to_snp_flag["chr" + str(i)] = np.load(
        os.path.join(snp_probability_matrix_path, "snp_flag_chr" + str(i) + ".npy")
    )


def convert_files(file_path, tokenizer, serializer):
    """
    Convert a single file to a litdata dataset.

    Args:
    ----
        file_path (str): Path to the file.
        tokenizer (transformers.tokenizer): Tokenizer to use.
        serializer (ListSerializer): Serializer to use.
        hic (bool): if the input is HiC data
        chromatin (bool): if the input is Chromatin Profile data

    Yields:
    ------
        str: Tokenized DNA sequence.
    """
    data = Parquet2LitDataset(file_path, tokenizer, serializer)
    yield from data


class Parquet2LitDataset:
    """

    Dataset class to read a parquet file and yield tokenized DNA sequences.

    Args:
    ----
        file_path (str): Path to the parquet file.
        tokenizer (transformers.tokenizer): Tokenizer to use.
        serializer (ListSerializer): Serializer to use.
    """

    def __init__(self, file_path, tokenizer, serializer):
        """
        Initialize the dataset.

        Args:
        ----
            file_path (str): Path to the parquet file.
            tokenizer (transformers.tokenizer): Tokenizer to use.
            serializer (ListSerializer): Serializer to use.
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.data = pq.ParquetFile(file_path)
        self.serializer = serializer
        self.pattern = re.compile(r"#+|N+")

    def __len__(self):
        """
        Return the number of rows in the parquet file.

        Returns
        -------
            int: Number of row groups.
        """
        return self.data.num_row_groups

    def clean(self, text):
        """
        Clean the text.

        Args:
        ----
            text (str): Text to clean.

        Returns:
        -------
            str: Cleaned text.
        """
        text = text.strip()
        return self.pattern.sub(lambda match: match.group()[0], text)

    def __iter__(self):
        for batch in self.data.iter_batches(
            batch_size=1000,
            use_threads=False,
            columns=["dna_chunk", "chunk_index"],
        ):
            df_dna = batch.to_pandas()
            for i in range(len(df_dna)):
                dna_chunk, chunk_index = df_dna.iloc[i, :]
                tokens = self.tokenizer.tokenize(dna_chunk)
                ## it could be chr11_60 or rcchr11_60
                target_chr, index = chunk_index.split("_")
                left = int(index) * len(dna_chunk)
                right = left + len(dna_chunk)
                # token_labels = ['0'] # 1st [CLS] token is not snp
                token_labels = []
                if target_chr[:2] == "rc":
                    snp_flag = chr_to_snp_flag[target_chr[2:]]
                    for token in tokens:
                        if sum(snp_flag[(right - len(token)) : right]) > 0:
                            token_labels.append("1")
                        else:
                            token_labels.append("0")
                        right -= len(token)
                else:
                    snp_flag = chr_to_snp_flag[target_chr]
                    for token in tokens:
                        if sum(snp_flag[left : (left + len(token))]) > 0:
                            token_labels.append("1")
                        else:
                            token_labels.append("0")
                        left += len(token)
                # token_labels.append('0') # last [SEP] token is not snp
                yield (
                    self.serializer.serialize(tokens),
                    self.serializer.serialize(token_labels),
                )

    def __getstate__(self):
        dict = self.__dict__.copy()
        del dict["data"]
        return dict

    def __setstate__(self, d):
        self.__dict__ = d
        self.data


def convert_parquet_to_litdata(
    input_dir,
    output_dir,
    tokenizer,
    num_workers=4,
    chunk_bytes="128MB",
    use_split=True,
):
    """
    Convert a directory of parquet files to a litdata dataset.

    Args:
    ----
        input_dir (str): Path to the input directory.
        output_dir (str): Path to the output directory.
        tokenizer (str): Tokenizer to use.
        num_workers (int): Number of workers.
        chunk_bytes (str): Size of chunks in bytes (example: 128MB).
        hic (bool): if the input is HiC data
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    list_serializer = get_user_serializer(list[str])
    if not use_split:
        input_names = [i.as_posix() for i in (input_dir).glob("*.parquet")]
        optimize(
            fn=partial(
                convert_files,
                tokenizer=tokenizer,
                serializer=list_serializer,
            ),
            inputs=input_names,
            output_dir=output_dir.as_posix(),
            num_workers=num_workers,
            chunk_bytes=chunk_bytes,
        )
        return

    for split in ("train", "dev", "test"):
        input_names = [i.as_posix() for i in (input_dir / split).glob("*.parquet")]
        optimize(
            fn=partial(
                convert_files,
                tokenizer=tokenizer,
                serializer=list_serializer,
            ),
            inputs=input_names,
            output_dir=(output_dir / split).as_posix(),
            num_workers=num_workers,
            chunk_bytes=chunk_bytes,
        )


def path_type_exists(name):
    path = Path(name)
    if path.exists():
        raise FileExistsError(f"The directory {name} already exists.")
    os.makedirs(name)
    return path


def path_type_not_exists(name):
    path = Path(name)
    if not path.exists():
        raise FileNotFoundError(f"The directory {name} does not exists.")
    return path


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        type=path_type_not_exists,
        help="Input directory with parquet dataset (train, dev, test).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=path_type_exists,
        help="Output directory for litdata dataset (train, dev, test).",
    )

    parser.add_argument(
        "-tz",
        "--tokenizer",
        help="Path to tokenizer.",
        type=str,
        required=False,
        default="zhihan1996/DNABERT-2-117M",
    )
    parser.add_argument(
        "-nw", "--num_workers", help="Number of workers.", type=int, default=2
    )
    parser.add_argument(
        "-cb",
        "--chunk_bytes",
        help="Size of chunks in bytes (example: 128MB).",
        type=str,
        default="128MB",
    )
    parser.add_argument(
        "--splits",
        help="if the input dir has internal splits",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=True,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        do_lower_case=False,
        tokenize_chinese_chars=False,
        clean_text=False,
        strip_accents=None,
    )

    convert_parquet_to_litdata(
        input_dir=args.input,
        output_dir=args.output,
        tokenizer=tokenizer,
        num_workers=args.num_workers,
        chunk_bytes=args.chunk_bytes,
        use_split=args.splits,
    )
