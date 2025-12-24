#!/usr/bin/env python

import argparse
import os
import re
from functools import partial
from pathlib import Path

import pyarrow.parquet as pq
from litdata import optimize
from transformers import AutoTokenizer

from bmfm_targets.datasets.data_conversion import get_user_serializer


def convert_files(file_path, tokenizer, serializer, data_source="dnaseq_base"):
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
    if data_source == "hic":
        data = Parquet2LitDatasetHiC(file_path, tokenizer, serializer)
    elif data_source == "insulation":
        data = Parquet2LitDatasetInsulation(file_path, tokenizer, serializer)
    elif data_source == "chromatin":
        data = Parquet2LitDatasetChromatinProfile(file_path, tokenizer, serializer)
    elif data_source == "snp2trait":
        data = Parquet2LitDatasetSnp2Trait(file_path, tokenizer, serializer)
    elif data_source == "dnaseq_base":
        data = Parquet2LitDataset(file_path, tokenizer, serializer)
    else:
        raise ValueError(f"data source {data_source} not implemented.")
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
            batch_size=1000, use_threads=False, columns=["dna_sequence"]
        ):
            for text in batch.to_pandas()["dna_sequence"].tolist():
                text = text.strip()
                if set(text) == {"N"}:
                    continue
                yield self.serializer.serialize(
                    self.tokenizer.tokenize(text)
                    # self.tokenizer.tokenize(self.clean(text))
                )

    def __getstate__(self):
        dict = self.__dict__.copy()
        del dict["data"]
        return dict

    def __setstate__(self, d):
        self.__dict__ = d
        self.data


class Parquet2LitDatasetHiC(Parquet2LitDataset):
    """Dataset class to read a parquet file and yield a pair of tokenized DNA sequences with HiC contact score."""

    def __iter__(self):
        for batch in self.data.iter_batches(
            batch_size=1000,
            use_threads=False,
            columns=["dna_chunk1", "dna_chunk2", "hic_contact1"],
        ):
            df_dna_hic = batch.to_pandas()
            for i in range(len(df_dna_hic)):
                dna_chunk1, dna_chunk2, hic_contact1 = df_dna_hic.iloc[i, :]
                tokens1 = self.tokenizer.tokenize(dna_chunk1)
                tokens2 = self.tokenizer.tokenize(dna_chunk2)
                yield (
                    self.serializer.serialize(tokens1),
                    self.serializer.serialize(tokens2),
                    self.serializer.serialize([str(hic_contact1)]),
                )


class Parquet2LitDatasetInsulation(Parquet2LitDataset):
    """Dataset class to read a parquet file and yield a pair of tokenized DNA sequences with HiC contact score."""

    def __iter__(self):
        for batch in self.data.iter_batches(
            batch_size=1000,
            use_threads=False,
            columns=["dna_chunk", "insulation"],
        ):
            df_dna_hic = batch.to_pandas()
            for i in range(len(df_dna_hic)):
                dna_chunk, insulation = df_dna_hic.iloc[i, :]
                tokens = self.tokenizer.tokenize(dna_chunk)
                yield (
                    self.serializer.serialize(tokens),
                    self.serializer.serialize([str(insulation)]),
                )


class Parquet2LitDatasetChromatinProfile(Parquet2LitDataset):
    """Dataset class to read a parquet file and yield a pair of tokenized DNA sequences with 919 class labels coming from three groups: dnase, tf and histone."""

    header = [
        "dna_chunks",
        "combined_chromatin_dnase",
        "combined_chromatin_tf",
        "combined_chromatin_histone",
    ]

    def __iter__(self):
        for batch in self.data.iter_batches(
            batch_size=1000,
            use_threads=False,
            columns=self.header,
        ):
            df_dna_chromatin = batch.to_pandas()
            df_dna_chromatin = df_dna_chromatin.astype(str)
            assert df_dna_chromatin.shape[1] == 4
            for i in range(len(df_dna_chromatin)):
                dna_chunk = df_dna_chromatin.iloc[i, 0]
                tokens = self.tokenizer.tokenize(dna_chunk)
                to_return_fncs = [self.serializer.serialize(tokens)] + [
                    self.serializer.serialize([str(df_dna_chromatin.iloc[i, col_ind])])
                    for col_ind in range(1, df_dna_chromatin.shape[1])
                ]
                yield tuple(to_return_fncs)


class Parquet2LitDatasetSnp2Trait(Parquet2LitDataset):
    """Dataset class to read a parquet file and yield a pair of tokenized DNA sequences with 2000 class labels for diseases or traits."""

    header = ["dna_chunks", "combined_snp2trait_labels"]

    def __iter__(self):
        for batch in self.data.iter_batches(
            batch_size=1000,
            use_threads=False,
            columns=self.header,
        ):
            df_dna_snp2trait = batch.to_pandas()
            df_dna_snp2trait = df_dna_snp2trait.astype(str)
            assert df_dna_snp2trait.shape[1] == 2
            for i in range(len(df_dna_snp2trait)):
                dna_chunk = df_dna_snp2trait.iloc[i, 0]
                tokens = self.tokenizer.tokenize(dna_chunk)
                to_return_fncs = [self.serializer.serialize(tokens)] + [
                    self.serializer.serialize([str(df_dna_snp2trait.iloc[i, col_ind])])
                    for col_ind in range(1, df_dna_snp2trait.shape[1])
                ]
                yield tuple(to_return_fncs)


def convert_parquet_to_litdata(
    input_dir,
    output_dir,
    tokenizer,
    num_workers=4,
    chunk_bytes="128MB",
    data_source="dnaseq_base",
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
                data_source=data_source,
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
                data_source=data_source,
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
        "--hic",
        help="if the dataset is HiC dataset",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
    )
    parser.add_argument(
        "--insulation",
        help="if the dataset is insulation dataset from HiC",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
    )
    parser.add_argument(
        "--chromatin",
        help="if the dataset is Chromatin dataset",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
    )
    parser.add_argument(
        "--snp2trait",
        help="if the dataset is SNP2Trait dataset",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
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

    if args.hic:
        data_source = "hic"
    elif args.insulation:
        data_source = "insulation"
    elif args.chromatin:
        data_source = "chromatin"
    elif args.snp2trait:
        data_source = "snp2trait"
    else:
        data_source = "dnaseq_base"

    convert_parquet_to_litdata(
        input_dir=args.input,
        output_dir=args.output,
        tokenizer=tokenizer,
        num_workers=args.num_workers,
        chunk_bytes=args.chunk_bytes,
        data_source=data_source,
        use_split=args.splits,
    )
