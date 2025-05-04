"""
Script to split the input file containing DNA / DNA with SNP sequences into train, dev, and test sets and write them to parquet files.
Run the script only if you have a new version or variation of the input file.
Contact bdand if you are not sure or have any questions.
"""

import os
import random
from argparse import ArgumentParser

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def write_parquet(data: list[str], output_file: str) -> None:
    """
    Write data to a parquet_file.


    Args:
    ----
        data (list[str]): List of lines.
        output_file (str): Path to the output file.
    """
    header = [
        "dna_chunks",
        "combined_chromatin_dnase",
        "combined_chromatin_tf",
        "combined_chromatin_histone",
    ]
    data_frame = pd.DataFrame(data, columns=header)
    table = pa.Table.from_pandas(data_frame)
    pq.write_table(table, output_file)


def get_chunk_lines(chunk_size, f):
    """
    Get lines for a chunk.


    Args:
    ----
        chunk_size (int): Size of the chunk.
        f (file): File object.


    Returns:
    -------
        list: List of lines.
    """
    lines = []
    for _ in range(chunk_size):
        try:
            line = next(f)
            lines.append(line.strip().split(","))
        except StopIteration:
            break
    return lines


def split_files_only(
    input_path: str,
    output_path: str,
    input_filename: str,
    chunk_size: int = 1000,
):
    """

    Split the input file(s) into train, dev, and test sets and write them to JSONL files.


    Args:
    ----
        input_path (str): Path to the input directory.
        output_path (str): Path to the output directory.
        chunk_size (int): Size of the chunk.
    """
    # os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path), exist_ok=True)

    # input_files = sorted(glob.glob(os.path.join(input_path, "*")))
    split_id = 0
    input_file = os.path.join(input_path, input_filename)
    with open(input_file) as f:
        while True:
            lines = get_chunk_lines(chunk_size, f)
            if not lines:
                break
            print(split_id, len(lines))
            random.shuffle(lines)

            write_parquet(
                lines,
                os.path.join(output_path, input_filename + str(split_id) + ".parquet"),
            )
            split_id += 1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        required=True,
        help="Path to the input directory containing DNA sequence files.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=True,
        help="Output directory where the parquet files will be stored.",
    )
    parser.add_argument(
        "-in_name",
        "--input_filename",
        type=str,
        default=None,
        help="Output directory where the parquet files will be stored.",
    )
    parser.add_argument(
        "-cs",
        "--chunk_size",
        type=int,
        default=10000,
        help="Chunk size, default is 10000, number of dna sequences per chunk.",
    )

    args = parser.parse_args()
    print(args)
    if args.input_filename:
        split_files_only(
            args.input_path,
            args.output_path,
            args.input_filename,
            args.chunk_size,
        )
    else:
        for filename in ["train.csv", "dev.csv", "test.csv"]:
            split_files_only(
                args.input_path,
                os.path.join(args.output_path, filename[:-4]),
                filename,
                args.chunk_size,
            )
