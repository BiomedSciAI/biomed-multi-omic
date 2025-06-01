"""
Script to split the input file containing DNA / DNA with SNP sequences into train, dev, and test sets and write them to parquet files.
Run the script only if you have a new version or variation of the input file.
Contact bdand if you are not sure or have any questions.
"""

import glob
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
    data_frame = pd.DataFrame(data, columns=["dna_chunk", "insulation"])
    # ["dna_sequence"] + ['label_' + str(x) for x in range(1,len(data[0]))])
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


def split(
    input_path: str,
    output_path: str,
    chunk_size: int = 1000,
    train_ratio: float = 0.7,
    dev_ratio: float = 0.2,
    test_ratio: float = 0.1,
):
    """

    Split the input file(s) into train, dev, and test sets and write them to JSONL files.


    Args:
    ----
        input_path (str): Path to the input directory.
        output_path (str): Path to the output directory.
        chunk_size (int): Size of the chunk.
        train_ratio (float): Ratio of the train set.
        dev_ratio (float): Ratio of the dev set.
    """
    os.makedirs(output_path, exist_ok=True)
    assert round((train_ratio + dev_ratio + test_ratio), 2) == 1.0
    for split in ["train", "dev", "test"]:
        os.makedirs(os.path.join(output_path, split), exist_ok=True)
    input_files = sorted(glob.glob(os.path.join(input_path, "*")))
    split_id = 0
    for input_file in input_files:
        with open(input_file) as f:
            while True:
                lines = get_chunk_lines(chunk_size, f)
                if not lines:
                    break
                print(split_id, len(lines))
                random.shuffle(lines)
                num_lines = len(lines)
                train_size = int(num_lines * train_ratio)
                dev_size = int(num_lines * dev_ratio)
                write_parquet(
                    lines[:train_size],
                    os.path.join(
                        output_path, "train", "train" + str(split_id) + ".parquet"
                    ),
                )
                write_parquet(
                    lines[train_size : (train_size + dev_size)],
                    os.path.join(
                        output_path, "dev", "dev" + str(split_id) + ".parquet"
                    ),
                )
                write_parquet(
                    lines[(train_size + dev_size) :],
                    os.path.join(
                        output_path, "test", "test" + str(split_id) + ".parquet"
                    ),
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
        "-cs",
        "--chunk_size",
        type=int,
        default=10000,
        help="Chunk size, default is 10000, number of dna sequences per chunk.",
    )
    parser.add_argument(
        "-tr",
        "--train_ratio",
        type=float,
        default=0.7,
        help="Train ratio, default is 0.7.",
    )
    parser.add_argument(
        "-vs", "--val_ratio", type=float, default=0.2, help="val ratio, default is 0.2."
    )
    parser.add_argument(
        "-te",
        "--test_ratio",
        type=float,
        default=0.1,
        help="Test ratio, default is 0.1.",
    )
    args = parser.parse_args()
    split(
        args.input_path,
        args.output_path,
        args.chunk_size,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
    )
