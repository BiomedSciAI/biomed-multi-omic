import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from bmfm_targets.datasets.SNPdb.tabix_converter import (
    extract_chr_seq_and_len,
    sample_variant,
)

SNPDB_RESOURCES_PATH = "/dccstor/bmfm-targets/data/omics/genome/snpdb/raw/resources/"
fasta_path = SNPDB_RESOURCES_PATH + "GCA_000001405.15_GRCh38_no_alt_analysis_set.fna"
variation_matrix_path = (
    "/dccstor/bmfm-targets/data/omics/genome/snpdb/raw/matrix_snp_probability/"
)


def sample_insulation_chunks(
    sequence: str,
    insulation_path: str | Path,
    output_path: str | Path,
    target_chr="chr22",
    hic_resolution=1000,
    exclude_N_islands=True,
):
    """
    Read a insulation table (from Hi-C) and DNA sequence, sequentially sample DNA chunks with insulation scores.

    Args:
    ----
        sequence (str): DNA sequence (variation-encoded or reference) of the target chromosome
        insulation_path (str): insulation score table
        output_path (str): output file
        target_chr (str): the target chromosome to be processed
        hic_resolution (int): the basic unit of Hi-C contact
        exclude_N_islands (bool): if all-N chunks will be excluded or not

    Returns:
    -------
        An output txt file with DNA chunks with insulation scores
    """
    df_insulation = pd.read_csv(insulation_path)
    df_insulation = df_insulation.loc[
        df_insulation["chrom"] == target_chr,
        ["start", "log2_insulation_score_" + str(hic_resolution * 10)],
    ]
    seq_len = len(sequence)
    num_chunks = (
        (seq_len // hic_resolution) + 1
        if seq_len % hic_resolution
        else seq_len // hic_resolution
    )
    with open(output_path, "w") as f:
        for i in range(num_chunks):
            label = df_insulation[
                "log2_insulation_score_" + str(hic_resolution * 10)
            ].iloc[i]
            if np.isnan(label):
                continue
            chunk = sequence[(i * hic_resolution) : ((i + 1) * hic_resolution)]
            if exclude_N_islands and set(chunk) == {"N"}:
                continue
            label = [label.round(2).astype("str")]
            f.write(",".join([chunk] + label) + "\n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-chr",
        "--chromosome",
        help="The chromosome to be processed",
        type=str,
        required=False,
        default="chr21",
    )
    parser.add_argument(
        "--insulation_path",
        help="The insulation score table",
        type=str,
        required=False,
        default="/dccstor/bmfm-targets1/data/omics/genome/hic/insulation_hepg2_10kb.csv",
    )
    parser.add_argument(
        "--hic_resolution",
        help="The resolution for Hi-C data",
        type=int,
        required=False,
        default=1000,
    )
    parser.add_argument(
        "--process_ref_genome",
        help="create samples for the reference genome",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
    )
    parser.add_argument(
        "--process_snp_aware_genome",
        help="create samples for the SNP-aware genome",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=True,
    )
    parser.add_argument(
        "--output_variation_biallele_path",
        help="output directory for biallele-encoded samples",
        type=str,
        required=False,
        default="/dccstor/bmfm-targets/data/omics/genome/snpdb/raw/pretraining_data/biallele_insulation_hepg2_1kb/",
    )
    parser.add_argument(
        "--output_reference_genome_path",
        help="output directory for reference genome samples",
        type=str,
        required=False,
        default="/dccstor/bmfm-targets/data/omics/genome/snpdb/raw/pretraining_data/ref_insulation_hepg2_1kb/",
    )
    args = parser.parse_args()
    return args


def main(args):
    chromosome = args.chromosome
    hic_resolution = args.hic_resolution
    insulation_path = args.insulation_path
    ## 1. reference genome
    if args.process_ref_genome:
        output_path = args.output_reference_genome_path
        os.makedirs(output_path, exist_ok=True)
        chr_to_seq, _ = extract_chr_seq_and_len(fasta_path, ">")
        sequence = chr_to_seq[chromosome]
        sample_insulation_chunks(
            sequence,
            insulation_path,
            output_path + "sample_" + chromosome + ".txt",
            target_chr=chromosome,
            hic_resolution=hic_resolution,
            exclude_N_islands=True,
        )
    ## 2. biallele encoded genome
    if args.process_snp_aware_genome:
        output_path = args.output_variation_biallele_path
        os.makedirs(output_path, exist_ok=True)
        # load the variation sparse matrix
        snp_probability_matrix = sparse.load_npz(
            variation_matrix_path + "snp_prob_" + chromosome + ".npz"
        )
        # sample and encode variants in the biallele fashion
        sequence = sample_variant(
            snp_probability_matrix,
            replacement=False,
        )
        sample_insulation_chunks(
            sequence,
            insulation_path,
            output_path + "sample_" + chromosome + ".txt",
            target_chr=chromosome,
            hic_resolution=hic_resolution,
            exclude_N_islands=True,
        )


if __name__ == "__main__":
    args = get_args()
    main(args)
