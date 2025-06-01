import argparse
import glob
import os
from pathlib import Path

import hicstraw
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

dna_to_complement = {
    "A": "T",
    "C": "G",
    "G": "C",
    "T": "A",
    "N": "N",
}


def fixed_distance_sample_hic_contact_chunks(
    sequence: str,
    hic_path: str | Path,
    output_path: str | Path,
    target_chr="chr22",
    hic_resolution=1000,
    stride=1,
    complete_chunk=False,
    exclude_N_islands=True,
    binarize_label=True,
):
    """
    Read a Hi-C file and DNA sequence, sequentially sample pairs of DNA chunks, and their contact scores from Hi-C.

    Args:
    ----
        sequence (str): DNA sequence (variation-encoded or reference) of the target chromosome
        hic_path (str): input Hi-C file
        output_path (str): output file
        target_chr (str): the target chromosome to be processed
        hic_resolution (int): the basic unit of Hi-C contact
        stride (int): stride * resolution is the distances between chunks; stride = 1 means a pair of connected chunks
        complete_chunk (bool): if true, the pair of chunks and sequences between them will be combined as one complete chunks. For example, when stride = 3 and input [chunk1][chunk2][chunk3][chunk4], it returns [[chunk1], [chunk4], hic_contact] if complete_chunk = False, it returns [[chunk1][chunk2][chunk3][chunk4], hic_contact] if complete_chunk = True
        exclude_N_islands (bool): if all-N chunks will be excluded or not

    Returns:
    -------
        Number of samples and the total number of contacts
        (the sample of DNA chunks and pairwise contact scores will be written in the output file).
    """
    hic = hicstraw.HiCFile(hic_path)
    mzd = hic.getMatrixZoomData(
        target_chr[3:], target_chr[3:], "oe", "KR", "BP", hic_resolution
    )
    sequence_len = len(sequence)
    number_of_samples = 0
    number_of_contacts = 0
    with open(output_path, "w") as f:
        for left in range(0, sequence_len - hic_resolution * stride, hic_resolution):
            ## for the last chunk, hicstraw return 0 if out of range; the return sequence could be shorter
            right = left + hic_resolution * stride
            if complete_chunk:
                dna_chunks = [sequence[left : (right + hic_resolution)]]
            else:
                dna_chunks = [
                    sequence[x : (x + hic_resolution)]
                    for x in range(
                        left, right + hic_resolution, hic_resolution * stride
                    )
                ]
            nucleotide_set = [list(set(x)) for x in dna_chunks]
            nucleotide_set = {x for xx in nucleotide_set for x in xx}
            if exclude_N_islands and (nucleotide_set == {"N"}):
                continue
            else:
                label = np.array(mzd.getRecordsAsMatrix(left, left, right, right)[0, 0])
                number_of_samples += 1
                number_of_contacts += label > 0
                if binarize_label:
                    label = [(label != 0).astype("int").astype("str")]
                else:
                    label = [label.round(2).astype("str")]
                f.write(",".join(dna_chunks + label) + "\n")
    return number_of_samples, number_of_contacts


def sequential_sample_hic_contact_chunks(
    sequence: str,
    hic_path: str | Path,
    output_path: str | Path,
    target_chr="chr22",
    hic_resolution=1000,
    number_of_chunks=10,
    exclude_N_islands=True,
    binarize_label=True,
):
    """
    Read a Hi-C file and DNA sequence, sequentially create samples of multiple DNA chunks separated by "#", and their pairwise contact scores from Hi-C.

    Args:
    ----
        sequence (str): DNA sequence (variation-encoded or reference) of the target chromosome
        hic_path (str): input Hi-C file
        output_path (str): output file
        target_chr (str): the target chromosome to be processed
        hic_resolution (int): the basic unit of Hi-C contact
        number_of_chunks (int): the number of chunks within a sample

    Returns:
    -------
        Number of samples and the total number of contacts
        (the sample of DNA chunks and pairwise contact scores will be written in the output file).
    """
    hic = hicstraw.HiCFile(hic_path)
    mzd = hic.getMatrixZoomData(
        target_chr[3:], target_chr[3:], "oe", "KR", "BP", hic_resolution
    )
    sequence_len = len(sequence)
    sample_len = hic_resolution * number_of_chunks
    number_of_samples = 0
    number_of_contacts = 0
    with open(output_path, "w") as f:
        for i in range(0, sequence_len // sample_len):
            left = i * sample_len
            right = left + hic_resolution * (number_of_chunks - 1)
            contact_matrix = mzd.getRecordsAsMatrix(left, right, left, right)
            contact_count = np.sum(np.triu(contact_matrix, k=1) != 0)
            ## if any inter-chunk contact exists, create a sample of multiple chunks
            if contact_count > 0:
                label = contact_matrix[np.triu_indices(number_of_chunks, k=1)]
                dna_chunks = [
                    sequence[x : (x + hic_resolution)]
                    for x in range(left, right + hic_resolution, hic_resolution)
                ]
                # here I use "#" as [SEP] for now
                sample_sequence = "#".join(dna_chunks)
                f.write(
                    ",".join(dna_chunks + label.round(2).astype("str").tolist()) + "\n"
                )
                number_of_samples += 1
                number_of_contacts += contact_count
    return number_of_samples, number_of_contacts


def random_sample_hic_contact_chunks(
    sequence: str,
    hic_path: str | Path,
    output_path: str | Path,
    target_chr="chr22",
    hic_resolution=1000,
    number_of_chunks=10,
    shifted_bin=0,
    sample_ratio=1.0,
    seed=0,
    exclude_N_islands=True,
    binarize_label=True,
):
    """
    Read a Hi-C file and DNA sequence, randomly create samples of multiple DNA chunks separated by "#", and their pairwise contact scores from Hi-C.

    Args:
    ----
        sequence (str): DNA sequence (variation-encoded or reference) of the target chromosome
        hic_path (str): input Hi-C file
        output_path (str): output file
        target_chr (str): the target chromosome to be processed
        hic_resolution (int): the basic unit of Hi-C contact
        number_of_chunks (int): the number of chunks within a sample
        shifted_bin (int): the shifted bins if not start with 0
        sample_ratio (float): the sample_ratio of the chromosome/TAD - 1.0 means the total length of samples is roughly the same length of the chromosome/TAD. Higher means more samples
        exclude_N_islands (boolean): if True, all-N bins will be excluded from sampling
        binarize_label (boolean): if True, convert label into 0/1 as a classification problem

    Returns:
    -------
        Number of samples and the total number of contacts
        (the sample of DNA chunks and pairwise contact scores will be written in the output file).
    """
    hic = hicstraw.HiCFile(hic_path)
    mzd = hic.getMatrixZoomData(
        target_chr[3:], target_chr[3:], "oe", "KR", "BP", hic_resolution
    )
    sequence_len = len(sequence)
    maximum_position = sequence_len // hic_resolution
    if exclude_N_islands:
        all_dna_chunks = [
            sequence[(x * hic_resolution) : (x * hic_resolution + hic_resolution)]
            for x in range(maximum_position)
        ]
        all_bins = np.where([x != {"N"} for x in [set(x) for x in all_dna_chunks]])[0]
    else:
        all_bins = range(maximum_position)
    number_of_samples = int(
        sequence_len // (hic_resolution * number_of_chunks) * sample_ratio
    )
    np.random.seed(seed)
    number_of_contacts = 0
    with open(output_path, "a") as f:
        for _ in range(0, number_of_samples):
            label = []
            bins = np.random.choice(all_bins, number_of_chunks, replace=False)
            for i in range(number_of_chunks - 1):
                for j in range(i + 1, number_of_chunks):
                    ## shift bins for hic data
                    left = (bins[i] + shifted_bin) * hic_resolution
                    right = (bins[j] + shifted_bin) * hic_resolution
                    label.append(mzd.getRecordsAsMatrix(left, left, right, right)[0, 0])
            label = np.array(label)
            dna_chunks = [
                sequence[(x * hic_resolution) : (x * hic_resolution + hic_resolution)]
                for x in bins.tolist()
            ]
            if binarize_label:
                f.write(
                    ",".join(
                        dna_chunks + (label != 0).astype("int").astype("str").tolist()
                    )
                    + "\n"
                )
            else:
                f.write(
                    ",".join(dna_chunks + label.round(2).astype("str").tolist()) + "\n"
                )
            number_of_contacts += sum(label != 0)
    return number_of_samples, number_of_contacts


def concatenate_hic_contact_chunks(
    input_path: str | Path,
    output_path: str | Path,
    augmentation=False,
):
    """
    Concatenate DNA chunks if they are in contact based on HiC.
    The input samples are pre-generated from HiC. With augmentation, it generates 7 versions:
    rc = reverse complement; r = reverse only
    (1) seq1 + seq2
    (2) seq1_r + seq2
    (3) seq1 + seq2_r
    (4) seq1_r + seq2_r
    (5) seq1_rc + seq2
    (6) seq1 + seq2_rc
    (7) seq1_rc + seq1_rc
    There could be other augmentations to be added including seq1 + seq1, random shifting seq1 and seq2.
    """
    os.makedirs(output_path, exist_ok=True)
    for input_file in sorted(glob.glob(os.path.join(input_path + "*"))):
        output_file = open(os.path.join(output_path, input_file.split("/")[-1]), "w")
        with open(input_file) as f:
            for line in f:
                seq1, seq2, label = line.strip().split(",")
                if label == "1":
                    output_file.write(seq1 + seq2 + "\n")
                    if augmentation:
                        seq1_r = seq1[::-1]
                        seq2_r = seq2[::-1]
                        seq1_rc = "".join([dna_to_complement[x] for x in seq1_r])
                        seq2_rc = "".join([dna_to_complement[x] for x in seq2_r])
                        output_file.write(seq1_r + seq2 + "\n")
                        output_file.write(seq1 + seq2_r + "\n")
                        output_file.write(seq1_r + seq2_r + "\n")
                        output_file.write(seq1_rc + seq2 + "\n")
                        output_file.write(seq1 + seq2_rc + "\n")
                        output_file.write(seq1_rc + seq2_rc + "\n")
        output_file.close()


def cut_sequence_by_tad_bin(sequence, target_chr, tad_path, hic_resolution=1000):
    df_tad = pd.read_csv(tad_path, header=None)
    df_tad.columns = ["chromosome", "start", "end"]
    df_tad = df_tad[df_tad.chromosome == target_chr]
    df_tad = df_tad.sort_values(by="start")
    start_all = df_tad.start.tolist()
    end_all = df_tad.end.tolist()
    ## start and end position at the hic bin level
    start_all = [x // hic_resolution * hic_resolution for x in start_all]
    end_all = [x // hic_resolution * hic_resolution for x in end_all]
    sub_sequences = []
    for start, end in zip(start_all, end_all):
        sub_sequences.append(sequence[start:end])
    return sub_sequences, start_all


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_strategy",
        help="sample DNA chunks sequentially or randomly",
        type=str,
        required=False,
        default="fixed_distance",
    )
    parser.add_argument(
        "-chr",
        "--chromosome",
        help="The chromosome to be processed",
        type=str,
        required=False,
        default="chr21",
    )
    parser.add_argument(
        "--hic_path",
        help="The Hi-C file",
        type=str,
        required=False,
        default="/dccstor/bmfm-targets/data/omics/genome/hic/4DNFICSTCJQZ.hic",
    )
    parser.add_argument(
        "--hic_resolution",
        help="The resolution for Hi-C data",
        type=int,
        required=False,
        default=1000,
    )
    parser.add_argument(
        "--number_of_chunks",
        help="The number of chunks within a sample",
        type=int,
        required=False,
        default=10,
    )
    parser.add_argument(
        "--stride",
        help="The stride for fixed distance sampling",
        type=int,
        required=False,
        default=1,
    )
    parser.add_argument(
        "--complete_chunk",
        help="if return a single long chunk where the pairs of chunks are at both end",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
    )
    parser.add_argument(
        "--binarize_label",
        help="if binarize the label into a classification problem",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=True,
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
        default="/dccstor/bmfm-targets/data/omics/genome/snpdb/raw/pretraining_data/biallele_hic_hepg2_1kb_10/",
    )
    parser.add_argument(
        "--output_reference_genome_path",
        help="output directory for reference genome samples",
        type=str,
        required=False,
        default="/dccstor/bmfm-targets/data/omics/genome/snpdb/raw/pretraining_data/ref_hic_hepg2_1kb_10/",
    )
    parser.add_argument(
        "--tad_path",
        help="input TAD file",
        type=str,
        required=False,
        default=None,
    )
    args = parser.parse_args()
    return args


def main_fixed_distance(args):
    chromosome = args.chromosome
    hic_resolution = args.hic_resolution
    stride = args.stride
    hic_path = args.hic_path
    ## 1. reference genome
    if args.process_ref_genome:
        output_path = args.output_reference_genome_path
        os.makedirs(output_path, exist_ok=True)
        chr_to_seq, _ = extract_chr_seq_and_len(fasta_path, ">")
        sequence = chr_to_seq[chromosome]
        (
            number_of_samples,
            number_of_contacts,
        ) = fixed_distance_sample_hic_contact_chunks(
            sequence,
            hic_path,
            output_path + "sample_" + chromosome + ".txt",
            target_chr=chromosome,
            hic_resolution=hic_resolution,
            stride=stride,
            complete_chunk=args.complete_chunk,
            exclude_N_islands=True,
            binarize_label=args.binarize_label,
        )
        print(
            f"stride={stride} {chromosome} has {number_of_samples} samples and {number_of_contacts} non-zero contacts ({number_of_contacts/number_of_samples*100:.2f}%)"
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
        (
            number_of_samples,
            number_of_contacts,
        ) = fixed_distance_sample_hic_contact_chunks(
            sequence,
            hic_path,
            output_path + "sample_" + chromosome + ".txt",
            target_chr=chromosome,
            hic_resolution=hic_resolution,
            stride=stride,
            complete_chunk=args.complete_chunk,
            exclude_N_islands=True,
            binarize_label=args.binarize_label,
        )
        print(
            f"stride={stride} {chromosome} has {number_of_samples} samples and {number_of_contacts} non-zero contacts ({number_of_contacts/number_of_samples*100:.2f}%)"
        )


def main_random(args):
    chromosome = args.chromosome
    hic_resolution = args.hic_resolution
    number_of_chunks = args.number_of_chunks
    sample_function = random_sample_hic_contact_chunks
    # load the variation sparse matrix
    snp_probability_matrix = sparse.load_npz(
        variation_matrix_path + "snp_prob_" + chromosome + ".npz"
    )
    # sample and encode variants in the biallele fashion
    encoded_seq = sample_variant(
        snp_probability_matrix,
        replacement=False,
    )
    ## 1. biallele encoded genome
    os.makedirs(args.output_variation_biallele_path, exist_ok=True)
    # cut into TAD sub sequences if TAD file is provided
    if args.tad_path:
        sub_seqs, starts = cut_sequence_by_tad_bin(
            encoded_seq,
            target_chr=chromosome,
            tad_path=args.tad_path,
            hic_resolution=hic_resolution,
        )
    else:
        sub_seqs = [encoded_seq]
        starts = [0]
    total_number_of_samples = 0
    total_number_of_contacts = 0
    for sequence, start in zip(sub_seqs, starts):
        if len(sequence) < hic_resolution * number_of_chunks:
            print(
                "Exclulde a short TAD of %d bp starts at %d " % (len(sequence), start)
            )
        else:
            number_of_samples, number_of_contacts = sample_function(
                sequence=sequence,
                hic_path=args.hic_path,
                output_path=args.output_variation_biallele_path
                + "sample_"
                + chromosome
                + ".txt",
                target_chr=chromosome,
                shifted_bin=start // hic_resolution,
                hic_resolution=hic_resolution,
                number_of_chunks=number_of_chunks,
                exclude_N_islands=True,
            )
            total_number_of_samples += number_of_samples
            total_number_of_contacts += number_of_contacts
    print(
        "%s has %d samples, each sample has %.1f contacts (%.1f%%) on average"
        % (
            chromosome,
            total_number_of_samples,
            total_number_of_contacts / total_number_of_samples,
            total_number_of_contacts
            / total_number_of_samples
            / number_of_chunks
            / (number_of_chunks - 1)
            * 2
            * 100,
        )
    )
    ## 2. reference genome
    if args.process_ref_genome:
        os.makedirs(args.output_reference_genome_path, exist_ok=True)
        chr_to_seq, _ = extract_chr_seq_and_len(fasta_path, ">")
        # cut into TAD sub sequences if TAD file is provided
        if args.tad_path:
            sub_seqs, starts = cut_sequence_by_tad_bin(
                chr_to_seq[chromosome],
                target_chr=chromosome,
                tad_path=args.tad_path,
                hic_resolution=hic_resolution,
            )
        else:
            sub_seqs = [chr_to_seq[chromosome]]
            starts = [0]
        for sequence, start in zip(sub_seqs, starts):
            if len(sequence) < hic_resolution * number_of_chunks:
                print(
                    "Exclulde a short TAD of %d bp starts at %d "
                    % (len(sequence), start)
                )
            else:
                number_of_samples, number_of_contacts = sample_function(
                    sequence=sequence,
                    hic_path=args.hic_path,
                    output_path=args.output_reference_genome_path
                    + "sample_"
                    + chromosome
                    + ".txt",
                    target_chr=chromosome,
                    shifted_bin=start // hic_resolution,
                    hic_resolution=hic_resolution,
                    number_of_chunks=number_of_chunks,
                    exclude_N_islands=True,
                )


if __name__ == "__main__":
    args = get_args()
    assert args.sample_strategy in ["random", "fixed_distance"]
    ## TODO support sequential sampling
    if args.sample_strategy == "random":
        main_random(args)
    else:
        main_fixed_distance(args)
