import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import tqdm
from scipy import sparse
from sklearn.preprocessing import normalize

## load lexicons
RESOURCES_ROOT = Path(__file__).parent / "resources"
# A -> 0, N -> 4, NI -> 10
nucleotide_to_index = json.load(open(str(RESOURCES_ROOT / "nucleotide_lexicon.json")))
# 0 -> A, 4 -> N, 10 -> NI
index_to_nucleotide = {v: k for k, v in nucleotide_to_index.items()}
# N_A -> 美
biallele_to_encoded = json.load(open(str(RESOURCES_ROOT / "biallele_lexicon.json")))
# A -> 0, N -> 4, NI -> 10, B -> 4, BI -> 10
variant_to_index = json.load(open(str(RESOURCES_ROOT / "variant_lexicon.json")))

dna_to_complement = {
    "A": "T",
    "C": "G",
    "G": "C",
    "T": "A",
    "N": "N",
}


def extract_chr_seq_and_len(
    fasta_path: str | Path,
    header_identifier=">",
):
    """
    Extract chromosome length and sequence from fasta
    A header example: >1 dna:chromosome chromosome:GRCh37:1:1:249250621:1.

    Args:
    ----
        fasta_path (str): input fasta file
        header_identifier (str, optional): identify a header line if it starts with the identifier

    Returns:
    -------
        dict: dictionary mapping chromosome to length
        dict: dictionary mapping chromosome to chromosome sequence in the upper case
    """
    chr_to_seq = {}
    chromosome = None
    sequence = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line[0] == header_identifier:
                # save if finish reading all sequence lines for a chromosome
                if chromosome:
                    chr_to_seq[chromosome] = cleanup_dna_sequence(
                        "".join(sequence).upper()
                    )
                # parse header and rename it
                # e.g. '>chr1  AC:CM000663.2  gi:568336023  LN:248956422 ...' -> 'chr1'
                chromosome = line.split(" ")[0][1:]
                sequence = []
            else:
                sequence.append(line)
    # add the last chromosome
    chr_to_seq[chromosome] = cleanup_dna_sequence("".join(sequence).upper())
    chr_to_len = {k: len(v) for k, v in chr_to_seq.items()}
    return chr_to_seq, chr_to_len


def cleanup_dna_sequence(sequence):
    special_nucleotides = ["U", "R", "Y", "K", "M", "S", "W", "B", "H", "V", "D"]
    for nucleotide in special_nucleotides:
        sequence = sequence.replace(nucleotide, "N")
    return sequence


def parse_freq(freq_str: str):
    """
    Extract frequencies for reference and alternative alleles.

    Args:
    ----
    A frequency string.
    Example: GnomAD:0.9028,0.09722|TOMMO:0.9999,0.0001196|dbGaP_PopFreq:0.6048,0.3952

    Returns:
    -------
    The average probabilities for referecne and alternative alleles.
    """
    probability_list = []
    for src_probability_str in freq_str.split("|"):
        probabilities = src_probability_str.split(":")[-1].split(",")
        probability_list.append(
            [
                0.0 if probability == "." else float(probability)
                for probability in probabilities
            ]
        )
    # Average across all resources
    probabilities = np.array(probability_list).mean(axis=0)
    return probabilities


def _process_tabix(d_csr, tabix_handle, tabix_id, chr_seq, probability_cutoff=0):
    last_processed_pos = 0
    for row in tabix_handle.fetch(tabix_id, 0, len(chr_seq), parser=pysam.asVCF()):
        # try - catch to skip tabix format error in chromosome 3
        try:
            # only keep COMMON variants
            if ";COMMON" in row.info:
                # parser already shift the position by -1 for python
                curr_snp_pos = row.pos
                ref = row.ref
                ref_and_alts = np.array([ref] + row.alt.split(","))
                probabilities = parse_freq(row.info.split("FREQ=")[-1].split(";")[0])
                # fix dbSNP records that have 0% or extremely low probability of reference allele
                if probabilities[0] <= probability_cutoff:
                    probabilities[:] = 0.0
                    probabilities[0] = 1.0
                # exclude alt with probability = 0; otherwise 0 will be added to csr matrix
                ref_and_alts = ref_and_alts[probabilities > probability_cutoff]
                probabilities = probabilities[probabilities > probability_cutoff]
                # update the segment between the last snp and this snp
                _update_snp_probability_reference(
                    d_csr, chr_seq, last_processed_pos, curr_snp_pos
                )
                for alt, probability in zip(ref_and_alts, probabilities):
                    if len(alt) == len(ref):
                        # SNV / MUTATION - assign probability to mutated nucleotide or ref nucleotide
                        _update_snp_probability_mutation(
                            d_csr, curr_snp_pos, alt, probability
                        )
                    elif len(alt) < len(ref):
                        # DELETION
                        # assign probability to non-deleted prefix
                        _update_snp_probability_mutation(
                            d_csr, curr_snp_pos, alt, probability
                        )
                        # assign probability to deleted suffix
                        _update_snp_probability_deletion(
                            d_csr, curr_snp_pos, alt, ref, probability
                        )
                    else:  # if len(alt) > len(ref):
                        # INSERTION
                        _update_snp_probability_insertion(
                            d_csr, curr_snp_pos, alt, ref, probability
                        )
                # assign new last next position
                last_processed_pos = curr_snp_pos + len(ref)
        except:
            continue
    # update the segment between the last snp to the end of chromosome
    _update_snp_probability_reference(d_csr, chr_seq, last_processed_pos, len(chr_seq))


def _update_snp_probability_reference(d_csr, chr_seq, last_processed_pos, curr_snp_pos):
    for i in range(last_processed_pos, curr_snp_pos):
        x, y = i, variant_to_index[chr_seq[i]]
        d_csr[(x, y)] = 1


def _update_snp_probability_mutation(d_csr, curr_snp_pos, alt, probability):
    # example ref ACG -> alt TTT; curr_snp_pos at A
    for i, nt in enumerate(alt):
        x, y = curr_snp_pos + i, variant_to_index[nt]
        if (x, y) not in d_csr.keys():
            d_csr[(x, y)] = 0
        d_csr[(x, y)] += probability


def _update_snp_probability_deletion(d_csr, curr_snp_pos, alt, ref, probability):
    # example ref ACG -> alt A; curr_snp_pos at C NOT A
    for i in range(len(alt), len(ref)):
        x, y = curr_snp_pos + i, variant_to_index["DEL"]
        if (x, y) not in d_csr.keys():
            d_csr[(x, y)] = 0
        d_csr[(x, y)] += probability


def _update_snp_probability_insertion(d_csr, curr_snp_pos, alt, ref, probability):
    # example ref ACG -> alt ACGTT; curr_snp_pos at A
    # 1. assign probability to the prefix nucleotide; the 1st A ans 2nd C
    for i in range(len(ref) - 1):
        x, y = curr_snp_pos + i, variant_to_index[alt[i]]
        if (x, y) not in d_csr.keys():
            d_csr[(x, y)] = 0
        d_csr[(x, y)] += probability
    # 2. assign insertion probability to the last nucleotide; the 3rd G as GI
    x, y = curr_snp_pos + len(ref) - 1, variant_to_index[f"{alt[len(ref)-1]}I"]
    if (x, y) not in d_csr.keys():
        d_csr[(x, y)] = 0
    d_csr[(x, y)] += probability


def convert_tabix_into_csv(
    fasta_path: str | Path,
    tabix_path: str | Path,
    tabix_index_path: str | Path,
    output_path: str | Path,
    target_chr="chr22",
    keyword=";COMMON",
):
    """Load a tabix file and convert rows with keyword into csv file."""
    # input
    _, chr_to_len = extract_chr_seq_and_len(fasta_path, ">")
    tabix_handle = pysam.TabixFile(str(tabix_path), index=str(tabix_index_path))
    # output
    os.makedirs(output_path, exist_ok=True)
    ## map chr to tabix contig id e.g. chr1 -> NC_000001.10
    tabix_id = [
        x
        for x in tabix_handle.contigs
        if "NC_0000" in x and str(int(x.split(".")[0][-2:])) == target_chr[3:]
    ][0]
    chr_len = chr_to_len[target_chr]
    tabix_columns = []
    for row in tabix_handle.fetch(tabix_id, 0, chr_len, parser=pysam.asVCF()):
        # try - catch to skip tabix format error in chromosome 3
        try:
            # only keep COMMON variants
            if keyword in row.info:
                tabix_columns.append(str(row).split("\t"))
        except:
            continue
    df_tabix = pd.DataFrame(tabix_columns)
    df_tabix.columns = ["contig", "pos", "id", "ref", "alt", "qual", "filter", "info"]
    df_tabix.to_csv(os.path.join(output_path, target_chr + ".csv"), index=False)


def create_snp_probability_matrix(
    fasta_path: str | Path,
    tabix_path: str | Path,
    tabix_index_path: str | Path,
    output_path: str | Path,
    target_chr="chr22",
    probability_cutoff=0,
):
    """
    Read a tabix file and calculate the chromosome-wise N-by-11 snp probability matrix
    aboout cvf tabix: https://gatk.broadinstitute.org/hc/en-us/articles/360035531692-VCF-Variant-Call-Format
    The vcf tabix has 8 tab-separated columnn. An example row:
    NC_000022.10  45016214        rs111604384     C       CCAA,CCGA       .       .       RS=111604384;dbSNPBuildID=132;SSR=0;GENEINFO=LINC00229:414351;VC=INS;INT;GNO;FREQ=1000Genomes:0.9425,.,0.05746|ALSPAC:0.9956,.,0.004411|Estonian:0.9996,.,0.0004464|NorthernSweden:0.9967,.,0.003333|TOPMED:0.9423,.,0.05769|TWINSUK:0.9976,.,0.002427|dbGaP_PopFreq:0.9999,0,0.0001302;COMMON.

    Args:
    ----
        fasta_path (str): input fasta file
        tabix_path (str): input tabix file; the paired index file xxx.tbi is also needed
        output_path (str): output directory
        target_chr (str): the target chromosome to be processed

    Returns:
    -------
        A N-by-12 matrix where N is the chromsome length and 11 corresponds to nucleotide/variant types [A, G, T, C, N, DEL, AI, GI, TI, CI, NI]; the last column is the binary flag of whether a variant happens at the locus
    """
    # input
    chr_to_seq, chr_to_len = extract_chr_seq_and_len(fasta_path, ">")
    tabix_handle = pysam.TabixFile(str(tabix_path), index=str(tabix_index_path))
    # output
    os.makedirs(output_path, exist_ok=True)
    ## map chr to tabix contig id e.g. chr1 -> NC_000001.10
    tabix_id = [
        x
        for x in tabix_handle.contigs
        if "NC_0000" in x and str(int(x.split(".")[0][-2:])) == target_chr[3:]
    ][0]
    chr_len = chr_to_len[target_chr]
    chr_seq = chr_to_seq[target_chr]
    ## fetch snp from tabix and save probablities in dictionary
    d_csr = {}
    _process_tabix(d_csr, tabix_handle, tabix_id, chr_seq, probability_cutoff)
    # convert dictionary into sparse matrix
    snp_probability_csr = sparse.csr_matrix(
        (
            list(d_csr.values()),
            ([x for x, y in d_csr.keys()], [y for x, y in d_csr.keys()]),
        ),
        shape=(chr_len, 11),
    )
    # normalize each row so that the sum of probabilities equals 1
    normalize(snp_probability_csr, norm="l1", axis=1, copy=False)
    # add the flag column if a row has any variant
    flag_csr = (snp_probability_csr.max(axis=1) != 1).astype("int")
    snp_probability_matrix = sparse.hstack((snp_probability_csr, flag_csr))
    # save csr sparse matrix
    sparse.save_npz(
        os.path.join(output_path, "snp_prob_" + target_chr + ".npz"),
        snp_probability_matrix,
    )
    return snp_probability_matrix


def create_reverse_complement_snp_probability_matrix(
    fasta_path: str | Path,
    snp_probability_matrix_path: str | Path,
    output_path: str | Path,
    target_chr="chr22",
):
    """
    Load a snp probability matrix (sparse csr format) and mirror it to the reverse complement version. Note that the matrix/order is reversed - position N, N-1, N-2, ... So the 1st row of the matrix corresponds to the last nucleotide position of a chromosome. When this matrix is used with e.g. TAD-aware sample, the TAD regions should also be reversed.

    Args:
    ----
        fasta_path (str): input fasta file
        snp_probability_matrix_path (str): the N-by-12 snp probability matrix
        output_path (str): output directory
        target_chr (str): the target chromosome to be processed

    Returns:
    -------
        Similar to the input matrix, it returns the reverse complement version of the N-by-12 matrix where N is the chromsome length and 11 corresponds to nucleotide/variant types [A, G, T, C, N, DEL, AI, GI, TI, CI, NI]; the last column is the binary flag of whether a variant happens at the locus
    """
    # reference genome
    chr_to_seq, chr_to_len = extract_chr_seq_and_len(fasta_path, ">")
    sequence = chr_to_seq[target_chr]
    # original snp probability matrix
    snp_probability_matrix = sparse.load_npz(
        os.path.join(snp_probability_matrix_path, "snp_prob_" + target_chr + ".npz")
    )
    ## TODO This is a numpy version that requires more memory - maybe implement a sparse matrix version
    # exclude the last flag column N-by-12 -> N-by-111
    mat = snp_probability_matrix.toarray()[:, :-1]
    # [0  1  2  3  4  5    6   7   8   9   10  11  ]
    # [A, G, T, C, N, DEL, AI, GI, TI, CI, NI, flag]
    ## step 1. swap ACGT columns; 0 <-> 2 & 1 <-> 3
    mat[:, [0, 2, 1, 3]] = mat[:, [2, 0, 3, 1]]
    ## step 2. for Insertion columns, move probabilities to complement_next_nucleotide_I based on ref
    mat_mapped_insertion = np.zeros((len(mat), 5))
    # sum probabilities from AI - NI (columns 6 - 10 )
    sum_XI = np.sum(mat[:, 6:11], axis=1)
    for i in range(len(sum_XI) - 1):
        if sum_XI[i] != 0:
            mat_mapped_insertion[
                i + 1, nucleotide_to_index[dna_to_complement[sequence[i + 1]]]
            ] = sum_XI[i]
            # normalize other probabilities after relocating the insertion probability
            mat[i, :6] = mat[i, :6] / (1 - sum_XI[i])
            mat[i + 1, :6] = mat[i + 1, :6] * (1 - sum_XI[i])
    mat[:, 6:11] = mat_mapped_insertion
    ## step 3. reverse
    mat[:, :] = mat[::-1, :]
    ## step 4. normalize each row so that the sum of probabilities equals 1
    snp_probability_csr_rc = sparse.csr_matrix(mat)
    normalize(snp_probability_csr_rc, norm="l1", axis=1, copy=False)
    ## step 5. add the flag column if a row has any variant
    flag_csr = (snp_probability_csr_rc.max(axis=1) != 1).astype("int")
    snp_probability_matrix_rc = sparse.hstack((snp_probability_csr_rc, flag_csr))
    # save csr sparse matrix
    os.makedirs(output_path, exist_ok=True)
    sparse.save_npz(
        os.path.join(output_path, "snp_prob_" + target_chr + ".npz"),
        snp_probability_matrix_rc,
    )


def sample_variant(
    matrix,
    replacement=False,
    seed=42,
):
    """
    Based on the N-by-12 variant frequency matrix, this function performs multinomial sampling of bialleles at variant loci.
    Then DNA segments are sampled where variants are replaced with encoded bialleles.

    Example:
    -------
    A insertion variant at locus = 100:
    | locus |  A  |  G  |  T  |  C  |  N  | DEL | AI  | GI  | TI  | CI  | NI  | flag|
    |  99   |  0  | 1.0 |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |
    |  100  | 0.8 |  0  |  0  |  0  |  0  |  0  | 0.2 |  0  |  0  |  0  |  0  |  1  |
    |  101  |  0  |  0  | 1.0 |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |
    reference sequence around this locus [A] is: TCGAG[A]TTCCA
    If replacement=False, it only returns TCGAG吖TTCCA (A_AI -> 兮)
    If replacement=True, three possible segments are:
    TCGAG帝TTCCA (A_A -> 帝 with p=0.64)
    TCGAG兮TTCCA (A_AI -> 兮 with p=0.32)
    TCGAG夕TTCCA (AI_AI -> 夕 with p=0.04)

    Args:
    ----
        matrix: input N-by-12 variant frequecy matrix in csr format from function create_snp_probability_matrix
        seed (int): random seed for numpy sampling

    Returns:
    -------
        An encoded DNA segment where variations are encoded in the biallele fashion.
    """
    num_rows = matrix.shape[0]
    np.random.seed(seed)
    encoded_seq = []
    for i in tqdm.tqdm(range(num_rows), total=num_rows):
        row_start = matrix.indptr[i]
        row_end = matrix.indptr[i + 1]
        if row_end - row_start == 1:
            encoded_seq.append(
                biallele_to_encoded[index_to_nucleotide[matrix.indices[row_start]]]
            )
        else:
            # row_end - 1 to exclude the last flag column
            index = np.random.choice(
                matrix.indices[row_start : (row_end - 1)],
                size=2,
                replace=replacement,
                p=matrix.data[row_start : (row_end - 1)],
            )
            encoded_seq.append(
                biallele_to_encoded[
                    "_".join(sorted(index_to_nucleotide[x] for x in index.tolist()))
                ]
            )
    return "".join(encoded_seq)


def remove_N_island(
    sequence,
    cutoff_N_length=10000,
    flank_length=1000,
    identifier="N",
    return_index=False,
):
    # remove 'NNNN' islands if longer than cutoff and cut into subsequences with flanking regions
    # example: remove_N('AAANNNNNAANNA', 3, 1) -> (['AAAN', 'NAANNA'], [[3, 8]])
    if flank_length > cutoff_N_length:
        print(
            "The flanking region is too long - the length should be less than the cutoff"
        )
        return [sequence], []
    seq_len = len(sequence)
    ## 1. find indexes for N-islands
    flag = False
    N_indexes = []
    for i in range(seq_len):
        if (not flag) and (sequence[i] == identifier):
            left_N = i
            flag = True
        if flag and (sequence[i] != identifier):
            right_N = i
            flag = False
            if right_N - left_N >= cutoff_N_length:
                N_indexes.append([left_N, right_N])
    # if there is an N tail
    if flag and (seq_len - left_N >= cutoff_N_length):
        N_indexes.append([left_N, seq_len])
    ## 2. cut into sub sequences
    sub_sequences = []
    start_index = 0
    for left_N, right_N in N_indexes:
        sub_sequences.append(sequence[start_index : (left_N + flank_length)])
        start_index = right_N - flank_length
    sub_sequences.append(sequence[start_index:seq_len])
    # drop 'NNNN' sub sequences at both ends
    if len(N_indexes):
        if N_indexes[0][0] == 0:
            sub_sequences = sub_sequences[1:]
        if N_indexes[-1][-1] == seq_len:
            sub_sequences = sub_sequences[:-1]
    if return_index:
        return sub_sequences, N_indexes
    else:
        return sub_sequences


def cut_sequences_into_chunks_and_write(
    sequences,
    output_path,
    segment_multiplier=1000,
    segment_len_min_before_multiplier=1,
    segment_len_max_before_multiplier=10,
    augmentation_reverse_complement=False,
    number_of_sampling=1,
    save_chunk_index=False,
    target_chr="chr22",
    seed=42,
):
    """
    Given input sequences, this function cut each sequence into non-overlapping DNA chunks and save txt files.

    Args:
    ----
        sequences: a list of multiple sequence (e.g. multiple TADs or subsequencs after removing N-islands from a chromosome)
        output_path: output txt file
        segment_multiplier (int): multiply the min and max by the multiplier to obtain the true length. It's designed for data with resolutions (e.g. set it to 1000 when studying HiC data at resolution = 1kb)
        segment_len_min_before_multiplier (int): minimum length before multiplier
        segment_len_max_before_multiplier (int): maximum length before multiplier
        augmentation_reverse_complement (bool): if reverse compliment is applied to generate chunks
        number_of_sampling (int): how many times the sequences are sampled (e.g. if we want to sample a chromosome tem times, then set it to 10)
        save_chunk_index (bool): if the chunk index is saved (e.g. with fixed length of 1kb chunks, the indices would be 0, 1, 2, ... correspoinding to 0-1kb, 1kb-2kb, 2kb-3kb, ... chunks)
        #TODO save_chunk_index only supports fixed length chunks
        seed (int): random seed for numpy sampling

    Returns:
    -------
        A txt file with DNA chunks from input sequences
    """
    np.random.seed(seed)
    with open(output_path, "w") as f:
        for _ in range(number_of_sampling):
            for seq in sequences:
                seq_len = len(seq)
                # 1. segments with a fixed length
                if (
                    segment_len_min_before_multiplier
                    == segment_len_max_before_multiplier
                ):
                    segment_len = int(
                        segment_len_min_before_multiplier * segment_multiplier
                    )
                    num_segments = (
                        (seq_len // segment_len) + 1
                        if seq_len % segment_len
                        else seq_len // segment_len
                    )
                    for i in range(num_segments):
                        chunk = seq[(i * segment_len) : ((i + 1) * segment_len)]
                        if set(chunk) == {"N"}:
                            continue
                        if save_chunk_index:
                            f.write(chunk + "," + target_chr + "_" + str(i) + "\n")
                        else:
                            f.write(chunk + "\n")
                        if augmentation_reverse_complement:
                            chunk_reverse_complement = "".join(
                                [dna_to_complement[x] for x in chunk][::-1]
                            )
                            if save_chunk_index:
                                f.write(
                                    chunk_reverse_complement
                                    + ",rc"
                                    + target_chr
                                    + "_"
                                    + str(i)
                                    + "\n"
                                )
                            else:
                                f.write(chunk_reverse_complement + "\n")
                # 2. segments with random lengths
                elif (
                    segment_len_min_before_multiplier
                    < segment_len_max_before_multiplier
                ):
                    left = 0
                    while left < seq_len:
                        right = (
                            left
                            + np.random.choice(
                                np.arange(
                                    segment_len_min_before_multiplier,
                                    segment_len_max_before_multiplier + 1,
                                )
                            )
                            * segment_multiplier
                        )
                        chunk = seq[left:right]
                        # update left
                        left = right
                        if set(chunk) == {"N"}:
                            continue
                        f.write(chunk + "\n")
                        if augmentation_reverse_complement:
                            chunk_reverse_complement = "".join(
                                [dna_to_complement[x] for x in chunk][::-1]
                            )
                            f.write(chunk_reverse_complement + "\n")
                else:
                    raise ValueError(
                        "The minimum length should not be larger than the maximum length"
                    )


def split_snp_sequence(
    input_path,
    output_path_snp=None,
    output_path_non_snp=None,
):
    import glob

    if output_path_snp == None:
        output_path_snp = input_path.rstrip("/") + "_snp"
    if output_path_non_snp == None:
        output_path_non_snp = input_path.rstrip("/") + "_non_snp"
    os.makedirs(output_path_snp, exist_ok=True)
    os.makedirs(output_path_non_snp, exist_ok=True)
    filenames = sorted(glob.glob(os.path.join(input_path, "*")))
    for filename in filenames:
        input_file = open(filename)
        output_file_snp = open(
            os.path.join(output_path_snp, filename.split("/")[-1]), "w"
        )
        output_file_non_snp = open(
            os.path.join(output_path_non_snp, filename.split("/")[-1]), "w"
        )
        for line in input_file:
            line = line.rstrip()
            # 1. SNP sequence
            if any(x not in "ACGTN" for x in line):
                output_file_snp.write(line + "\n")
            # 2. non-SNP sequence
            else:
                output_file_non_snp.write(line + "\n")
        input_file.close()
        output_file_snp.close()
        output_file_non_snp.close()


def cut_sequence_by_tad(sequence, target_chr, tad_path):
    df_tad = pd.read_csv(tad_path, header=None)
    df_tad.columns = ["chromosome", "start", "end"]
    df_tad = df_tad[df_tad.chromosome == target_chr]
    df_tad = df_tad.sort_values(by="start")
    start_all = df_tad.start.tolist()
    end_all = df_tad.end.tolist()
    ## add both ends if not exist
    if start_all[0] != 0:
        end_all = [start_all[0]] + end_all
        start_all = [0] + start_all
    if end_all[-1] != len(sequence):
        start_all.append(end_all[-1])
        end_all.append(len(sequence))
    sub_sequences = []
    for start, end in zip(start_all, end_all):
        sub_sequences.append(sequence[start:end])
    return sub_sequences


def main(args):
    chromosome = args.chromosome
    SNPDB_RESOURCES_PATH = (
        "/dccstor/bmfm-targets/data/omics/genome/snpdb/raw/resources/"
    )
    fasta_path = (
        SNPDB_RESOURCES_PATH + "GCA_000001405.15_GRCh38_no_alt_analysis_set.fna"
    )
    tabix_path = SNPDB_RESOURCES_PATH + "GCF_000001405.40.gz"
    tabix_index_path = SNPDB_RESOURCES_PATH + "GCF_000001405.40.gz.tbi"
    tad_path = args.tad_path
    snp_matrix_path = args.snp_matrix_path
    if args.process_snp_aware_genome:
        if args.calculate_prob_matrix:
            # calculate the N-by-12 variant frequency matrix
            snp_probability_matrix = create_snp_probability_matrix(
                fasta_path,
                tabix_path,
                tabix_index_path,
                snp_matrix_path,
                target_chr=chromosome,
                probability_cutoff=0.001,
            )
        else:
            # load the sparse matrix
            snp_probability_matrix = sparse.load_npz(
                snp_matrix_path + "snp_prob_" + chromosome + ".npz"
            )
        # sample and encode variants in the biallele fashion
        encoded_seq = sample_variant(
            snp_probability_matrix,
            replacement=False,
        )
        if (tad_path != None) and args.remove_N_island:
            tad_sub_seqs = cut_sequence_by_tad(encoded_seq, chromosome, tad_path)
            # remove N-islands and flatten the nested list
            encoded_sub_seqs = [
                x
                for xx in tad_sub_seqs
                for x in remove_N_island(xx, cutoff_N_length=1000, flank_length=100)
            ]
        elif args.remove_N_island:
            encoded_sub_seqs = remove_N_island(
                encoded_seq, cutoff_N_length=1000, flank_length=100
            )
        else:
            encoded_sub_seqs = [encoded_seq]
        # encoded genome
        os.makedirs(args.output_variation_biallele_path, exist_ok=True)
        cut_sequences_into_chunks_and_write(
            encoded_sub_seqs,
            output_path=os.path.join(
                args.output_variation_biallele_path, "sample_" + chromosome + ".txt"
            ),
            segment_multiplier=args.segment_multiplier,
            segment_len_min_before_multiplier=args.segment_len_min_before_multiplier,
            segment_len_max_before_multiplier=args.segment_len_max_before_multiplier,
            number_of_sampling=args.number_of_sampling,
            save_chunk_index=args.save_chunk_index,
            target_chr=chromosome,
        )
    if args.process_ref_genome:
        os.makedirs(args.output_reference_genome_path, exist_ok=True)
        chr_to_seq, _ = extract_chr_seq_and_len(fasta_path, ">")
        if (tad_path != None) and args.remove_N_island:
            tad_sub_seqs = cut_sequence_by_tad(
                chr_to_seq[chromosome], chromosome, tad_path
            )
            ref_sub_seqs = [
                x
                for xx in tad_sub_seqs
                for x in remove_N_island(xx, cutoff_N_length=1000, flank_length=100)
            ]
        elif args.remove_N_island:
            ref_sub_seqs = remove_N_island(
                chr_to_seq[chromosome], cutoff_N_length=1000, flank_length=100
            )
        else:
            ref_sub_seqs = [chr_to_seq[chromosome]]
        cut_sequences_into_chunks_and_write(
            ref_sub_seqs,
            output_path=os.path.join(
                args.output_reference_genome_path, "sample_" + chromosome + ".txt"
            ),
            segment_multiplier=args.segment_multiplier,
            segment_len_min_before_multiplier=args.segment_len_min_before_multiplier,
            segment_len_max_before_multiplier=args.segment_len_max_before_multiplier,
            augmentation_reverse_complement=args.augmentation_reverse_complement,
            number_of_sampling=args.number_of_sampling,
            save_chunk_index=args.save_chunk_index,
            target_chr=chromosome,
        )


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
        "--segment_len_min_before_multiplier",
        help="The minimum length of DNA chunks before multiplying it by segment_multiplier",
        type=int,
        required=False,
        default=1,
    )
    parser.add_argument(
        "--segment_len_max_before_multiplier",
        help="The maximum length of DNA chunks before multiplying it by segment_multiplier",
        type=int,
        required=False,
        default=10,
    )
    parser.add_argument(
        "--segment_multiplier",
        help="The basic unit of DNA chunks; the real length should be segment_len * segment_multiplier",
        type=int,
        required=False,
        default=1000,
    )
    parser.add_argument(
        "--number_of_sampling",
        help="How many time the genome is sampled",
        type=int,
        required=False,
        default=1,
    )
    parser.add_argument(
        "--calculate_prob_matrix",
        help="Calculate or load the matrix",
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
        "--process_ref_genome",
        help="create samples for the reference genome",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
    )
    parser.add_argument(
        "--remove_N_island",
        help="remove N islands (otherwise only remove all-N chunks)",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=True,
    )
    parser.add_argument(
        "--save_chunk_index",
        help="save chunk index for e.g. determining if a token has SNPs",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
    )
    parser.add_argument(
        "--augmentation_reverse_complement",
        help="create the reverse complement samples",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
    )
    parser.add_argument(
        "--output_variation_biallele_path",
        help="output directory for biallele-encoded samples",
        type=str,
        required=False,
        default="/dccstor/bmfm-targets/data/omics/genome/snpdb/raw/pretraining_data/sample_biallele/",
    )
    parser.add_argument(
        "--output_reference_genome_path",
        help="output directory for reference genome samples",
        type=str,
        required=False,
        default="/dccstor/bmfm-targets/data/omics/genome/snpdb/raw/pretraining_data/sample_ref/",
    )
    parser.add_argument(
        "--tad_path",
        help="input TAD file",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--snp_matrix_path",
        help="path for snp probability matrix",
        type=str,
        required=False,
        default="/dccstor/bmfm-targets/data/omics/genome/snpdb/raw/matrix_snp_probability/",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
