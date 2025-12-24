import functools
import json
import pickle
from pathlib import Path

import cellxgene_census
import pandas as pd
from huggingface_hub import hf_hub_download

ABBRV_TAXA_NAMES = {
    "homo_sapiens": "hsapiens",
    "mus_musculus": "mmusculus",
    "danio_rerio": "drerio",
    "caenorhabditis_elegans": "celegans",
    "saccharomyces_cerevisiae": "scerevisiae",
    "arabidopsis_thaliana": "athaliana",
    "canis_familiaris": "cfamiliaris",
    "gallus_gallus": "ggallus",
}


class TaxaNamesUnavailableError(Exception):
    pass


def get_hgnc_df(filter_query=None):
    fname = Path(__file__).parent / "hgnc_complete_set_2024-08-23.tsv"

    hgnc = pd.read_csv(fname, sep="\t")
    if filter_query:
        return hgnc.query(filter_query)
    return hgnc


def get_protein_coding_genes():
    fname = Path(__file__).parent / "protein_coding_genes.json"
    with open(fname) as f:
        return json.load(f)


def _get_original_L1000_gene_names():
    """
    Read the L1000 genes from original file, parse the file to extract the list itself.
    If the file is stored localy use it, otherwise download it.
    """
    local_fname = (
        Path(__file__).parent / "GSE70138_Broad_LINCS_gene_info_2017-03-06.txt"
    )
    if local_fname.exists():
        genes_df = pd.read_csv(local_fname)
    else:
        url = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE70138&format=file&file=GSE70138%5FBroad%5FLINCS%5Fgene%5Finfo%5F2017%2D03%2D06%2Etxt%2Egz"
        genes_df = pd.read_csv(url, compression="gzip", delimiter="\t")
        genes_df.to_csv(local_fname, index=False)

    return genes_df[genes_df["pr_is_lm"] == 1]["pr_gene_symbol"]


def get_L1000_genes():
    """Loads L1000 gene list (978 genes) and map old gene names to new ones using the default gene name mapping same as in StandardizeGeneNamesTransform."""
    genes = _get_original_L1000_gene_names()

    # Load the default gene name mapping
    file_path = (
        Path(__file__).resolve().parents[2]
        / "transforms"
        / "protein_coding_gene_mapping_uppercase_hgnc_2024_08_23.json"
    )

    gene_map = json.load(file_path.open())
    genes = genes.map(gene_map).fillna(genes)

    return list(genes)


@functools.lru_cache
def get_gene_chromosome_locations(
    species: str = "human",
    chroms_file: Path | str | None = None,
) -> pd.DataFrame:
    """Get chromosomes for genes."""
    if chroms_file is None:
        fname = Path(__file__).parent / f"{species}_to_chrom_pos.csv"
        if not fname.exists():
            _get_gene_chromosome_info(
                species,
                file_name=fname,
            )
    else:
        fname = Path(chroms_file)

    chroms_df = pd.read_csv(fname, index_col="gene_symbol")

    return chroms_df


def get_median_from_geneformer(aggregate_transcripts: str | None = None):
    """
    Get median values for the gene expressions provided by geneformer: https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/gene_median_dictionary_gc104M.pkl
    and maps ENSG_ID to gene symbol using cellxgene_census.

    Returns
    -------
    pd.DataFrame: Gene_symbol and corresponding medians.
    """
    repo_id = "ctheodoris/Geneformer"
    filename = "geneformer/gene_median_dictionary_gc104M.pkl"

    median_file = hf_hub_download(repo_id=repo_id, filename=filename)

    with open(median_file, "rb") as f:
        ensg_median = pickle.load(f)

    with cellxgene_census.open_soma(census_version="latest") as census:
        human = census["census_data"]["homo_sapiens"]
        gene_data = human.ms["RNA"].var.read().concat().to_pandas()

    ensg_gene_name = gene_data[["feature_id", "feature_name"]]
    ensg_to_symbol = dict(
        zip(ensg_gene_name["feature_id"], ensg_gene_name["feature_name"])
    )
    converted_data = [
        {"gene_symbol": ensg_to_symbol.get(ensg, None), "value": val}
        for ensg, val in ensg_median.items()
        if ensg_to_symbol.get(ensg) is not None
    ]
    gene_symbol_median = pd.DataFrame(converted_data)
    gene_symbol_median = gene_symbol_median.set_index("gene_symbol")

    if aggregate_transcripts == "max":
        gene_symbol_median = gene_symbol_median.groupby(level=0).max()
    elif aggregate_transcripts == "sum":
        gene_symbol_median = gene_symbol_median.groupby(level=0).sum()
    else:
        raise NotImplementedError(
            "Aggregation method not implemented. Choose sum or max."
        )

    return gene_symbol_median


@functools.lru_cache
def get_gene_medians():
    """Read the file containing gene_symbol and corresponding medians. If the file is available locally use it otherwise download it."""
    fname = Path(__file__).parent / "geneName_median_cellxgene.tsv"

    if fname.exists():
        gene_medians = pd.read_csv(fname, sep="\t", index_col="gene_symbol")
    else:
        gene_medians = get_median_from_geneformer(aggregate_transcripts="max")

    return gene_medians


def get_ortholog_genes(
    return_mapping: bool = False,
    from_species: str = "mus_musculus",
    to_species: str = "homo_sapiens",
    id_type: str = "gene_name",
    mapping_file: Path | str = None,
) -> dict | list:
    """Map genes identifiers between species using orthologs. Current implementation assumes 1:1 mapping."""
    if mapping_file is None:
        fname = Path(__file__).parent / f"{from_species}_{to_species}_orthologos.tsv"
        if not fname.exists():
            _create_ortholog_mapping_table(
                from_species,
                to_species,
                file_name=fname,
            )
    else:
        fname = Path(mapping_file)

    mapping = pd.read_csv(fname, sep="\t")

    to_ids = f"{to_species}_{id_type}"
    from_ids = f"{from_species}_{id_type}"
    if return_mapping:
        return dict(zip(mapping[from_ids], mapping[to_ids]))
    else:
        return mapping[from_ids].tolist()


def _get_abbrv_taxa_names(from_species, to_species) -> tuple[str, str]:
    from_species = ABBRV_TAXA_NAMES.get(from_species, False)
    to_species = ABBRV_TAXA_NAMES.get(to_species, False)

    if from_species and to_species:
        return from_species, to_species
    else:
        raise TaxaNamesUnavailableError("Complete taxa names not available.")


def _create_ortholog_mapping_table(
    from_species: str,
    to_species: str,
    high_confidence_only: bool = True,
    one_to_one_only: bool = True,
    file_name: Path | str | None = None,
    drop_na: bool = True,
) -> dict | None:
    from io import StringIO

    import requests

    if "_" in from_species or "_" in to_species:
        from_species, to_species = _get_abbrv_taxa_names(from_species, to_species)

    query_bm = (
        "http://ensembl.org/biomart/martservice?query="
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<!DOCTYPE Query>"
        '<Query  virtualSchemaName = "default" formatter = "TSV" header = "0" uniqueRows = "0" count = "" datasetConfigVersion = "0.6" >'
        f'<Dataset name = "{from_species}_gene_ensembl" interface = "default" >'
        '<Attribute name = "ensembl_gene_id" />'
        '<Attribute name = "external_gene_name" />'
        f'<Attribute name = "{to_species}_homolog_ensembl_gene" />'
        f'<Attribute name = "{to_species}_homolog_associated_gene_name" />'
        f'<Attribute name = "{to_species}_homolog_orthology_confidence" />'
        f'<Attribute name = "{to_species}_homolog_orthology_type" />'
        "</Dataset>"
        "</Query>"
    )

    req = requests.get(query_bm)
    if str(req.text).startswith("Query ERROR"):
        raise requests.exceptions.RequestException(f"{req.text}")

    col_names = [
        f"{from_species}_gene_ensembl",
        f"{from_species}_gene_name",
        f"{to_species}_ensembl_gene",
        f"{to_species}_gene_name",
        f"{to_species}_homolog_orthology_confidence",
        f"{to_species}_homolog_orthology_type",
    ]
    serial_data = StringIO(req.text)
    mapping_df = pd.read_table(
        serial_data, header=None, names=col_names, index_col=None
    )

    if high_confidence_only:
        mapping_df = mapping_df[mapping_df[col_names[4]] == 1]

    if one_to_one_only:
        mapping_df = mapping_df[mapping_df[col_names[5]] == "ortholog_one2one"]

    if drop_na:
        mapping_df = mapping_df.dropna(how="any")

    if file_name is not None:
        mapping_df.to_csv(file_name, sep="\t", index=False)

    return mapping_df


def _get_gene_chromosome_info(
    species: str,
    file_name: Path | str | None = None,
) -> dict | None:
    import gzip
    import io

    import requests

    query_bm = "https://ftp.ensembl.org/pub/release-114/fasta/homo_sapiens/pep/Homo_sapiens.GRCh38.pep.all.fa.gz"

    req = requests.get(query_bm)
    if str(req.text).startswith("Query ERROR"):
        raise requests.exceptions.RequestException(f"{req.text}")

    all_pos_def = []

    missing_genes = {}
    missing_genes[species] = []

    gzip_file = io.BytesIO(req.content)
    with gzip.open(gzip_file, "rt") as f:
        proteome_lines = f.readlines()

    gene_symbol_to_location = {}
    gene_symbol_to_chrom = {}

    for line in proteome_lines:
        if line.startswith(">"):
            split_line = line.split()
            gene_symbol = [
                token for token in split_line if token.startswith("gene_symbol")
            ]
            if len(gene_symbol) > 0:
                gene_symbol = gene_symbol[0].split(":")

                if len(gene_symbol) == 2:
                    gene_symbol = gene_symbol[1]
                elif len(gene_symbol) > 2:
                    gene_symbol = ":".join(
                        gene_symbol[1:]
                    )  # fix for annoying zebrafish gene names with colons in them
                else:
                    1 / 0  # something weird happening, throw an error

                chrom = None

                chrom_arr = [
                    token for token in split_line if token.startswith("chromosome:")
                ]
                if len(chrom_arr) > 0:
                    chrom = chrom_arr[0].replace("chromosome:", "")
                else:
                    chrom_arr = [
                        token
                        for token in split_line
                        if token.startswith("primary_assembly:")
                    ]
                    if len(chrom_arr) > 0:
                        chrom = chrom_arr[0].replace("primary_assembly:", "")
                    else:
                        chrom_arr = [
                            token
                            for token in split_line
                            if token.startswith("scaffold:")
                        ]
                        if len(chrom_arr) > 0:
                            chrom = chrom_arr[0].replace("scaffold:", "")
                if chrom is not None:
                    gene_symbol_to_location[gene_symbol] = chrom.split(":")[2]
                    gene_symbol_to_chrom[gene_symbol] = chrom.split(":")[1]
                else:
                    missing_genes[species].append(gene_symbol)

    positional_df = pd.DataFrame()
    positional_df["gene_symbol"] = [
        gn.upper() for gn in list(gene_symbol_to_chrom.keys())
    ]
    positional_df["chromosome"] = list(gene_symbol_to_chrom.values())
    positional_df["start"] = list(gene_symbol_to_location.values())
    positional_df = positional_df.sort_values(["chromosome", "start"])
    positional_df["species"] = species
    all_pos_def.append(positional_df)

    master_pos_def = pd.concat(all_pos_def)
    # generate csv
    if file_name is not None:
        master_pos_def.to_csv(f"{file_name}", index=False)
