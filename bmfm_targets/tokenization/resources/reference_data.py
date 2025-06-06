import json
from pathlib import Path

import pandas as pd

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
