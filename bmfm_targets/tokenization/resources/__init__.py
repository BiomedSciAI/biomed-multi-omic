"""External resources related to tokenization."""

from .reference_data import (
    get_hgnc_df,
    get_protein_coding_genes,
    get_ortholog_genes,
    get_gene_chromosome_locations,
)

__all__ = [
    "get_hgnc_df",
    "get_protein_coding_genes",
    "get_ortholog_genes",
    "get_gene_chromosome_locations",
]
