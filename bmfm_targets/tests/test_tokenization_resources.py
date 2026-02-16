import pytest

from bmfm_targets.tokenization import resources


@pytest.mark.skip(reason="sps integration faile, requires additional review")
def test_chromosome_df_indexed_by_gene_symbol():
    chrom_df = resources.get_gene_chromosome_locations()
    assert chrom_df.index.name == "gene_symbol"


@pytest.mark.skip(reason="sps integration fails, requires additional review")
def test_all_chromosomes_present():
    chrom_df = resources.get_gene_chromosome_locations()
    # should this be exactly 25?? currently we have lots of pseudo-chromosomes
    assert chrom_df.chromosome.value_counts().shape[0] >= 25
