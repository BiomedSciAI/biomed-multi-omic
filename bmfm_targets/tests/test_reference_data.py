import pandas as pd

from bmfm_targets.tokenization.resources.reference_data import (
    get_L1000_genes,
    get_protein_coding_genes,
)


def test_get_L1000_genes():
    pcg = get_protein_coding_genes()
    l1000 = get_L1000_genes()

    assert len(l1000) == 978
    assert not pd.isna(l1000).any()

    intersect = {g for g in pcg if g in l1000}
    assert len(intersect) == 978
