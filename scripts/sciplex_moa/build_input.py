"""
Build the raw-count A549 input that the embeddings were computed on.

Source: Sci-Plex 3 (Srivatsan et al. 2020, Science, doi:10.1126/science.aax6234), A549
dataset as curated by CZ CELLxGENE Discover:
https://datasets.cellxgene.cziscience.com/8dc61aa8-8d55-4180-854e-9dde86aeaad2.h5ad
(143,015 cells, raw counts in X, obs includes target / product_name / dose / viability).

Steps, in this order (each one changes the result if moved):
1. index genes by symbol (var.feature_name), keeping duplicate symbols;
2. keep protein-coding genes (bmfm_targets/tokenization/resources/protein_coding_genes.json);
3. keep cells with >= 300 expressed genes (141,814 cells);
4. flag 5,000 highly variable genes with scanpy's seurat_v3 flavor on the raw counts; the
   duplicate symbols give 5,002 flagged genes;
5. subset to those genes and make the duplicate symbols unique (MALL-1, PINX1-1).

Writes one h5ad (141,814 cells x 5,002 genes, raw counts) to pass to the embedding models.
"""
import argparse
import json
from pathlib import Path

import anndata as ad
import scanpy as sc

PROTEIN_CODING = (
    Path(__file__).resolve().parents[2]
    / "bmfm_targets/tokenization/resources/protein_coding_genes.json"
)


def build(adata: sc.AnnData, protein_coding: list[str]) -> sc.AnnData:
    adata.var["ens_gene_ids"] = adata.var_names
    adata.var = adata.var.set_index("feature_name")
    adata = adata[:, adata.var_names.isin(protein_coding)].copy()
    sc.pp.filter_cells(adata, min_genes=300)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    adata = adata[:, adata.var["highly_variable"].values].copy()
    adata.var_names = adata.var_names.astype(str)
    adata.var_names_make_unique()
    return adata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="CELLxGENE A549 Sci-Plex 3 h5ad")
    p.add_argument("--out", required=True, help="output h5ad")
    p.add_argument("--protein-coding", default=str(PROTEIN_CODING))
    a = p.parse_args()
    with open(a.protein_coding) as f:
        protein_coding = json.load(f)
    adata = build(sc.read_h5ad(a.input), protein_coding)
    print(adata)
    # pandas 3 stores strings as StringArray; anndata 0.11 writes them only on opt-in
    if hasattr(ad.settings, "allow_write_nullable_strings"):
        ad.settings.allow_write_nullable_strings = True
    adata.write_h5ad(a.out)


if __name__ == "__main__":
    main()
