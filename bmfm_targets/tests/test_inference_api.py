import scanpy as sc

import bmfm_targets as bmfm


def test_bmfm_adata_inference():
    adata = sc.datasets.pbmc3k()
    adata = adata[:5, :].copy()
    sc.pp.log1p(adata)
    bmfm.inference(adata)

    assert adata.obs["bmfm_pred_cell_type"].values.tolist() == [
        "T cell",
        "B cell",
        "T cell",
        "monocyte",
        "natural killer cell",
    ]
    assert "bmfm_pred_tissue" in adata.obs.columns
    assert "bmfm_pred_tissue_general" in adata.obs.columns
    assert "bmfm_pred_tissue_general_donor_id" not in adata.obs.columns
    assert "X_bmfm" in adata.obsm
    assert adata.obsm["X_bmfm"].shape == (5, 768)
