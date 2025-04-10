from pathlib import Path

import numpy as np
import scanpy as sc
from anndata import read_h5ad


def obs_key_wise_subsampling(adata, obs_key, N):
    """
    Subsample each class to same cell numbers (N). Classes are given by obs_key pointing to categorical in adata.obs.

    From https://github.com/scverse/scanpy/issues/987
    """
    counts = adata.obs[obs_key].value_counts()
    counts = counts[counts >= N]
    # subsample indices per group defined by obs_key
    indices = [
        np.random.choice(
            adata.obs_names[adata.obs[obs_key] == group], size=N, replace=False
        )
        for group in counts.index
    ]
    selection = np.hstack(np.array(indices))
    return adata[selection].copy()


ad = read_h5ad(
    "/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/pretrain/cellxgene/h5ad/cellxgene.h5ad"
)

obs_name_for_stratification = "cell_type_ontology_term_id"

ad_downsample = obs_key_wise_subsampling(ad, obs_name_for_stratification, 10)

# remove genes which no longer appear in any cells
sc.pp.filter_genes(ad_downsample, min_cells=1)
ad_downsample.write_h5ad(
    Path(__file__).parent / "resources/pretrain/cellxgene/h5ad/cellxgene.h5ad"
)
