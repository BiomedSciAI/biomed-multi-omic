import numpy as np
import pytest
from scanpy import read_h5ad

from bmfm_targets.tests.helpers import Zheng68kPaths
from bmfm_targets.training.callbacks import CziBenchmarkCallback, align_embeddings


@pytest.fixture(scope="module")
def dummy_embeddings():
    ad = read_h5ad(Zheng68kPaths.root / "h5ad" / "zheng68k.h5ad")
    embeddings = np.random.randn(ad.shape[0], 768)
    results = {"embeddings": embeddings, "cell_name": ad.obs_names}
    return ad, results


def test_czi_benchhmark(dummy_embeddings):
    ad, results = dummy_embeddings
    callback_obj = CziBenchmarkCallback(
        batch_column_name="batch", target_column_name="celltype", n_folds=2
    )
    aligned = align_embeddings(adata=ad, results=results)
    adata_emb = ad.copy()
    adata_emb.obsm["BMFM_RNA"] = aligned

    results = callback_obj.execute_czi_cell_type_classification_benchmark(adata_emb)
    assert results.shape[0] > 0
    assert "knn" in results.columns
