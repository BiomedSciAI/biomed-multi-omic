import numpy as np
import pytest
from scanpy import read_h5ad

from bmfm_targets.tests.helpers import Zheng68kPaths
from bmfm_targets.training.callbacks import BatchIntegrationCallback


@pytest.fixture(scope="module")
def dummy_embeddings():
    ad = read_h5ad(Zheng68kPaths.root / "h5ad" / "zheng68k.h5ad")
    embeddings = np.random.randn(ad.shape[0], 768)
    results = {"embeddings": embeddings, "cell_name": ad.obs_names}
    return ad, results


@pytest.mark.skip(reason="sps integration fails, requires additional review")
def test_add_embed_to_adata(dummy_embeddings):
    ad, results = dummy_embeddings
    callback_obj = BatchIntegrationCallback()
    ad_emb = callback_obj.add_embed_to_obsm(adata=ad, results=results)
    assert ad_emb.obsm["BMFM-RNA"] is not None

    callback_obj.batch_column_name = "cell_type_ontology_term_id"
    ad_emb.obs["batch"] = ad_emb.obs[callback_obj.batch_column_name]
    callback_obj.benchmarking_methods = ["Scanorama"]
    assert all(callback_obj.liger_emb(ad_emb) == 0)

    callback_obj.batch_column_name = "celltype"
    callback_obj.target_column_name = "celltype"
    emb_int_table_wo_batch_name = callback_obj.generate_table_batch_integration(
        adata_emb=ad_emb
    )
    assert not "Avg_batch" in emb_int_table_wo_batch_name.columns

    callback_obj.batch_column_name = "cluster"
    emb_int_table_w_batch_name = callback_obj.generate_table_batch_integration(
        adata_emb=ad_emb
    )
    assert "Avg_batch" in emb_int_table_w_batch_name.columns
