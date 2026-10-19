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


@pytest.fixture(scope="module")
def clustered_adata():
    """Synthetic AnnData with well-separated cell-type clusters and two batches.

    Built here rather than read from disk so the invariant below is testable
    without a fixture h5ad. The clusters are separated far enough that the bio
    metrics are high and stable, which is what makes a scale regression visible.
    """
    import anndata
    import pandas as pd
    import scanpy as sc

    rng = np.random.default_rng(0)
    per_type, n_types, dim = 40, 4, 10
    centers = rng.normal(scale=12.0, size=(n_types, dim))
    X = np.vstack(
        [centers[i] + rng.normal(scale=0.6, size=(per_type, dim)) for i in range(n_types)]
    )
    labels = np.repeat([f"type_{i}" for i in range(n_types)], per_type)
    # Batch is assigned at random, so it carries no biological signal: the bio
    # metrics must not depend on it.
    batches = rng.choice(["b0", "b1"], size=X.shape[0])
    ad = anndata.AnnData(
        X=X.copy(),
        obs=pd.DataFrame(
            {"cell_type": pd.Categorical(labels), "donor": pd.Categorical(batches)},
            index=[f"cell{i}" for i in range(X.shape[0])],
        ),
    )
    ad.obsm["BMFM_RNA"] = X.copy()
    sc.pp.neighbors(ad, use_rep="BMFM_RNA")
    return ad


def test_single_batch_asw_is_scaled_to_unit_interval(clustered_adata):
    """ASW_label must be on scib's [0, 1] scale, not sklearn's raw [-1, 1].

    Regression test: the raw score was returned here while the multi-batch
    scib path rescaled by (asw + 1) / 2, so the two were ~0.35 apart in
    absolute terms and could not be averaged together.
    """
    from sklearn.metrics import silhouette_score

    from bmfm_targets.evaluation.embeddings import single_batch_bio_metrics

    out = single_batch_bio_metrics(clustered_adata.copy(), "BMFM_RNA", "cell_type")
    raw = float(
        silhouette_score(
            clustered_adata.obsm["BMFM_RNA"], clustered_adata.obs["cell_type"]
        )
    )
    assert out["ASW_label"] == pytest.approx((raw + 1) / 2, abs=1e-6)
    assert 0.0 <= out["ASW_label"] <= 1.0


def test_single_batch_asw_reported_with_a_singleton_class(clustered_adata):
    """A class with one cell must not suppress ASW_label.

    sklearn computes a silhouette fine in that case and scib applies no
    stricter condition, so the old `every class >= 2 cells` guard dropped the
    metric where scib would have reported it -- quietly turning avg_bio into a
    mean of two metrics instead of three.
    """
    from bmfm_targets.evaluation.embeddings import single_batch_bio_metrics

    ad = clustered_adata.copy()
    lab = ad.obs["cell_type"].astype(str).to_numpy()
    lab[0] = "singleton"
    ad.obs["cell_type"] = lab
    out = single_batch_bio_metrics(ad, "BMFM_RNA", "cell_type")
    assert not np.isnan(out["ASW_label"])


def test_single_batch_asw_is_nan_for_one_class(clustered_adata):
    """With a single label class a silhouette is undefined, so NaN is correct."""
    from bmfm_targets.evaluation.embeddings import single_batch_bio_metrics

    ad = clustered_adata.copy()
    ad.obs["cell_type"] = "only_one"
    out = single_batch_bio_metrics(ad, "BMFM_RNA", "cell_type")
    assert np.isnan(out["ASW_label"])


def test_bio_metrics_do_not_depend_on_batch_count(clustered_adata):
    """THE INVARIANT: bio conservation is independent of the batch assignment.

    Scoring the same embedding with a real batch column (scib path) and with
    batch_column_name=None (single-batch path) must give the same bio block.
    This is what makes a per-tissue mean of Avg_bio over a mix of single- and
    multi-batch tissues meaningful, and it is what the three-way scale
    mismatch broke.
    """
    from bmfm_targets.evaluation.embeddings import calculate_batch_integration_metrics

    multi = calculate_batch_integration_metrics(
        clustered_adata.copy(),
        batch_column_name="donor",
        target_column_name="cell_type",
        embed_key="BMFM_RNA",
    )
    single = calculate_batch_integration_metrics(
        clustered_adata.copy(),
        batch_column_name=None,
        target_column_name="cell_type",
        embed_key="BMFM_RNA",
    )
    # Avg_batch only exists for the multi-batch call; the bio block must match.
    assert "Avg_batch" in multi.columns
    assert "Avg_batch" not in single.columns
    for col in (
        "NMI_cluster_by_cell_type_(bio)",
        "ARI_cluster_by_cell_type_(bio)",
        "ASW_by_cell_type_(bio)",
        "Avg_bio",
    ):
        assert col in single.columns, f"{col} missing from the single-batch result"
        assert single[col].iloc[0] == pytest.approx(multi[col].iloc[0], abs=0.02), (
            f"{col} differs between the batch paths: "
            f"single={single[col].iloc[0]} multi={multi[col].iloc[0]}"
        )
