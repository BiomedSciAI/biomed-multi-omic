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


def test_czi_label_prediction_keeps_every_classifier(dummy_embeddings):
    """Every classifier czbenchmarks ran must be collected, not just ``lr``.

    ``MetadataLabelPredictionTaskInput`` has no ``classifiers`` field, so a
    request for a subset is silently dropped and czbenchmarks always fits lr,
    knn and rf plus a ``MEAN(...)`` row. Selecting one column afterwards
    therefore paid for all of them and threw two thirds away, and -- worse --
    made us incomparable to the published CZI tables, which quote the
    per-classifier values. This pins the whole set.
    """
    ad, results = dummy_embeddings
    callback_obj = CziBenchmarkCallback(
        batch_column_name="batch", target_column_name="celltype", n_folds=2
    )
    aligned = align_embeddings(adata=ad, results=results)
    adata_emb = ad.copy()
    adata_emb.obsm["BMFM_RNA"] = aligned

    # on_predict_end normally sets these up; there is no ClearML in tests, which
    # must not stop the metrics being collected.
    callback_obj.cl = None
    callback_obj._rows = []
    callback_obj._report_label_prediction(aligned, adata_emb.obs["celltype"])

    assert callback_obj._rows, "no label prediction metrics were collected"
    classifiers = {r["classifier"] for r in callback_obj._rows}
    assert {"lr", "knn", "rf"} <= classifiers, classifiers
    assert any(c.startswith("MEAN(") for c in classifiers), classifiers
    assert {"mean_fold_f1", "mean_fold_accuracy"} <= {
        r["metric"] for r in callback_obj._rows
    }
    assert all(r["task"] == "CZI Cell Type Classification" for r in callback_obj._rows)
