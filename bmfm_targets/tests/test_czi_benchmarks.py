from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
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


@pytest.fixture()
def emb_and_labels(dummy_embeddings):
    ad, results = dummy_embeddings
    aligned = align_embeddings(adata=ad, results=results)
    adata_emb = ad.copy()
    adata_emb.obsm["BMFM_RNA"] = aligned
    # zheng68k has no batch column; two interleaved batches are enough here
    adata_emb.obs["batch"] = np.tile(["b1", "b2"], adata_emb.n_obs)[: adata_emb.n_obs]
    return adata_emb, aligned, adata_emb.obs["celltype"]


def _callback(tmp_path, **kwargs):
    """A callback set up as on_predict_end would leave it, without ClearML."""
    cb = CziBenchmarkCallback(
        batch_column_name="batch", target_column_name="celltype", n_folds=2, **kwargs
    )
    cb.cl = None
    cb._rows = []
    cb._failed = []
    return cb, SimpleNamespace(log_dir=str(tmp_path))


def _classifiers(cb):
    return {r["classifier"] for r in cb._rows}


def test_default_runs_only_label_prediction_with_lr():
    cb = CziBenchmarkCallback(target_column_name="celltype")
    assert cb.tasks == ("label_prediction",)
    assert cb.classifiers == ("lr",)


@pytest.mark.parametrize(
    "kwargs",
    [{"tasks": ["nope"]}, {"classifiers": ["svm"]}, {"classifiers": []}],
)
def test_invalid_tasks_or_classifiers_raise_at_construction(kwargs):
    with pytest.raises(ValueError):
        CziBenchmarkCallback(target_column_name="celltype", **kwargs)


@pytest.mark.parametrize("requested", [("lr",), ("lr", "knn"), ("lr", "knn", "rf")])
def test_only_requested_classifiers_are_run(emb_and_labels, tmp_path, requested):
    _, embeddings, labels = emb_and_labels
    cb, _ = _callback(tmp_path, classifiers=requested)

    cb._report_label_prediction(embeddings, labels)

    ran = _classifiers(cb)
    assert set(requested) <= ran
    assert ran - set(requested) <= {f"MEAN({','.join(sorted(requested))})"}, ran


def test_csv_has_one_row_per_metric_with_context_columns(emb_and_labels, tmp_path):
    adata_emb, embeddings, labels = emb_and_labels
    cb, trainer = _callback(tmp_path)
    cb._ctx = (trainer, adata_emb, labels)

    cb._run_task(
        "label_prediction", lambda: cb._report_label_prediction(embeddings, labels)
    )

    metrics_df = pd.read_csv(tmp_path / "czbenchmarks_metrics.csv")
    assert len(metrics_df) == len(cb._rows) > 0
    assert (metrics_df["n_cells"] == adata_emb.n_obs).all()
    assert (metrics_df["n_classes"] == labels.nunique()).all()
    assert (metrics_df["n_batches"] == adata_emb.obs["batch"].nunique()).all()
    assert (metrics_df["target_column"] == "celltype").all()
    assert {"mean_fold_f1", "mean_fold_accuracy"} <= set(metrics_df["metric"])
    assert metrics_df["value"].between(0, 1).all()


def test_f1_scalar_is_still_reported_for_lr(emb_and_labels, tmp_path):
    """The ``f1`` scalar in ClearML is the lr ``mean_fold_f1``; dashboards query it."""
    _, embeddings, labels = emb_and_labels
    cb, _ = _callback(tmp_path, classifiers=("lr", "knn"))
    cb.cl = MagicMock()

    cb._report_label_prediction(embeddings, labels)

    lr_f1 = next(
        r["value"]
        for r in cb._rows
        if r["metric"] == "mean_fold_f1" and r["classifier"] == "lr"
    )
    reported = {
        c.kwargs["name"]: c.kwargs["value"]
        for c in cb.cl.report_single_value.call_args_list
    }
    assert reported["f1"] == pytest.approx(lr_f1)
    assert reported["czi_label_prediction_mean_fold_f1_knn"] != reported["f1"]


def test_failed_task_is_recorded_and_earlier_results_stay_on_disk(
    emb_and_labels, tmp_path
):
    adata_emb, embeddings, labels = emb_and_labels
    cb, trainer = _callback(tmp_path)
    cb._ctx = (trainer, adata_emb, labels)

    cb._run_task(
        "label_prediction", lambda: cb._report_label_prediction(embeddings, labels)
    )
    n_rows = len(pd.read_csv(tmp_path / "czbenchmarks_metrics.csv"))

    def boom():
        raise RuntimeError("clustering exploded")

    cb._run_task("clustering", boom)

    assert cb._failed == ["clustering"]
    assert len(pd.read_csv(tmp_path / "czbenchmarks_metrics.csv")) == n_rows


def test_task_raising_inside_report_task_is_recorded_as_failed(
    emb_and_labels, tmp_path
):
    adata_emb, embeddings, labels = emb_and_labels
    cb, trainer = _callback(tmp_path)
    cb._ctx = (trainer, adata_emb, labels)

    class RaisingTask:
        def run(self, cell_representation, task_input):
            raise RuntimeError("task blew up")

    cb._run_task(
        "clustering",
        lambda: cb._report_task("CZI Clustering", RaisingTask(), embeddings, None),
    )

    assert cb._failed == ["clustering"]


def test_opt_in_clustering_and_embedding_tasks_produce_bounded_metrics(
    emb_and_labels, tmp_path
):
    """Runs the real czbenchmarks clustering, embedding and batch integration tasks."""
    from czbenchmarks.tasks import (
        ClusteringTask,
        ClusteringTaskInput,
        EmbeddingTask,
        EmbeddingTaskInput,
    )

    adata_emb, embeddings, labels = emb_and_labels
    sub_emb, sub_labels, sub_obs = embeddings, labels, adata_emb.obs
    cb, _ = _callback(tmp_path)

    cb._report_task(
        "CZI Clustering",
        ClusteringTask(),
        sub_emb,
        ClusteringTaskInput(obs=sub_obs, input_labels=sub_labels),
    )
    cb._report_task(
        "CZI Embedding",
        EmbeddingTask(),
        sub_emb,
        EmbeddingTaskInput(input_labels=sub_labels),
    )

    cb._report_batch_integration(adata_emb, sub_emb, sub_labels)

    by_task = {}
    for r in cb._rows:
        by_task.setdefault(r["task"], []).append(r)
    assert set(by_task) == {"CZI Clustering", "CZI Embedding", "CZI Batch Integration"}
    assert all(0 <= r["value"] <= 1 for r in by_task["CZI Batch Integration"])
    clustering = {r["metric"]: r["value"] for r in by_task["CZI Clustering"]}
    assert {"adjusted_rand_index", "normalized_mutual_info"} <= set(clustering)
    # random embeddings carry no label information
    assert all(abs(v) < 0.2 for v in clustering.values())


def test_batch_integration_metrics_are_written_without_clearml(
    emb_and_labels, tmp_path
):
    """BatchIntegrationCallback must persist its table even with no ClearML logger."""
    from bmfm_targets.training.callbacks import (
        MODEL_EMBED_KEY,
        BatchIntegrationCallback,
    )

    adata_emb, embeddings, _ = emb_and_labels
    sub = adata_emb.copy()
    sub.obsm[MODEL_EMBED_KEY] = embeddings
    cb = BatchIntegrationCallback(
        batch_column_name="batch", target_column_name="celltype"
    )
    cb.cl = None

    cb._execute_batch_integration(SimpleNamespace(log_dir=str(tmp_path)), sub)

    metrics_df = pd.read_csv(tmp_path / "batch_integration_metrics.csv")
    assert len(metrics_df) > 0
    assert (metrics_df["n_cells"] == adata_emb.n_obs).all()
    assert (metrics_df["target_column"] == "celltype").all()
    assert (metrics_df["batch_column"] == "batch").all()
