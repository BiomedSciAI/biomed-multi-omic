import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from bmfm_targets.training.callbacks import (
    SGDCallback,
    align_embeddings,
    create_adata_from_predictions_df,
    extract_predictions,
)
from bmfm_targets.training.metrics import perturbation_metrics as pm


def test_create_adata_from_predictions_df():
    index = ["S1", "S1", "S2", "S2"]  # two samples
    data = {
        "input_genes": ["G1", "G2", "G1", "G3"],
        "predicted_expressions": [0.1, 0.2, 0.3, 0.4],
        "perturbed_genes": ["G1", "G1", "G2", "G2"],
    }
    preds_df = pd.DataFrame(data, index=index)
    preds_df.index.name = "sample_id"
    baseline = pd.DataFrame(
        index=["G1", "G2", "G3", "G4"], data={"Control": [1.1, 1.2, 1.3, 1.4]}
    )
    grouped_predicted_expressions = pm.get_grouped_predictions(preds_df, baseline)
    adata = create_adata_from_predictions_df(preds_df, grouped_predicted_expressions)

    assert adata.shape == (2, 4)  # 2 samples × 3 unique input genes

    expected_X = np.array(
        [
            [0.1, 0.2, 1.3, 1.4],
            [0.3, 1.2, 0.4, 1.4],
        ]
    )
    assert np.allclose(adata.X.toarray(), expected_X)

    assert "target_gene" in adata.obs.columns
    assert set(adata.var.index) == {"G1", "G2", "G3", "G4"}
    assert list(adata.obs["target_gene"]) == ["G1", "G2"]


def test_extract_predictions_concatenates_batches():
    """Test that extract_predictions correctly concatenates batch predictions."""

    # Mock trainer with predict_loop.predictions
    class MockTrainer:
        class MockPredictLoop:
            predictions = [
                {"embeddings": np.array([[1, 2], [3, 4]]), "cell_name": ["c1", "c2"]},
                {"embeddings": np.array([[5, 6]]), "cell_name": ["c3"]},
            ]

        predict_loop = MockPredictLoop()

    result = extract_predictions(MockTrainer())
    assert result["embeddings"].shape == (3, 2)
    assert np.array_equal(result["embeddings"], np.array([[1, 2], [3, 4], [5, 6]]))
    assert list(result["cell_name"]) == ["c1", "c2", "c3"]


def test_align_embeddings_reorders_correctly():
    """Test that align_embeddings reorders embeddings to match adata.obs_names."""
    adata = sc.AnnData(np.zeros((3, 5)), obs=pd.DataFrame(index=["c3", "c1", "c2"]))
    results = {
        "embeddings": np.array([[1, 2], [3, 4], [5, 6]]),
        "cell_name": np.array(["c1", "c2", "c3"]),
    }
    aligned = align_embeddings(adata, results)
    # Should reorder to match adata.obs_names: c3, c1, c2
    assert np.array_equal(aligned, np.array([[5, 6], [1, 2], [3, 4]]))


def test_align_embeddings_missing_cell_raises():
    """Test that align_embeddings raises KeyError for missing cell names."""
    adata = sc.AnnData(np.zeros((2, 5)), obs=pd.DataFrame(index=["c1", "c_missing"]))
    results = {
        "embeddings": np.array([[1, 2], [3, 4]]),
        "cell_name": np.array(["c1", "c2"]),
    }
    with pytest.raises(KeyError):
        align_embeddings(adata, results)


def test_sgd_callback_evaluate_sgd_binary():
    """Test SGDCallback.evaluate_sgd with binary classification."""
    # Create mock AnnData with embeddings and labels
    n_samples = 100
    n_features = 10
    np.random.seed(42)

    # Create embeddings
    X_emb = np.random.randn(n_samples, n_features)

    # Create labels (binary)
    labels = np.array(["TypeA"] * 50 + ["TypeB"] * 50)

    # Create splits
    splits = np.array(["train"] * 40 + ["dev"] * 10 + ["test"] * 30 + ["train"] * 20)

    obs = pd.DataFrame(
        {
            "cell_type": labels,
            "split": splits,
        }
    )

    adata = sc.AnnData(np.zeros((n_samples, 5)), obs=obs)
    adata.obsm["X_emb"] = X_emb

    # Test evaluate_sgd
    callback = SGDCallback()
    results = callback.evaluate_sgd(adata, "X_emb", "split", "cell_type")

    # Check that all expected metrics are present
    assert "accuracy" in results
    assert "balanced_accuracy" in results
    assert "f1" in results
    assert "precision" in results
    assert "recall" in results

    # Check that metrics are in valid range [0, 1]
    for metric, value in results.items():
        assert 0 <= value <= 1, f"{metric} should be between 0 and 1, got {value}"


def test_sgd_callback_evaluate_sgd_multiclass():
    """Test SGDCallback.evaluate_sgd with multi-class classification."""
    # Create mock AnnData with embeddings and labels
    n_samples = 150
    n_features = 10
    np.random.seed(42)

    # Create embeddings with strong signal to make classes separable
    X_emb = np.random.randn(n_samples, n_features) * 0.1  # Small noise
    # Add strong signal to make classes separable
    X_emb[:50, 0] += 5  # TypeA
    X_emb[50:100, 1] += 5  # TypeB
    X_emb[100:, 2] += 5  # TypeC

    # Create labels (3 classes)
    labels = np.array(["TypeA"] * 50 + ["TypeB"] * 50 + ["TypeC"] * 50)

    # Create splits - ensure each class is represented in train/dev/test
    splits = np.array(
        ["train"] * 30
        + ["dev"] * 10
        + ["test"] * 10
        + ["train"] * 30  # TypeA
        + ["dev"] * 10
        + ["test"] * 10
        + ["train"] * 30  # TypeB
        + ["dev"] * 10
        + ["test"] * 10  # TypeC
    )

    obs = pd.DataFrame(
        {
            "cell_type": labels,
            "split": splits,
        }
    )

    adata = sc.AnnData(np.zeros((n_samples, 5)), obs=obs)
    adata.obsm["X_emb"] = X_emb

    # Test evaluate_sgd
    callback = SGDCallback()
    results = callback.evaluate_sgd(adata, "X_emb", "split", "cell_type")

    # Check that all expected metrics are present
    assert "accuracy" in results
    assert "balanced_accuracy" in results
    assert "f1" in results
    assert "precision" in results
    assert "recall" in results

    # Check that metrics are in valid range [0, 1]
    for metric, value in results.items():
        assert 0 <= value <= 1, f"{metric} should be between 0 and 1, got {value}"

    # With strong signal, accuracy should be very high
    assert (
        results["accuracy"] > 0.8
    ), "Accuracy should be high for well-separated classes"


def test_sgd_callback_confidence_intervals():
    """Test that SGDCallback calculates confidence intervals correctly."""
    from unittest.mock import MagicMock, patch

    import pandas as pd

    # Create mock AnnData with embeddings and labels
    n_samples = 100
    n_features = 10
    np.random.seed(42)

    # Create embeddings with strong signal
    X_emb = np.random.randn(n_samples, n_features) * 0.1
    X_emb[:50, 0] += 5  # TypeA
    X_emb[50:, 1] += 5  # TypeB

    labels = np.array(["TypeA"] * 50 + ["TypeB"] * 50)
    splits = np.array(["train"] * 40 + ["dev"] * 10 + ["test"] * 30 + ["train"] * 20)

    obs = pd.DataFrame({"cell_type": labels, "split": splits})
    adata = sc.AnnData(np.zeros((n_samples, 5)), obs=obs)
    adata.obsm["X_emb"] = X_emb

    # Mock trainer and pl_module
    mock_trainer = MagicMock()
    mock_trainer.datamodule.label_columns = [MagicMock(label_column_name="cell_type")]

    mock_pl_module = MagicMock()

    # Mock ClearML logger
    mock_logger = MagicMock()

    # Test with binomial CI method
    with patch(
        "bmfm_targets.training.callbacks.get_adata_with_embeddings", return_value=adata
    ), patch(
        "bmfm_targets.training.callbacks.Logger.current_logger",
        return_value=mock_logger,
    ):
        callback = SGDCallback(ci_method="binomial", obsm_key="X_emb")
        callback.on_predict_end(mock_trainer, mock_pl_module)

        # Check that report_table was called
        assert mock_logger.report_table.called
        call_args = mock_logger.report_table.call_args

        # Get the DataFrame that was reported
        reported_df = call_args[1]["table_plot"]

        # Check that the DataFrame has the expected structure
        # The table should NOT be transposed, so Metric should be a column
        assert "Metric" in reported_df.columns, "Metric should be a column"

        # Get metrics from the Metric column
        metrics = reported_df["Metric"].values

        # Check that all expected metrics are present
        expected_metrics = [
            "accuracy",
            "balanced_accuracy",
            "f1",
            "precision",
            "recall",
        ]
        for metric in expected_metrics:
            assert metric in metrics, f"Metric {metric} not found in reported table"

        # Check that CI column exists (should be a column, not in index)
        ci_col_name = "binomial CI [Lower bound, Upper bound]"
        assert ci_col_name in reported_df.columns, "Binomial CI should be a column"

        # Check that report_single_value was called for F1
        assert mock_logger.report_single_value.called
        single_value_calls = list(mock_logger.report_single_value.call_args_list)
        f1_calls = [call for call in single_value_calls if "SGD_f1" in str(call)]
        assert len(f1_calls) > 0, "F1 scalar should be reported"


def test_sgd_callback_wilson_ci():
    """Test that SGDCallback works with wilson CI method."""
    from unittest.mock import MagicMock, patch

    import pandas as pd

    # Create mock AnnData
    n_samples = 100
    n_features = 10
    np.random.seed(42)

    X_emb = np.random.randn(n_samples, n_features) * 0.1
    X_emb[:50, 0] += 5
    X_emb[50:, 1] += 5

    labels = np.array(["TypeA"] * 50 + ["TypeB"] * 50)
    splits = np.array(["train"] * 40 + ["dev"] * 10 + ["test"] * 30 + ["train"] * 20)

    obs = pd.DataFrame({"cell_type": labels, "split": splits})
    adata = sc.AnnData(np.zeros((n_samples, 5)), obs=obs)
    adata.obsm["X_emb"] = X_emb

    mock_trainer = MagicMock()
    mock_trainer.datamodule.label_columns = [MagicMock(label_column_name="cell_type")]
    mock_pl_module = MagicMock()
    mock_logger = MagicMock()

    # Test with wilson CI method
    with patch(
        "bmfm_targets.training.callbacks.get_adata_with_embeddings", return_value=adata
    ), patch(
        "bmfm_targets.training.callbacks.Logger.current_logger",
        return_value=mock_logger,
    ):
        callback = SGDCallback(ci_method="wilson", obsm_key="X_emb")
        callback.on_predict_end(mock_trainer, mock_pl_module)

        # Check that report_table was called
        assert mock_logger.report_table.called
        call_args = mock_logger.report_table.call_args
        reported_df = call_args[1]["table_plot"]

        # Check that wilson CI column exists (should be a column, not in index)
        ci_col_name = "wilson CI [Lower bound, Upper bound]"
        assert ci_col_name in reported_df.columns, "Wilson CI should be a column"
