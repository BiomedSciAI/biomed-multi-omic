import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from bmfm_targets.training.callbacks import (
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
