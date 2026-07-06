"""
WP-D unit tests: generic perturbation/ChIP metrics wiring.

Tests:
  1. sample_metadata_keys includes configured perturbation_group_column
  2. Default config leaves sample_metadata_keys unchanged and uses perturbed_genes path
  3. log_perturbation_specific_metrics with tissue_label groups by configured column,
     does not call discrimination_score, does not raise on missing baseline columns
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import anndata as ad
import numpy as np
import pandas as pd

from bmfm_targets.config import FieldInfo, TrainerConfig
from bmfm_targets.config.model_config import SCBertConfig
from bmfm_targets.datasets.datasets_utils import make_group_means
from bmfm_targets.training.modules.sequence_labeling import (
    SequenceLabelingTrainingModule,
)

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _label_expr_field() -> FieldInfo:
    return FieldInfo(
        field_name="label_expressions",
        is_input=False,
        is_masked=False,
        decode_modes={
            "wced": {"vocab_field": "genes", "logit_outputs": ["mse"]},
        },
    )


def _genes_field(v: int = 10) -> FieldInfo:
    return FieldInfo(field_name="genes", is_masked=False, vocab_size=v)


def _make_tiny_sl_module(
    perturbation_group_column: str = "perturbed_genes",
    enable_perturbation_metrics: bool = True,
) -> SequenceLabelingTrainingModule:
    """Build a SequenceLabelingTrainingModule with a mocked model for unit tests."""
    fields = [_genes_field(), _label_expr_field()]
    model_config = SCBertConfig(
        fields=fields,
        label_columns=None,
        num_hidden_layers=1,
        num_attention_heads=1,
        hidden_size=8,
        intermediate_size=16,
    )
    trainer_config = TrainerConfig(
        perturbation_group_column=perturbation_group_column,
        enable_perturbation_metrics=enable_perturbation_metrics,
    )
    mock_model = MagicMock()
    with patch(
        "bmfm_targets.training.modules.base.get_model_from_config",
        return_value=mock_model,
    ):
        module = SequenceLabelingTrainingModule(
            model_config=model_config,
            trainer_config=trainer_config,
            tokenizer=None,
        )
    return module


def _small_tissue_group_means() -> ad.AnnData:
    """Build a small group_means AnnData (no Control, no avg row) for tissue test."""
    rng = np.random.default_rng(42)
    gene_names = [f"gene{i}" for i in range(4)]
    tissues = ["T cell"] * 3 + ["B cell"] * 3
    n = len(tissues)
    X = rng.random((n, len(gene_names)))
    cells = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {"tissue_label": tissues},
            index=[f"cell{i}" for i in range(n)],
        ),
        var=pd.DataFrame(index=gene_names),
    )
    return make_group_means(
        cells,
        perturbation_column_name="tissue_label",
        split_column_name=None,
        avg_row_label=None,
    )


def _stub_module_loggers(module: SequenceLabelingTrainingModule) -> None:
    """Replace all logging/plotting calls with MagicMocks."""
    module.log_aggregate_perturbation_metrics = MagicMock()
    module.log_table = MagicMock()
    module.log_mean_aggregate_perturbation_metrics = MagicMock()
    module.plot_agg_pred_vs_baseline_scatter = MagicMock()
    module.plot_heatmap = MagicMock()


# ─── Tests ────────────────────────────────────────────────────────────────────


def test_sample_metadata_keys_include_configured_identifier():
    """When perturbation_group_column='tissue_label', the module tracks tissue_label."""
    module = _make_tiny_sl_module(perturbation_group_column="tissue_label")
    assert "tissue_label" in module.sample_metadata_keys


def test_default_config_unchanged():
    """
    With default perturbation_group_column, sample_metadata_keys is the original list
    and the discrimination_score path is still invoked for perturbation data.
    """
    # Part 1: sample_metadata_keys is unchanged
    module = _make_tiny_sl_module(perturbation_group_column="perturbed_genes")
    assert module.sample_metadata_keys == ["cell_name", "seq_id", "perturbed_genes"]

    # Part 2: with "perturbed_genes", discrimination_score IS called
    rng = np.random.default_rng(0)
    gene_names = ["g0", "g1"]
    pert_groups = ["geneA", "Control", "Average_Perturbation_Train"]
    group_means_ad = ad.AnnData(
        X=rng.random((len(pert_groups), len(gene_names))).astype("float32"),
        obs=pd.DataFrame(index=pert_groups),
        var=pd.DataFrame(index=gene_names),
    )
    preds_df = pd.DataFrame(
        {
            "perturbed_genes": ["geneA", "geneA"],
            "input_genes": gene_names,
            "predicted_expressions": [1.0, 2.0],
            "input_expressions": [0.5, 0.8],
        }
    )
    module.kwargs = {"group_means": group_means_ad}
    _stub_module_loggers(module)

    with patch(
        "bmfm_targets.training.modules.sequence_labeling.pm.discrimination_score"
    ) as mock_disc:
        mock_disc.return_value = (
            {"geneA": 0.5},
            pd.DataFrame({"geneA": [0.0]}, index=["geneA"]),
        )
        module.log_perturbation_specific_metrics("test", preds_df)
        mock_disc.assert_called_once()


def test_log_perturbation_metrics_groups_by_configured_column():
    """
    log_perturbation_specific_metrics with tissue_label:
    - produces per-tissue agg_pcc without raising
    - does NOT invoke discrimination_score
    - does NOT raise on missing baseline columns.
    """
    group_means = _small_tissue_group_means()

    # preds_df: two tissues, two genes each
    preds_df = pd.DataFrame(
        {
            "tissue_label": ["T cell", "T cell", "B cell", "B cell"],
            "input_genes": ["gene0", "gene1", "gene0", "gene1"],
            "predicted_expressions": [1.0, 2.0, 3.0, 4.0],
        }
    )

    module = _make_tiny_sl_module(perturbation_group_column="tissue_label")
    module.kwargs = {"group_means": group_means}
    _stub_module_loggers(module)

    with patch(
        "bmfm_targets.training.modules.sequence_labeling.pm.discrimination_score"
    ) as mock_disc:
        module.log_perturbation_specific_metrics("validation", preds_df)
        mock_disc.assert_not_called()

    # agg metrics were computed (log_aggregate_perturbation_metrics called for each tissue)
    # 2 tissues, both <=10 conditions → called once per tissue
    assert module.log_aggregate_perturbation_metrics.call_count == 2
