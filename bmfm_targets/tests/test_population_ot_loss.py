"""Tests for PopulationOTObjective and WCEDPopulationSource (TDD — written before implementation)."""

from __future__ import annotations

import pytest
import torch

import bmfm_targets.training.losses.objectives as obj_module
from bmfm_targets.config import FieldInfo
from bmfm_targets.training.losses import LossTask, loss_dict_to_task
from bmfm_targets.training.losses.objectives import PopulationOTObjective
from bmfm_targets.training.losses.sources import WCEDPopulationSource

# ─── Objective ────────────────────────────────────────────────────────────────


def test_population_ot_self_divergence_near_zero():
    """SD(X, X) ≈ 0 for identical point clouds."""
    obj = PopulationOTObjective(eps=1.0, n_iters=100)
    X = torch.randn(4, 8)
    result = obj.compute(X, X.clone())
    assert result is not None
    assert abs(result.item()) < 1e-3


def test_population_ot_positive_for_separated_clouds():
    """SD(X, Y) is finite and positive when X and Y are well-separated."""
    obj = PopulationOTObjective(eps=1.0, n_iters=100)
    X = torch.zeros(4, 8)
    Y = torch.ones(6, 8) * 10.0
    result = obj.compute(X, Y)
    assert result is not None
    assert torch.isfinite(result)
    assert result.item() > 0.0


def test_population_ot_returns_none_when_pred_batch_too_small():
    """compute() returns None when B < 2."""
    obj = PopulationOTObjective()
    pred = torch.randn(1, 8)
    target = torch.randn(6, 8)
    assert obj.compute(pred, target) is None


def test_population_ot_returns_none_when_target_too_small():
    """compute() returns None when M < 2."""
    obj = PopulationOTObjective()
    pred = torch.randn(4, 8)
    target = torch.randn(1, 8)
    assert obj.compute(pred, target) is None


# ─── Cache ────────────────────────────────────────────────────────────────────


def test_population_ot_cache_ot_yy_computed_once(monkeypatch):
    """
    OT(Y, Y) is computed only once across two calls with the same target.

    Without cache: 2 calls × 3 sinkhorn_cost each = 6 total.
    With cache:    first call = 3, second call = 2 (Y,Y served from cache) = 5 total.
    """
    total_calls = {"n": 0}
    _real = obj_module.sinkhorn_cost

    def _counting(X, Y, **kwargs):
        total_calls["n"] += 1
        return _real(X, Y, **kwargs)

    monkeypatch.setattr(obj_module, "sinkhorn_cost", _counting)

    obj = PopulationOTObjective(eps=0.5, n_iters=20)
    Y = torch.randn(6, 8)  # same target tensor reused across calls
    obj.compute(torch.randn(4, 8), Y)  # 3 calls: OT(X1,Y) + OT(X1,X1) + OT(Y,Y)
    obj.compute(torch.randn(4, 8), Y)  # 2 calls: OT(X2,Y) + OT(X2,X2) + [cached]

    assert total_calls["n"] == 5


# ─── Source ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def wced_field_multi():
    """FieldInfo with two-output WCED (mse at index 0, is_zero_bce at index 1)."""
    return FieldInfo(
        "label_expressions",
        vocab_size=8,
        decode_modes={
            "wced": {"vocab_field": "genes", "logit_outputs": ["mse", "is_zero_bce"]}
        },
    )


def test_wced_population_source_extract_logits_multi_output(wced_field_multi):
    """[B, T, V, 2]: selects MSE channel (index 0) at token 0 → [B, V]."""
    B, T, V, n = 4, 5, 8, 2
    logits_tensor = torch.randn(B, T, V, n)
    logits = {"label_expressions_wced": logits_tensor}

    src = WCEDPopulationSource("label_expressions", "all_genes", wced_output="mse")
    src.field = wced_field_multi
    src.decoder_output_index = 0  # mse → index 0 (would be set by resolve_schema)

    result = src.extract_logits(logits)
    assert result.shape == (B, V)
    torch.testing.assert_close(result, logits_tensor[:, 0, :, 0])


def test_wced_population_source_extract_logits_single_output():
    """[B, T, V]: no channel indexing, just token 0 → [B, V]."""
    field = FieldInfo(
        "label_expressions",
        vocab_size=8,
        decode_modes={"wced": {"vocab_field": "genes", "logit_outputs": ["mse"]}},
    )
    B, T, V = 4, 5, 8
    logits_tensor = torch.randn(B, T, V)
    logits = {"label_expressions_wced": logits_tensor}

    src = WCEDPopulationSource("label_expressions", "all_genes", wced_output="mse")
    src.field = field
    src.decoder_output_index = None  # single output → lookup returns None

    result = src.extract_logits(logits)
    assert result.shape == (B, V)
    torch.testing.assert_close(result, logits_tensor[:, 0, :])


def test_wced_population_source_extract_labels():
    """extract_labels returns the [M, V] population tensor keyed by population_key."""
    M, V = 6, 8
    population = torch.randn(M, V)
    # labels dict may also contain other keys that must be ignored
    labels = {
        "chip_population": population,
        "label_expressions": {"some_label_set": torch.zeros(4, 8)},
    }

    src = WCEDPopulationSource(
        "label_expressions", "all_genes", population_key="chip_population"
    )
    result = src.extract_labels(labels)
    assert result.shape == (M, V)
    torch.testing.assert_close(result, population)


# ─── End-to-end ───────────────────────────────────────────────────────────────


def test_loss_dict_to_task_population_ot_end_to_end():
    """Dict config → LossTask → finite scalar loss via calculate_loss."""
    B, M, V, T = 4, 6, 8, 5

    field = FieldInfo(
        "label_expressions",
        vocab_size=V,
        decode_modes={
            "wced": {"vocab_field": "genes", "logit_outputs": ["mse", "is_zero_bce"]}
        },
    )
    fields = [field]
    label_columns = []

    loss_config = {
        "field_name": "label_expressions",
        "name": "population_ot",
        "wced_target": "all_genes",
        "population_key": "chip_population",
        "eps": 1.0,
        "n_iters": 50,
        "cost": "euclidean",
        "link_function": None,
        "weight": 1.0,
    }

    task = loss_dict_to_task(loss_config, fields, label_columns)
    task.bind(fields, label_columns)

    assert isinstance(task, LossTask)

    logits = {"label_expressions_wced": torch.randn(B, T, V, 2)}
    mock_outputs = type("Outputs", (), {"logits": logits})()
    mock_batch = {"labels": {"chip_population": torch.randn(M, V)}}

    loss_val = task.calculate_loss(mock_outputs, mock_batch)
    assert loss_val is not None
    assert torch.isfinite(loss_val)


# Made with Bob
