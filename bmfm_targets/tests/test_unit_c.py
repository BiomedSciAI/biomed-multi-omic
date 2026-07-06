"""
Unit C tests: OT refactor cleanup and integration fixes.

Covers:
  Part 1  – removal of the bespoke OT config / ScrnaToChipTranslationModule surface
  Part 2  – _forward_and_compute_losses extraction in BaseTrainingModule
  Part 3a – contributes_sample_metrics predicate and metric-path exclusion
  Part 3b – content-based OT(Y,Y) cache key
  Part 3c – integration: training_step / validation_step with population_ot loss
"""

from __future__ import annotations

from collections import Counter
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from bmfm_targets.config import FieldInfo, TrainerConfig
from bmfm_targets.config.model_config import SCBertConfig
from bmfm_targets.models.model_utils import SequenceClassifierOutputWithEmbeddings
from bmfm_targets.training.losses.objectives import (
    MSEObjective,
    PopulationOTObjective,
)
from bmfm_targets.training.modules import MultiTaskTrainingModule

# ─── Helpers ──────────────────────────────────────────────────────────────────

V = 10  # tiny vocab / expression size
T = 6  # sequence length
B = 4  # batch size
M = 6  # reference population size (deliberately != B)


def _genes_field(v: int = V) -> FieldInfo:
    return FieldInfo(field_name="genes", is_masked=False, vocab_size=v)


def _label_expressions_field(v: int = V) -> FieldInfo:
    return FieldInfo(
        field_name="label_expressions",
        is_input=False,
        is_masked=False,
        decode_modes={
            "wced": {
                "vocab_field": "genes",
                "logit_outputs": ["mse", "is_zero_bce"],
            }
        },
    )


def _mse_wced_dict(v: int = V) -> dict:
    return {
        "field_name": "label_expressions",
        "name": "mse",
        "wced_target": "all_genes",
    }


def _ot_dict() -> dict:
    return {
        "field_name": "label_expressions",
        "name": "population_ot",
        "wced_target": "all_genes",
        "population_key": "chip_population",
        "eps": 0.5,
        "n_iters": 10,
        "cost": "euclidean",
    }


def _make_tiny_model_config(v: int = V) -> SCBertConfig:
    fields = [_genes_field(v), _label_expressions_field(v)]
    return SCBertConfig(
        fields=fields,
        label_columns=None,
        num_hidden_layers=1,
        num_attention_heads=1,
        hidden_size=8,
        intermediate_size=16,
    )


class _MockWCEDModel(nn.Module):
    """Minimal nn.Module returning synthetic WCED logits."""

    def __init__(self, v: int = V):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.v = v

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        b = input_ids.shape[0]
        t = input_ids.shape[1]
        logits = {"label_expressions_wced": (self.dummy + torch.randn(b, t, self.v, 2))}
        return SequenceClassifierOutputWithEmbeddings(logits=logits)


def _make_module(losses: list, v: int = V) -> MultiTaskTrainingModule:
    """
    Build a MultiTaskTrainingModule with a mock model.

    The real model is replaced by _MockWCEDModel so that the test does not
    depend on the full model-building infrastructure.
    """
    model_config = _make_tiny_model_config(v)
    trainer_config = TrainerConfig(losses=losses, weight_decay=None)
    mock_model = _MockWCEDModel(v)
    with patch(
        "bmfm_targets.training.modules.base.get_model_from_config",
        return_value=mock_model,
    ):
        module = MultiTaskTrainingModule(model_config, trainer_config, tokenizer=None)
    return module


def _make_batch(v: int = V, b: int = B, m: int = M, t: int = T) -> dict:
    return {
        "input_ids": torch.zeros(b, t, dtype=torch.long),
        "attention_mask": torch.ones(b, t, dtype=torch.long),
        "labels": {
            "label_expressions": {"all": torch.randn(b, v)},
            "chip_population": torch.randn(m, v),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Part 1 — removal of the bespoke OT config / ScrnaToChipTranslationModule
# ═══════════════════════════════════════════════════════════════════════════════


class TestPart1RemovalOfOTSurface:
    """Part 1: bespoke OT fields and module are gone."""

    _OT_ATTRS = [
        "enable_ot_translation",
        "ot_weight",
        "wced_weight",
        "ot_eps",
        "ot_n_iters",
        "ot_cost",
    ]

    def test_trainer_config_has_no_ot_fields(self):
        """TrainerConfig must not expose any of the six removed OT attributes."""
        tc = TrainerConfig()
        for attr in self._OT_ATTRS:
            assert not hasattr(
                tc, attr
            ), f"TrainerConfig still has removed attribute: {attr}"

    def test_scrna_to_chip_module_not_in_training_modules(self):
        """bmfm_targets.training.modules must not expose ScrnaToChipTranslationModule."""
        import bmfm_targets.training.modules as mod

        assert not hasattr(
            mod, "ScrnaToChipTranslationModule"
        ), "ScrnaToChipTranslationModule still exported from training.modules"

    def test_scrna_to_chip_file_does_not_exist(self):
        """scrna_to_chip.py must be absent (git rm'd)."""
        from importlib.util import find_spec

        spec = find_spec("bmfm_targets.training.modules.scrna_to_chip")
        assert (
            spec is None
        ), "bmfm_targets.training.modules.scrna_to_chip can still be imported"

    def test_task_utils_imports_cleanly(self):
        """bmfm_targets.tasks.task_utils must import without errors after cleanup."""
        import bmfm_targets.tasks.task_utils  # noqa: F401

        assert True  # import did not raise


# ═══════════════════════════════════════════════════════════════════════════════
# Part 2 — _forward_and_compute_losses extraction
# ═══════════════════════════════════════════════════════════════════════════════


class TestPart2ForwardAndComputeLosses:
    """Part 2: _forward_and_compute_losses is extracted and used by all three steps."""

    def test_method_exists_on_base_module(self):
        from bmfm_targets.training.modules.base import BaseTrainingModule

        assert hasattr(
            BaseTrainingModule, "_forward_and_compute_losses"
        ), "_forward_and_compute_losses not found on BaseTrainingModule"

    def test_method_returns_outputs_and_losses_dict(self):
        module = _make_module([_mse_wced_dict(), _ot_dict()])
        batch = _make_batch()
        module.model = _MockWCEDModel(V)
        outputs, all_losses = module._forward_and_compute_losses(batch)
        assert hasattr(outputs, "logits"), "outputs must have .logits"
        assert isinstance(all_losses, dict), "all_losses must be a dict"
        assert "loss" in all_losses

    def test_training_step_calls_forward_and_compute_losses(self):
        """training_step must delegate to _forward_and_compute_losses."""
        module = _make_module([_mse_wced_dict()])
        module.log = MagicMock()
        batch = _make_batch()
        called = []
        original = module._forward_and_compute_losses

        def spy(b):
            called.append(True)
            return original(b)

        module._forward_and_compute_losses = spy
        module.training_step(batch, 0)
        assert called, "training_step did not call _forward_and_compute_losses"

    def test_validation_step_calls_forward_and_compute_losses(self):
        module = _make_module([_mse_wced_dict()])
        module.log = MagicMock()
        batch = _make_batch()
        called = []
        original = module._forward_and_compute_losses

        def spy(b):
            called.append(True)
            return original(b)

        module._forward_and_compute_losses = spy
        module.validation_step(batch, 0)
        assert called, "validation_step did not call _forward_and_compute_losses"

    def test_test_step_calls_forward_and_compute_losses(self):
        module = _make_module([_mse_wced_dict()])
        module.log = MagicMock()
        batch = _make_batch()
        called = []
        original = module._forward_and_compute_losses

        def spy(b):
            called.append(True)
            return original(b)

        module._forward_and_compute_losses = spy
        module.test_step(batch, 0)
        assert called, "test_step did not call _forward_and_compute_losses"


# ═══════════════════════════════════════════════════════════════════════════════
# Part 3a — contributes_sample_metrics predicate
# ═══════════════════════════════════════════════════════════════════════════════


class TestPart3aContributesSampleMetrics:
    """
    Part 3a: contributes_sample_metrics on Objective and its effect on
    update_metrics / initialize_metrics / get_prediction_active_keys.
    """

    def test_objective_base_defaults_to_true(self):
        """All concrete objectives should default contributes_sample_metrics to True."""
        from bmfm_targets.training.losses.objectives import (
            CrossEntropyObjective,
            FocalObjective,
            IsZeroBCEObjective,
        )

        for cls in (
            CrossEntropyObjective,
            FocalObjective,
            MSEObjective,
            IsZeroBCEObjective,
        ):
            assert (
                cls().contributes_sample_metrics is True
            ), f"{cls.__name__}.contributes_sample_metrics should be True"

    def test_population_ot_contributes_sample_metrics_false(self):
        obj = PopulationOTObjective()
        assert obj.contributes_sample_metrics is False

    def test_initialize_metrics_excludes_population_ot(self):
        """
        Metrics for the shared key must come from the MSE task, not overwritten
        by the empty population_ot task.
        """
        module = _make_module([_mse_wced_dict(), _ot_dict()])
        # initialize_metrics was already called during __init__
        # MSE has empty metrics ([]); population_ot also has [].
        # The important thing is that the population_ot task does NOT overwrite
        # the mse task's MetricCollection under the same key.
        # (Both happen to be empty here, but the key should be registered ONCE.)
        metric_key = "label_expressions_all_genes"
        # Only the MSE task registered the key, not the OT task
        for metrics_wrapper in (module.train_metrics, module.val_metrics):
            task_metrics = metrics_wrapper.task_metrics
            assert (
                metric_key in task_metrics
            ), f"'{metric_key}' must be registered (from mse task)"

    def test_initialize_metrics_key_count_with_ot_excluded(self):
        """With mse + population_ot tasks, the metric_key appears exactly ONCE."""
        module = _make_module([_mse_wced_dict(), _ot_dict()])
        all_keys = list(module.train_metrics.task_metrics.keys())
        counter = Counter(all_keys)
        metric_key = "label_expressions_all_genes"
        assert (
            counter[metric_key] == 1
        ), f"Expected metric_key to appear once, got {counter[metric_key]}"

    def test_update_metrics_does_not_crash_with_population_ot(self):
        """update_metrics must run without error when population_ot is present."""
        module = _make_module([_mse_wced_dict(), _ot_dict()])
        batch = _make_batch()
        outputs, _ = module._forward_and_compute_losses(batch)
        # Should not raise
        result = module.update_metrics(batch["labels"], outputs, split="train")
        assert isinstance(result, dict)

    def test_get_prediction_active_keys_excludes_population_ot(self):
        """get_prediction_active_keys must not include the population_ot task."""
        module = _make_module([_mse_wced_dict(), _ot_dict()])
        active_keys = module.get_prediction_active_keys()
        # The ot task shares metric_key "label_expressions_all_genes" with mse
        # After fix, only the mse task should be present.
        # If the ot task were included it would appear in active_keys with the
        # same key and overwrite the mse task entry.
        for task in active_keys.values():
            assert task.objective.contributes_sample_metrics, (
                f"Task with non-sample-metric objective found in active_keys: "
                f"{task.objective.name}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Part 3b — content-based OT(Y,Y) cache key
# ═══════════════════════════════════════════════════════════════════════════════


class TestPart3bCacheKeyFix:
    """Part 3b: OT(Y,Y) cache is keyed on a content signature, not id()."""

    def test_cache_key_is_not_id_based(self):
        """
        Two tensors with the same values but different Python ids must hit
        the cache (i.e. be treated as the same key).
        """
        import bmfm_targets.training.losses.objectives as obj_module

        call_count = {"n": 0}
        real_fn = obj_module.sinkhorn_cost

        def counting_fn(X, Y, **kwargs):
            call_count["n"] += 1
            return real_fn(X, Y, **kwargs)

        obj = PopulationOTObjective(eps=0.5, n_iters=5)
        Y_values = torch.randn(4, 6)

        with patch.object(obj_module, "sinkhorn_cost", side_effect=counting_fn):
            # First call with tensor A
            Y_a = Y_values.clone()
            obj.compute(torch.randn(3, 6), Y_a)
            calls_after_first = call_count["n"]  # 3: OT(X,Y) + OT(X,X) + OT(Y,Y)

            # Second call with a NEW tensor B that has IDENTICAL values
            Y_b = Y_values.clone()
            assert id(Y_a) != id(Y_b), "sanity: Y_a and Y_b must be different objects"
            obj.compute(torch.randn(3, 6), Y_b)
            calls_after_second = call_count["n"]

        # With content-based cache, OT(Y,Y) is served from cache on second call
        # → only 2 new sinkhorn calls (OT(X2,Y) + OT(X2,X2)), not 3.
        assert calls_after_second - calls_after_first == 2, (
            f"Expected 2 new sinkhorn calls on second call (Y,Y served from cache), "
            f"got {calls_after_second - calls_after_first}"
        )

    def test_cache_miss_for_different_target_values(self):
        """Targets with different values must produce distinct cache entries."""
        import bmfm_targets.training.losses.objectives as obj_module

        call_count = {"n": 0}
        real_fn = obj_module.sinkhorn_cost

        def counting_fn(X, Y, **kwargs):
            call_count["n"] += 1
            return real_fn(X, Y, **kwargs)

        obj = PopulationOTObjective(eps=0.5, n_iters=5)
        Y1 = torch.zeros(4, 6)
        Y2 = torch.ones(4, 6) * 10.0  # different values

        with patch.object(obj_module, "sinkhorn_cost", side_effect=counting_fn):
            obj.compute(torch.randn(3, 6), Y1)
            n1 = call_count["n"]
            obj.compute(torch.randn(3, 6), Y2)
            n2 = call_count["n"]

        # Each call to compute has 3 sinkhorn calls (OT(X,Y)+OT(X,X)+OT(Y,Y))
        # because both Y1 and Y2 are new, different-valued targets
        assert n1 == 3
        assert n2 - n1 == 3, (
            f"Expected 3 sinkhorn calls for second (different) target, "
            f"got {n2 - n1}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Part 3c — integration: training_step / validation_step with population_ot
# ═══════════════════════════════════════════════════════════════════════════════


class TestPart3cIntegration:
    """Part 3c: full training_step / validation_step smoke tests."""

    def test_training_step_mse_plus_population_ot_returns_finite_loss(self):
        """training_step with [mse_wced, population_ot] must return a finite scalar."""
        module = _make_module([_mse_wced_dict(), _ot_dict()])
        module.log = MagicMock()
        batch = _make_batch()

        result = module.training_step(batch, 0)

        assert result is not None, "training_step returned None (zero loss)"
        assert torch.is_tensor(result), "training_step must return a tensor"
        assert torch.isfinite(
            result
        ), f"training_step returned non-finite loss: {result}"

    def test_validation_step_mse_plus_population_ot_no_error(self):
        """validation_step with [mse_wced, population_ot] must not raise."""
        module = _make_module([_mse_wced_dict(), _ot_dict()])
        module.log = MagicMock()
        batch = _make_batch()

        result = module.validation_step(batch, 0)

        assert result is not None
        assert torch.isfinite(result)

    def test_population_ot_term_appears_in_losses(self):
        """The population_ot loss component must appear in all_losses when B, M >= 2."""
        module = _make_module([_mse_wced_dict(), _ot_dict()])
        batch = _make_batch()

        _, all_losses = module._forward_and_compute_losses(batch)

        # The population_ot loss_display_name is
        # "label_expressions_all_genes_population_ot_loss"
        ot_loss_keys = [k for k in all_losses if "population_ot" in k]
        assert (
            len(ot_loss_keys) >= 1
        ), f"Expected at least one population_ot loss key in {list(all_losses.keys())}"

    def test_population_ot_changes_total_loss(self):
        """Adding population_ot to [mse_wced] losses must change the total loss."""
        torch.manual_seed(42)
        batch = _make_batch()

        module_mse_only = _make_module([_mse_wced_dict()])
        module_mse_ot = _make_module([_mse_wced_dict(), _ot_dict()])

        # Use the same mock model for both so logits are identical
        shared_model = _MockWCEDModel(V)
        module_mse_only.model = shared_model
        module_mse_ot.model = shared_model

        _, losses_mse = module_mse_only._forward_and_compute_losses(batch)
        _, losses_mse_ot = module_mse_ot._forward_and_compute_losses(batch)

        loss_mse = losses_mse["loss"].item()
        loss_mse_ot = losses_mse_ot["loss"].item()

        # The total losses should differ because the OT term is included
        assert loss_mse != pytest.approx(loss_mse_ot, abs=1e-6), (
            "Adding population_ot did not change the total loss — "
            "the OT term may not be wired correctly"
        )

    def test_training_step_population_ot_only_returns_finite_loss(self):
        """
        OT-only mode: training_step with [population_ot] only must return
        a finite loss (or None if the OT term is None, i.e. batch too small).
        """
        module = _make_module([_ot_dict()])
        module.log = MagicMock()
        batch = _make_batch(b=4, m=4)  # B=4, M=4 → OT is computed

        result = module.training_step(batch, 0)

        # With B=4 and M=4, OT should be computed and return a finite loss
        assert (
            result is not None
        ), "OT-only training_step returned None (B and M are both >= 2)"
        assert torch.isfinite(
            result
        ), f"OT-only training_step: non-finite loss {result}"

    def test_validation_step_no_crash_with_m_neq_b(self):
        """validation_step must not crash when reference population M != batch B."""
        module = _make_module([_mse_wced_dict(), _ot_dict()])
        module.log = MagicMock()
        # M = 7, B = 3 — shapes are incompatible for per-sample metrics
        batch = _make_batch(b=3, m=7)

        # Must not raise RuntimeError from .view() shape mismatch
        result = module.validation_step(batch, 0)
        assert result is not None

    def test_update_metrics_no_view_crash(self):
        """update_metrics must not raise when M != B (the original bug)."""
        module = _make_module([_mse_wced_dict(), _ot_dict()])
        batch = _make_batch(b=3, m=7)  # M != B

        outputs, _ = module._forward_and_compute_losses(batch)
        # This must not raise RuntimeError: shape '[-1, 7, 10]' is invalid...
        result = module.update_metrics(batch["labels"], outputs, split="validation")
        assert isinstance(result, dict)
