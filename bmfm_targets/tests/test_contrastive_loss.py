"""
Tests for scConcept-on-WCED contrastive loss components.

Tests follow test-against-oracle discipline: assert known-answer cases, never
re-derive the loss formula in the test.
"""
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

# ─── ContrastiveObjective ──────────────────────────────────────────────────


class TestContrastiveObjective:
    def _make_z(self, N, D=32, identical=False):
        za = F.normalize(torch.randn(N, D), dim=-1)
        zb = za.clone() if identical else F.normalize(torch.randn(N, D), dim=-1)
        return torch.cat([za, zb], dim=0)

    def test_identical_views_recall1(self):
        """Identical views -> recall@1 == 1 (perfect alignment)."""
        from bmfm_targets.training.losses.objectives import ContrastiveObjective

        obj = ContrastiveObjective(
            temperature=0.07, symmetric=True, gather_distributed=False
        )
        z = self._make_z(8, D=32, identical=True)
        loss = obj.compute(z, None)
        assert torch.isfinite(loss)
        assert hasattr(obj, "_last_recall_at1")
        assert obj._last_recall_at1 == pytest.approx(1.0)

    def test_random_views_positive_loss(self):
        """Random (unrelated) views -> finite positive loss."""
        from bmfm_targets.training.losses.objectives import ContrastiveObjective

        obj = ContrastiveObjective(
            temperature=0.07, symmetric=True, gather_distributed=False
        )
        torch.manual_seed(0)
        z = self._make_z(16, D=32, identical=False)
        loss = obj.compute(z, None)
        assert torch.isfinite(loss)
        assert loss.item() > 0.0

    def test_small_batch_returns_none(self):
        """N < 2 -> compute returns None (batch too small)."""
        from bmfm_targets.training.losses.objectives import ContrastiveObjective

        obj = ContrastiveObjective(temperature=0.07)
        # 2N=2 -> N=1, which is < 2
        z = torch.randn(2, 32)
        assert obj.compute(z, None) is None

    def test_learnable_scale_gradient(self):
        """Learnable scale tensor (logit_scale) receives non-zero gradient."""
        from bmfm_targets.training.losses.objectives import ContrastiveObjective

        obj = ContrastiveObjective(temperature=None, gather_distributed=False)
        z = self._make_z(8, D=32, identical=False)
        scale = torch.nn.Parameter(torch.tensor(float(np.log(1 / 0.07))))
        loss = obj.compute(z, scale)
        loss.backward()
        assert scale.grad is not None
        assert scale.grad.abs().item() > 0.0

    def test_contributes_sample_metrics_false(self):
        """ContrastiveObjective must not contribute to per-sample metrics."""
        from bmfm_targets.training.losses.objectives import ContrastiveObjective

        obj = ContrastiveObjective()
        assert obj.contributes_sample_metrics is False

    def test_single_process_gather_noop(self):
        """gather_distributed=True in single process gives same result as False."""
        from bmfm_targets.training.losses.objectives import ContrastiveObjective

        torch.manual_seed(42)
        z = self._make_z(8, D=32, identical=False)
        obj_gather = ContrastiveObjective(temperature=0.07, gather_distributed=True)
        obj_nogather = ContrastiveObjective(temperature=0.07, gather_distributed=False)
        loss_gather = obj_gather.compute(z.clone(), None)
        loss_nogather = obj_nogather.compute(z.clone(), None)
        assert loss_gather.item() == pytest.approx(loss_nogather.item(), rel=1e-5)

    def test_recall5_set_and_bounded(self):
        """Compute stashes recall@5 in [0, 1] (identical views -> 1)."""
        from bmfm_targets.training.losses.objectives import ContrastiveObjective

        obj = ContrastiveObjective(temperature=0.07, gather_distributed=False)
        z = self._make_z(8, D=32, identical=True)
        obj.compute(z, None)
        assert obj._last_recall_at5 == pytest.approx(1.0)
        assert 0.0 <= obj._last_recall_at5 <= 1.0


class TestBothBatchContrastive:
    """The 2N×2N combined loss (scConcept steady-state, model.py loss_switch_step)."""

    def _make_z(self, N, D=32, identical=False):
        za = F.normalize(torch.randn(N, D), dim=-1)
        zb = za.clone() if identical else F.normalize(torch.randn(N, D), dim=-1)
        return torch.cat([za, zb], dim=0)

    def test_identical_views_recall1(self):
        """Identical views -> recall@1 == 1 in combined mode."""
        from bmfm_targets.training.losses.objectives import ContrastiveObjective

        obj = ContrastiveObjective(
            temperature=0.07, gather_distributed=False, both_batch=True
        )
        z = self._make_z(8, D=32, identical=True)
        loss = obj.compute(z, None)
        assert torch.isfinite(loss)
        assert obj._last_recall_at1 == pytest.approx(1.0)
        assert obj._last_recall_at5 == pytest.approx(1.0)

    def test_excludes_self_match(self):
        """
        Oracle: recall@1 == 1 even when each anchor's self-similarity (1.0)
        strictly exceeds its positive-pair similarity (~0.894).

        This can only hold if the trivial self-match at column (i+N) mod 2N is
        excluded from the 2N×2N gallery — the load-bearing detail of the
        combined loss. Without exclusion, every anchor's argmax would be its own
        embedding (the wrong column), so recall@1 would be 0.
        """
        from bmfm_targets.training.losses.objectives import ContrastiveObjective

        N = 6
        za = torch.eye(N)  # orthonormal rows e_0..e_{N-1}
        zb = torch.eye(N).clone()
        for i in range(N):
            zb[i, (i + 1) % N] += 0.5  # positive sim = 1/sqrt(1.25) ≈ 0.894 < 1.0
        z = torch.cat([za, zb], dim=0)

        obj = ContrastiveObjective(
            temperature=1.0, gather_distributed=False, both_batch=True
        )
        obj.compute(z, None)
        assert obj._last_recall_at1 == pytest.approx(1.0)

    def test_more_negatives_than_simple(self):
        """
        Oracle: for the same embeddings, the combined loss is strictly larger
        than the simple N×N loss.

        Each anchor in the combined loss keeps its positive but competes against
        a strict superset of the simple loss's negatives (the other views of
        every other cell too), so its cross-entropy denominator — and thus the
        loss — is strictly larger. Independent of the exact loss formula.
        """
        from bmfm_targets.training.losses.objectives import ContrastiveObjective

        torch.manual_seed(0)
        z = self._make_z(32, D=64, identical=False)
        simple = ContrastiveObjective(temperature=0.07, gather_distributed=False)
        combined = ContrastiveObjective(
            temperature=0.07, gather_distributed=False, both_batch=True
        )
        loss_simple = simple.compute(z.clone(), None)
        loss_combined = combined.compute(z.clone(), None)
        assert loss_combined.item() > loss_simple.item()

    def test_learnable_scale_gradient(self):
        """Combined mode: gradient reaches the learnable logit_scale."""
        from bmfm_targets.training.losses.objectives import ContrastiveObjective

        obj = ContrastiveObjective(
            temperature=None, gather_distributed=False, both_batch=True
        )
        z = self._make_z(8, D=32, identical=False)
        scale = torch.nn.Parameter(torch.tensor(float(np.log(1 / 0.07))))
        loss = obj.compute(z, scale)
        loss.backward()
        assert scale.grad is not None
        assert scale.grad.abs().item() > 0.0

    def test_small_batch_returns_none(self):
        """N < 2 -> None (skipped by calculate_losses), same guard as simple."""
        from bmfm_targets.training.losses.objectives import ContrastiveObjective

        obj = ContrastiveObjective(both_batch=True)
        assert obj.compute(torch.randn(2, 32), None) is None


# ─── CellEmbeddingContrastiveSource ────────────────────────────────────────


class TestCellEmbeddingContrastiveSource:
    def _mock_outputs(self, field="expressions", D=32, N=8, V=50):
        z = torch.randn(2 * N, D)
        scale = torch.tensor(float(np.log(1 / 0.07)))
        # wced output: [2N, V, D] -- source does cloud[:, 0, :] to get first-token row
        wced_out = torch.randn(2 * N, V, D)
        logits = {
            f"{field}_contrastive": z,
            f"{field}_contrastive_scale": scale,
            f"{field}_wced": wced_out,
        }
        out = MagicMock()
        out.logits = logits
        return out

    def test_extract_cls_target(self):
        """contrast_target='cls' pulls {field}_contrastive from logits."""
        from bmfm_targets.training.losses.sources import CellEmbeddingContrastiveSource

        src = CellEmbeddingContrastiveSource("expressions", contrast_target="cls")
        out = self._mock_outputs()
        z, scale = src.extract(out, {})
        assert z.shape == (16, 32)
        assert scale is not None

    def test_extract_cls_scale_present(self):
        """Extract returns the learnable scale tensor when present."""
        from bmfm_targets.training.losses.sources import CellEmbeddingContrastiveSource

        src = CellEmbeddingContrastiveSource("expressions", contrast_target="cls")
        out = self._mock_outputs(D=16)
        z, scale = src.extract(out, {})
        assert scale is not None
        assert isinstance(scale, torch.Tensor)

    def test_extract_wced_target_shape(self):
        """contrast_target='wced' returns [2N, D] from cloud[:, 0, :]."""
        from bmfm_targets.training.losses.sources import CellEmbeddingContrastiveSource

        # _wced mock is [2N, V, D]; source does cloud[:, 0, :] -> [2N, D]
        src = CellEmbeddingContrastiveSource("expressions", contrast_target="wced")
        out = self._mock_outputs(D=32, N=8, V=50)
        z, scale = src.extract(out, {})
        assert z is not None
        assert z.shape == (16, 32)  # [2*N, D]

    def test_source_name(self):
        """source.name returns field_name."""
        from bmfm_targets.training.losses.sources import CellEmbeddingContrastiveSource

        src = CellEmbeddingContrastiveSource("expressions")
        assert src.name == "expressions"


# ─── _PairedViewCollator ────────────────────────────────────────────────────


class TestPairedViewCollator:
    def _make_mfi(self, n_genes=100):
        from bmfm_targets.tokenization.multifield_instance import MultiFieldInstance

        return MultiFieldInstance(
            data={
                "genes": np.arange(n_genes),
                "expressions": np.random.randn(n_genes).astype(np.float32),
            },
            metadata={"celltype": "T cell"},
        )

    def test_block_layout_2n(self):
        """N examples -> 2N rows forwarded to base_collate."""
        from bmfm_targets.datasets.paired_view_collator import _PairedViewCollator

        collected = []

        def base_collate(examples):
            collected.extend(examples)
            return {"n": len(examples)}

        collator = _PairedViewCollator(
            base_collate, min_frac=0.4, max_frac=0.6, overlap_prob=0.0, seed=0
        )
        examples = [self._make_mfi(100) for _ in range(8)]
        result = collator(examples)
        assert result["n"] == 16  # 2N

    def test_disjoint_panels(self):
        """overlap_prob=0 -> viewA and viewB gene sets are disjoint."""
        from bmfm_targets.datasets.paired_view_collator import _PairedViewCollator

        collected_views = []

        def base_collate(examples):
            collected_views.extend(examples)
            return {"n": len(examples)}

        collator = _PairedViewCollator(
            base_collate, min_frac=0.4, max_frac=0.6, overlap_prob=0.0, seed=42
        )
        N = 4
        examples = [self._make_mfi(100) for _ in range(N)]
        collator(examples)

        assert len(collected_views) == 2 * N
        for i in range(N):
            genes_a = set(collected_views[i].data["genes"].tolist())
            genes_b = set(collected_views[N + i].data["genes"].tolist())
            assert genes_a.isdisjoint(
                genes_b
            ), f"Views {i} overlap: {genes_a & genes_b}"

    def test_deterministic_with_seed(self):
        """Fixed seed -> identical paired-view splits across two collators."""
        from bmfm_targets.datasets.paired_view_collator import _PairedViewCollator

        def base_collate(ex):
            return [e.data["genes"].tolist() for e in ex]

        np.random.seed(1)
        examples = [self._make_mfi(100) for _ in range(4)]
        c1 = _PairedViewCollator(base_collate, seed=7)
        c2 = _PairedViewCollator(base_collate, seed=7)
        assert c1(examples) == c2(examples)

    def test_panel_size_within_frac_bounds(self):
        """View-A panel size stays in [min_frac*n, max_frac*n] for both samplers."""
        from bmfm_targets.datasets.paired_view_collator import _PairedViewCollator

        collected = []

        def base_collate(ex):
            collected.extend(ex)
            return {"n": len(ex)}

        n_genes = 200
        min_frac, max_frac = 0.25, 0.75
        for sampling in ("log_uniform", "uniform"):
            collected.clear()
            collator = _PairedViewCollator(
                base_collate,
                min_frac=min_frac,
                max_frac=max_frac,
                overlap_prob=0.0,
                panel_size_sampling=sampling,
                seed=3,
            )
            N = 40
            collator([self._make_mfi(n_genes) for _ in range(N)])
            lo = int(min_frac * n_genes)
            hi = int(max_frac * n_genes)
            for i in range(N):  # view A occupies the first N slots
                size_a = len(collected[i].data["genes"])
                assert lo <= size_a <= hi, f"{sampling}: {size_a} outside [{lo},{hi}]"

    def test_log_uniform_biases_smaller_than_uniform(self):
        """
        Oracle: log-uniform panel sizes are on average smaller than uniform.

        The reference samples panel size log-uniformly (collate.log_int_samping),
        which puts more mass on small, sparse panels. Averaged over many cells,
        the mean log-uniform size must be below the mean uniform size.
        """
        from bmfm_targets.datasets.paired_view_collator import _PairedViewCollator

        def sizes(sampling):
            collected = []

            def base_collate(ex):
                collected.extend(ex)
                return None

            collator = _PairedViewCollator(
                base_collate,
                min_frac=0.05,
                max_frac=0.95,
                overlap_prob=0.0,
                panel_size_sampling=sampling,
                seed=11,
            )
            N = 200
            collator([self._make_mfi(500) for _ in range(N)])
            return np.mean([len(collected[i].data["genes"]) for i in range(N)])

        assert sizes("log_uniform") < sizes("uniform")

    def test_invalid_panel_size_sampling_raises(self):
        from bmfm_targets.datasets.paired_view_collator import _PairedViewCollator

        with pytest.raises(ValueError, match="panel_size_sampling"):
            _PairedViewCollator(lambda x: x, panel_size_sampling="nonsense")


# ─── ConditionHomogeneousBatchSampler ───────────────────────────────────────


class TestConditionHomogeneousBatchSampler:
    def _make_obs(self, n_per_cond=30, conds=("A", "B", "C")):
        labels = []
        for c in conds:
            labels.extend([c] * n_per_cond)
        return pd.Series(labels)

    def test_batch_condition_homogeneous(self):
        """Every index in a yielded batch shares the same condition label."""
        from bmfm_targets.datasets.samplers import ConditionHomogeneousBatchSampler

        obs = self._make_obs()
        sampler = ConditionHomogeneousBatchSampler(
            obs, batch_size=10, num_batches=30, seed=0
        )
        cond_map = {i: obs.iloc[i] for i in range(len(obs))}
        for batch in sampler:
            conds = {cond_map[i] for i in batch}
            assert len(conds) == 1, f"Batch not condition-homogeneous: {conds}"

    def test_replacement_false_distinct_indices(self):
        """replacement=False yields batches with no repeated indices when pool is large enough."""
        from bmfm_targets.datasets.samplers import ConditionHomogeneousBatchSampler

        obs = self._make_obs()
        sampler = ConditionHomogeneousBatchSampler(
            obs, batch_size=10, num_batches=20, seed=1, replacement=False
        )
        for batch in sampler:
            assert len(set(batch)) == len(
                batch
            ), "replacement=False must yield distinct indices"

    def test_replacement_true_allows_repeats(self):
        """replacement=True may repeat indices when the pool is small."""
        from bmfm_targets.datasets.samplers import ConditionHomogeneousBatchSampler

        # Only 3 samples for condition "X" -> with replacement, repeats are expected
        obs = pd.Series(["X"] * 3 + ["Y"] * 30)
        sampler = ConditionHomogeneousBatchSampler(
            obs, batch_size=10, num_batches=50, seed=2, replacement=True
        )
        x_batches = [b for b in sampler if all(i < 3 for i in b)]
        # With only 3 pool items and batch_size=10, any X batch must repeat
        has_repeats = any(len(set(b)) < len(b) for b in x_batches)
        assert (
            has_repeats
        ), "replacement=True should produce repeated indices for small pools"

    def test_yields_correct_batch_count(self):
        """Sampler yields exactly num_batches batches."""
        from bmfm_targets.datasets.samplers import ConditionHomogeneousBatchSampler

        obs = self._make_obs()
        num_batches = 15
        sampler = ConditionHomogeneousBatchSampler(
            obs, batch_size=5, num_batches=num_batches, seed=3
        )
        assert len(list(sampler)) == num_batches
        assert len(sampler) == num_batches


# ─── Compat integration ─────────────────────────────────────────────────────


class TestCompatIntegration:
    def test_loss_dict_creates_contrastive_task(self):
        """loss_dict_to_task with loss_name=contrastive returns a LossTask with ContrastiveObjective."""
        from bmfm_targets.config.tokenization_config import FieldInfo
        from bmfm_targets.training.losses.compat import loss_dict_to_task
        from bmfm_targets.training.losses.objectives import ContrastiveObjective

        field = FieldInfo(
            field_name="expressions",
            vocab_size=200,
            decode_modes={
                "contrastive": {
                    "projection_dim": 32,
                    "learnable_scale": True,
                    "decode_from": "cls",
                }
            },
        )
        loss_config = {
            "loss_name": "contrastive",
            "field_name": "expressions",
            "name": "contrastive",
            "weight": 1.0,
        }
        task = loss_dict_to_task(loss_config, [field], [])
        assert isinstance(task.objective, ContrastiveObjective)
        assert not task.objective.contributes_sample_metrics

    def test_contrastive_source_type(self):
        """loss_dict_to_task with contrastive uses CellEmbeddingContrastiveSource."""
        from bmfm_targets.config.tokenization_config import FieldInfo
        from bmfm_targets.training.losses.compat import loss_dict_to_task
        from bmfm_targets.training.losses.sources import CellEmbeddingContrastiveSource

        field = FieldInfo(
            field_name="expressions",
            vocab_size=200,
            decode_modes={"contrastive": {}},
        )
        task = loss_dict_to_task(
            {"field_name": "expressions", "name": "contrastive", "weight": 1.0},
            [field],
            [],
        )
        assert isinstance(task.source, CellEmbeddingContrastiveSource)

    def test_contrastive_compute_via_task(self):
        """ContrastiveObjective.compute via a LossTask with mock outputs returns finite loss."""
        from bmfm_targets.config.tokenization_config import FieldInfo
        from bmfm_targets.training.losses.compat import loss_dict_to_task

        field = FieldInfo(
            field_name="expressions",
            vocab_size=200,
            decode_modes={"contrastive": {"projection_dim": 32}},
        )
        task = loss_dict_to_task(
            {"field_name": "expressions", "name": "contrastive", "weight": 1.0},
            [field],
            [],
        )

        N, D = 8, 32
        z = F.normalize(torch.randn(2 * N, D), dim=-1)
        scale = torch.tensor(float(np.log(1 / 0.07)))
        outputs = MagicMock()
        outputs.logits = {
            "expressions_contrastive": z,
            "expressions_contrastive_scale": scale,
        }
        batch = {"labels": {}}

        loss_val = task.calculate_loss(outputs, batch)
        assert torch.isfinite(loss_val), f"Loss not finite: {loss_val}"
        assert loss_val.item() >= 0.0


# ─── ContrastiveMetricsCallback ──────────────────────────────────────────────


class TestContrastiveMetricsCallback:
    def _module_with_contrastive(self, both_batch=False):
        """A stand-in pl_module whose loss_tasks hold a computed ContrastiveObjective."""
        from bmfm_targets.training.losses.objectives import ContrastiveObjective

        obj = ContrastiveObjective(
            temperature=0.07, gather_distributed=False, both_batch=both_batch
        )
        z = F.normalize(torch.randn(16, 32), dim=-1)
        obj.compute(z, None)  # populates _last_recall_at1 / _last_recall_at5

        task = MagicMock()
        task.objective = obj

        module = MagicMock()
        module.loss_tasks = [task]
        logged = {}
        module.log = lambda name, value, **kw: logged.__setitem__(name, value)
        return module, logged

    def test_logs_recall_metrics(self):
        """Callback logs the stashed recall diagnostics that the metric path skips."""
        from bmfm_targets.training.callbacks import ContrastiveMetricsCallback

        module, logged = self._module_with_contrastive()
        cb = ContrastiveMetricsCallback()
        cb.on_train_batch_end(None, module, None, None, 0)
        assert "train/recall@1" in logged
        assert "train/recall@5" in logged
        assert 0.0 <= logged["train/recall@1"] <= 1.0

    def test_validation_prefix(self):
        """Validation batches log under the val/ prefix."""
        from bmfm_targets.training.callbacks import ContrastiveMetricsCallback

        module, logged = self._module_with_contrastive()
        cb = ContrastiveMetricsCallback()
        cb.on_validation_batch_end(None, module, None, None, 0)
        assert "val/recall@1" in logged

    def test_ignores_non_contrastive_tasks(self):
        """Non-contrastive tasks are skipped; no metrics logged for them."""
        from bmfm_targets.training.callbacks import ContrastiveMetricsCallback

        non_contrastive = MagicMock()
        non_contrastive.objective.name = "mse"
        module = MagicMock()
        module.loss_tasks = [non_contrastive]
        logged = {}
        module.log = lambda name, value, **kw: logged.__setitem__(name, value)

        ContrastiveMetricsCallback().on_train_batch_end(None, module, None, None, 0)
        assert logged == {}


# ─── Model head -> source -> objective seam ──────────────────────────────────


class TestContrastiveHeadSeam:
    """
    Exercise the real model head (ContrastiveHead in SCBaseFieldDecoder), not a
    mocked logits dict: the head must write {field}_contrastive and
    {field}_contrastive_scale into the logits so the source can find them and
    the objective can compute a backward-able loss.
    """

    def _decoder(self, projection_dim=16, learnable_scale=True):
        from types import SimpleNamespace

        from bmfm_targets.config.tokenization_config import FieldInfo
        from bmfm_targets.models.predictive.layers import SCBaseFieldDecoder

        field = FieldInfo(
            field_name="expressions",
            vocab_size=200,
            decode_modes={
                "contrastive": {
                    "projection_dim": projection_dim,
                    "learnable_scale": learnable_scale,
                    "decode_from": "cls",
                }
            },
        )
        config = SimpleNamespace(hidden_size=32, fields=[field], label_columns=None)
        return SCBaseFieldDecoder(config)

    def test_head_writes_contrastive_keys(self):
        """Forward produces {field}_contrastive [2N, proj] and _scale in logits."""
        decoder = self._decoder(projection_dim=16)
        N = 8
        hidden_states = torch.randn(2 * N, 12, 32)  # [2N, seq_len, hidden]
        logits = decoder(hidden_states)
        assert "expressions_contrastive" in logits
        assert "expressions_contrastive_scale" in logits
        assert logits["expressions_contrastive"].shape == (2 * N, 16)

    def test_seam_end_to_end_backprops(self):
        """Head -> CellEmbeddingContrastiveSource -> ContrastiveObjective -> backward."""
        from bmfm_targets.training.losses.objectives import ContrastiveObjective
        from bmfm_targets.training.losses.sources import (
            CellEmbeddingContrastiveSource,
        )

        decoder = self._decoder(projection_dim=16, learnable_scale=True)
        N = 8
        hidden_states = torch.randn(2 * N, 12, 32)
        logits = decoder(hidden_states)

        outputs = MagicMock()
        outputs.logits = logits

        src = CellEmbeddingContrastiveSource("expressions", contrast_target="cls")
        obj = ContrastiveObjective(
            temperature=None, gather_distributed=False, both_batch=True
        )
        z, scale = src.extract(outputs, {})
        loss = obj.compute(z, scale)

        assert torch.isfinite(loss)
        loss.backward()
        # gradient must reach both the projection weights and the logit_scale
        head = decoder.field_decoders["expressions_contrastive"]
        assert head.projection.weight.grad is not None
        assert head.projection.weight.grad.abs().sum().item() > 0.0
        assert head.logit_scale.grad is not None
        assert head.logit_scale.grad.abs().item() > 0.0
