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
