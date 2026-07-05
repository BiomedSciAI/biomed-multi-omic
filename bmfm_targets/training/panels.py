"""Per-batch gene-panel sampler for scConcept panel-split pretraining (spec §3.4).

A *panel* is a set of gene tokens. Applying a panel to a cell is exactly
``limit_fields_to_token_list`` (a.k.a. ``limit_input_tokens``). This module samples
one ``(panel1, panel2)`` pair per mini-batch from the batch's expressed-gene pool,
producing two disjoint token sets whose sizes are drawn (log-uniformly by default)
with an absolute floor, clamped to the pool size with graceful degradation when the
pool is too small for two floor-sized disjoint panels.
"""
from __future__ import annotations

import numpy as np

from bmfm_targets.training.sample_transforms import get_genes_expressed_in_batch


class PanelSampler:
    """
    Sample two disjoint gene panels (token sets) from a mini-batch's expressed pool.

    The batch is single-dataset (``ConditionHomogeneousBatchSampler``), so
    ``get_genes_expressed_in_batch`` is the batch's measured/expressed universe.
    Panel sizes are drawn log-uniformly (biasing toward smaller, sparser panels like
    real targeted assays) with an absolute floor ``panel_size_floor``, clamped to the
    available pool.

    Parameters
    ----------
    panel_size_floor : int
        Minimum panel size (paper's 400-gene threshold). When the pool is large
        enough (``P >= 2 * floor``) both panels are at least this size.
    panel_size_sampling : str
        ``"log_uniform"`` (default) or ``"uniform"`` size sampling.
    overlap_prob : float
        Probability of re-drawing panel2 independently (may overlap panel1).
        Default ``0.0`` (fully disjoint panels — spec v1).
    seed : int | None
        Base seed; per-worker RNG is reseeded from ``worker_info.seed`` so that
        pickled workers do not replay the same panel sequence.
    """

    def __init__(
        self,
        panel_size_floor: int = 400,
        panel_size_sampling: str = "log_uniform",
        overlap_prob: float = 0.0,
        seed: int | None = None,
    ):
        if panel_size_sampling not in ("log_uniform", "uniform"):
            raise ValueError(
                f"panel_size_sampling must be 'log_uniform' or 'uniform', "
                f"got {panel_size_sampling!r}"
            )
        self._floor = panel_size_floor
        self._panel_size_sampling = panel_size_sampling
        self._overlap_prob = overlap_prob
        self._seed = seed
        self._rng = None

    @property
    def rng(self) -> np.random.Generator:
        """
        Per-worker RNG, created lazily inside each dataloader worker.

        The sampler is pickled to every worker; if the RNG were built in
        ``__init__`` all workers (and the main process) would replay the *same*
        panel-split sequence. Reseeding from ``worker_info.seed`` (as the
        reference collator does) decorrelates the panels drawn by different
        workers.
        """
        if self._rng is None:
            seed = self._seed
            try:
                from torch.utils.data import get_worker_info

                info = get_worker_info()
                if info is not None:
                    seed = info.seed
            except Exception:
                pass
            self._rng = np.random.default_rng(seed)
        return self._rng

    def _sample_size(self, lo: int, hi: int) -> int:
        """Sample a size in ``[lo, hi]`` (log-uniform or uniform)."""
        if lo >= hi:
            return lo
        if self._panel_size_sampling == "log_uniform":
            size = int(np.exp2(self.rng.uniform(np.log2(lo), np.log2(hi))))
        else:
            size = int(self.rng.uniform(lo, hi))
        return int(np.clip(size, lo, hi))

    def sample_panels(self, examples) -> tuple[set, set]:
        """
        Draw ``(panel1, panel2)`` disjoint token sets from the batch's expressed pool.

        Clamp / degrade rule (floor = ``panel_size_floor``):
        - ``P >= 2 * floor``: both panels >= floor, disjoint; sizes drawn in
          ``[floor, P - floor]`` (panel1) and ``[floor, P - k]`` (panel2).
        - ``2 <= P < 2 * floor``: split the pool in half (panels < floor, still
          disjoint) — graceful degradation for small pools.
        - ``P < 2``: both panels = the whole pool (pathological empty/singleton batch).
        """
        pool = np.array(sorted(get_genes_expressed_in_batch(examples)), dtype=object)
        P = len(pool)
        floor = self._floor

        if P >= 2 * floor:
            k = self._sample_size(floor, P - floor)
            m = self._sample_size(floor, P - k)
            perm = self.rng.permutation(P)
            panel1 = pool[perm[:k]]
            panel2 = pool[perm[k : k + m]]
        elif P >= 2:
            k = P // 2
            perm = self.rng.permutation(P)
            panel1 = pool[perm[:k]]
            panel2 = pool[perm[k:]]
        else:
            panel1 = panel2 = pool

        if self._overlap_prob > 0 and P >= 2 and self.rng.random() < self._overlap_prob:
            m = self._sample_size(min(floor, P), P)
            panel2 = pool[self.rng.choice(P, size=m, replace=False)]

        return set(panel1.tolist()), set(panel2.tolist())
