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
    panel_size_ceil : int | None
        Maximum panel size. The paper draws panel sizes up to the full
        transcriptome, but at small batch sizes (few negatives) large panels make
        each view a near-complete cell profile, so the two views become trivially
        matchable and contrastive recall saturates. Capping the panel size keeps
        views PARTIAL and restores a non-trivial matching task. ``None`` = no cap
        (paper-faithful, up to pool size). See the panel-size diagnosis in
        docs/scconcept_paper_faithful_spec.md.
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
        panel_size_ceil: int | None = None,
        panel_size_sampling: str = "log_uniform",
        overlap_prob: float = 0.0,
        seed: int | None = None,
    ):
        if panel_size_sampling not in ("log_uniform", "uniform"):
            raise ValueError(
                f"panel_size_sampling must be 'log_uniform' or 'uniform', "
                f"got {panel_size_sampling!r}"
            )
        if panel_size_ceil is not None and panel_size_ceil < panel_size_floor:
            raise ValueError(
                f"panel_size_ceil ({panel_size_ceil}) must be >= "
                f"panel_size_floor ({panel_size_floor})"
            )
        self._floor = panel_size_floor
        self._ceil = panel_size_ceil
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
        # Upper bound per panel: leave room for the other panel's floor, and honor
        # the optional ceil that keeps views partial (else recall saturates; see
        # class docstring). ``P`` when no ceil is set (paper-faithful, full pool).
        cap = P if self._ceil is None else min(self._ceil, P)

        if P >= 2 * floor:
            k = self._sample_size(floor, min(cap, P - floor))
            m = self._sample_size(floor, min(cap, P - k))
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


class PerCellViewSplitter:
    """
    Split EACH cell's own nonzero genes into two disjoint random views.

    This restores the pre-refactor ``_PairedViewCollator._make_views`` semantics that
    produced a *non-trivial* contrastive task (val recall climbing from chance to
    ~0.70), in contrast to batch-level shared panels which the model solves trivially
    (a fixed gene basis across the batch is an easy shortcut; the objective becomes
    vacuous). Because each cell gets an INDEPENDENT random split there is no shared
    basis, so the encoder must learn genuine co-expression to align view A with view B.

    Unlike the old collator it returns gene SETS (per cell) to be threaded as
    ``input_token_panels`` rather than pre-subsetting the mfi, so ``batch["mfi"]``
    stays whole-cell and the WCED masker's whole-cell label / ``is_zero_bce``
    calibration is preserved (used by the ``both`` arm).

    Parameters
    ----------
    min_frac, max_frac : float
        View-A size is drawn in ``[min_frac, max_frac] * n_nonzero``; view B is the
        complement. Mirrors the reference collator (0.25 / 0.75).
    feature_dropout : float
        Fraction of each view's genes independently dropped (augmentation; keeps
        >= 4). ``0`` disables.
    panel_size_sampling : str
        ``"log_uniform"`` (default, biases toward smaller view A) or ``"uniform"``.
    seed : int | None
        Base seed; per-worker RNG reseeds from ``worker_info.seed`` (see ``rng``).
    """

    def __init__(
        self,
        min_frac: float = 0.25,
        max_frac: float = 0.75,
        feature_dropout: float = 0.1,
        panel_size_sampling: str = "log_uniform",
        seed: int | None = None,
    ):
        if panel_size_sampling not in ("log_uniform", "uniform"):
            raise ValueError(
                f"panel_size_sampling must be 'log_uniform' or 'uniform', "
                f"got {panel_size_sampling!r}"
            )
        self._min_frac = min_frac
        self._max_frac = max_frac
        self._feature_dropout = feature_dropout
        self._panel_size_sampling = panel_size_sampling
        self._seed = seed
        self._rng = None

    # Reuse the same lazy per-worker RNG contract as PanelSampler.
    rng = PanelSampler.rng

    def _sample_size(self, n: int) -> int:
        lo = max(1, int(self._min_frac * n))
        hi = max(lo, int(self._max_frac * n))
        if lo >= hi:
            return lo
        if self._panel_size_sampling == "log_uniform":
            size = int(np.exp2(self.rng.uniform(np.log2(lo), np.log2(hi))))
        else:
            size = int(self.rng.uniform(lo, hi))
        return int(np.clip(size, lo, hi))

    def _dropout(self, idx: np.ndarray) -> np.ndarray:
        if self._feature_dropout <= 0 or len(idx) <= 4:
            return idx
        keep = self.rng.random(len(idx)) >= self._feature_dropout
        if int(keep.sum()) < 4:
            return idx
        return idx[keep]

    def split(self, mfi) -> tuple[set, set]:
        """Return ``(viewA_genes, viewB_genes)`` — disjoint sets of the cell's nonzero genes."""
        genes = mfi["genes"]
        exprs = mfi["expressions"]
        nz = [g for g, e in zip(genes, exprs) if float(e) != 0.0]
        n = len(nz)
        if n < 8:
            # Too few nonzero genes to split into two informative views; use all in
            # both (rare, pathological cell). Contrastive loss skips tiny batches anyway.
            s = set(nz)
            return s, s
        nz = np.array(nz, dtype=object)
        perm = self.rng.permutation(n)
        size_a = self._sample_size(n)
        size_a = max(4, min(size_a, n - 4))  # keep >= 4 genes per view
        idx_a = self._dropout(perm[:size_a])
        idx_b = self._dropout(perm[size_a:])
        return set(nz[idx_a].tolist()), set(nz[idx_b].tolist())
