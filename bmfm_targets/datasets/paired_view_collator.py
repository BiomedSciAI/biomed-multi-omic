"""Module-level picklable collator that generates two gene-panel views of each cell."""
from __future__ import annotations

import numpy as np

from bmfm_targets.tokenization.multifield_instance import MultiFieldInstance


class _PairedViewCollator:
    """
    Wraps a base collator to produce block-layout paired views for contrastive pretraining.

    Input: N MultiFieldInstance examples (one condition, from ConditionHomogeneousBatchSampler).
    Output: 2N examples in block layout [viewA_0..viewA_{N-1}, viewB_0..viewB_{N-1}].
    """

    def __init__(
        self,
        base_collate,
        min_frac: float = 0.25,
        max_frac: float = 0.75,
        overlap_prob: float = 0.0,
        feature_dropout: float = 0.1,
        seed: int | None = None,
    ):
        self._base_collate = base_collate
        self._min_frac = min_frac
        self._max_frac = max_frac
        self._overlap_prob = overlap_prob
        self._feature_dropout = feature_dropout
        self._rng = np.random.default_rng(seed)

    def _make_views(
        self, mfi: MultiFieldInstance
    ) -> tuple[MultiFieldInstance, MultiFieldInstance]:
        """Split one cell's genes into two disjoint (or overlapping) views."""
        # Get gene list from the first array field
        field_keys = list(mfi.data.keys())
        # Find the gene/token field (typically 'genes' or first field)
        gene_field = None
        for k in field_keys:
            val = mfi.data[k]
            if hasattr(val, "__len__") and len(val) > 0:
                gene_field = k
                break
        if gene_field is None:
            return mfi, mfi

        all_indices = np.arange(len(mfi.data[gene_field]))
        n = len(all_indices)

        # Shuffle and split
        perm = self._rng.permutation(n)
        size_a = int(n * self._rng.uniform(self._min_frac, self._max_frac))
        size_a = max(4, min(size_a, n - 4))  # keep at least 4 genes per view

        if self._rng.random() < self._overlap_prob:
            # overlapping: both views sample independently
            idx_a = self._rng.choice(n, size=size_a, replace=False)
            idx_b = self._rng.choice(n, size=n - size_a, replace=False)
        else:
            # disjoint: split the permutation
            idx_a = np.sort(perm[:size_a])
            idx_b = np.sort(perm[size_a:])

        def subset_mfi(idx):
            new_data = {}
            for k, v in mfi.data.items():
                if isinstance(v, list | tuple):
                    new_data[k] = [v[i] for i in idx]
                else:
                    try:
                        new_data[k] = v[idx]
                    except (IndexError, TypeError):
                        new_data[k] = v
            return MultiFieldInstance(data=new_data, metadata=mfi.metadata)

        return subset_mfi(idx_a), subset_mfi(idx_b)

    def __call__(self, examples: list) -> dict:
        views_a = []
        views_b = []
        for ex in examples:
            if isinstance(ex, MultiFieldInstance):
                va, vb = self._make_views(ex)
            else:
                va, vb = ex, ex
            views_a.append(va)
            views_b.append(vb)
        # Block layout: all A views then all B views
        all_views = views_a + views_b
        return self._base_collate(all_views)
