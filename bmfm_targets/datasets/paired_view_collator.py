"""Module-level picklable collator that panel-splits each mini-batch (spec §3.5).

``PanelSplitCollator`` samples one ``(panel1, panel2)`` pair per batch from a
``PanelSampler`` over the batch's cells, then hands the **full cells** to the base
``MultiFieldCollator`` while threading the per-view panel as ``input_token_panels``.
The panel therefore limits only the **input** transform; the masker's label source
(``batch["mfi"]``, pre-transform) stays the whole cell so WCED ``non_input_genes``
labels cover the entire held-out transcriptome (spec §2.1).
"""
from __future__ import annotations

from bmfm_targets.tokenization.multifield_instance import MultiFieldInstance


class PanelSplitCollator:
    """
    Wrap a base collator to panel-split a single-dataset mini-batch.

    Once per ``__call__``: sample ``(panel1, panel2)`` from ``panel_sampler`` over the
    batch's cells, pass the FULL cells (never pre-subset) to ``base_collate`` and
    thread per-cell ``input_token_panels`` so only the input transform is
    panel-limited.

    n_views=2 (contrastive / both): emit block layout
    ``[A_0..A_{N-1}, B_0..B_{N-1}]`` (2N full cells) with panels
    ``[panel1]*N + [panel2]*N`` — preserves the ``ContrastiveObjective(both_batch=True)``
    ``za=z[:N] / zb=z[N:]`` split.

    n_views=1 (wced): emit N full cells with panels ``[panel1]*N``.
    """

    def __init__(self, base_collate, panel_sampler, n_views: int = 2):
        if n_views not in (1, 2):
            raise ValueError(f"n_views must be 1 or 2, got {n_views!r}")
        self._base_collate = base_collate
        self._panel_sampler = panel_sampler
        self._n_views = n_views

    def __call__(self, examples: list) -> dict:
        mfis = [e for e in examples if isinstance(e, MultiFieldInstance)]
        if len(mfis) != len(examples):
            # Some entries are not MultiFieldInstance (already collated?) —
            # fall back to the base collator, matching the old passthrough.
            return self._base_collate(examples)

        panel1, panel2 = self._panel_sampler.sample_panels(mfis)
        n = len(mfis)
        if self._n_views == 2:
            all_cells = mfis + mfis  # SAME full cells twice; panels differ
            panels = [panel1] * n + [panel2] * n
        else:
            all_cells = mfis
            panels = [panel1] * n
        return self._base_collate(all_cells, input_token_panels=panels)


# Backward-compatible alias for any external importers of the old name.
_PairedViewCollator = PanelSplitCollator
