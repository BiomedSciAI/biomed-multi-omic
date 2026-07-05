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

    ``view_source`` may be either:
    - a BATCH-level sampler with ``sample_panels(examples) -> (panel1, panel2)`` — one
      shared panel pair applied to every cell (paper-faithful; but a shared gene basis
      makes the contrastive task trivially solvable at our data scale); or
    - a PER-CELL splitter with ``split(mfi) -> (viewA_genes, viewB_genes)`` — each cell
      independently split into two views, so there is NO shared basis and the encoder
      must learn real co-expression (val recall climbs from chance, the desired
      non-trivial objective). Detected by the presence of a ``split`` method.

    Either way the FULL cells are passed to the base collator (never pre-subset), so
    ``batch["mfi"]`` stays whole-cell for the WCED masker's whole-cell label.
    """

    def __init__(self, base_collate, view_source, n_views: int = 2):
        if n_views not in (1, 2):
            raise ValueError(f"n_views must be 1 or 2, got {n_views!r}")
        self._base_collate = base_collate
        self._view_source = view_source
        self._n_views = n_views

    def __call__(self, examples: list) -> dict:
        mfis = [e for e in examples if isinstance(e, MultiFieldInstance)]
        if len(mfis) != len(examples):
            # Some entries are not MultiFieldInstance (already collated?) —
            # fall back to the base collator, matching the old passthrough.
            return self._base_collate(examples)

        n = len(mfis)
        if hasattr(self._view_source, "split"):
            # Per-cell: independent (viewA, viewB) split for every cell (no shared basis).
            splits = [self._view_source.split(m) for m in mfis]
            a_panels = [s[0] for s in splits]
            b_panels = [s[1] for s in splits]
        else:
            # Batch-level: one shared (panel1, panel2) applied to all cells.
            panel1, panel2 = self._view_source.sample_panels(mfis)
            a_panels = [panel1] * n
            b_panels = [panel2] * n

        if self._n_views == 2:
            all_cells = mfis + mfis  # SAME full cells twice; per-view panels differ
            panels = a_panels + b_panels
        else:
            all_cells = mfis
            panels = a_panels
        return self._base_collate(all_cells, input_token_panels=panels)


# Backward-compatible alias for any external importers of the old name.
_PairedViewCollator = PanelSplitCollator
