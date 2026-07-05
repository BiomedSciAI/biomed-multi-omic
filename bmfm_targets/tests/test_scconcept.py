"""Unit tests for the scConcept panel-split pretraining pipeline (spec §7.1, §7.2, §7.5).

Covers:
- Panel split (§7.1): ``PanelSampler`` yields two disjoint panels from the batch's
  expressed pool, sizes respect the floor / clamp / degrade rules, and a toy batch run
  through ``PanelSplitCollator`` limits each view's input genes to
  ``panel ∩ nonzero(cell)`` with all cells in a block sharing one panel.
- WCED target (§7.2): for a known toy cell + panel, the masker's whole-cell label has
  the correct input vs non_input split with the right zeros/nonzeros.
- is_zero_bce sanity (§7.5): the non_input label set contains real measured zeros — the
  headline calibration check.

Tiny in-memory ``MultiFieldInstance`` objects + the repo test tokenizer keep these fast
(no 7.5GB h5ad load).
"""
import numpy as np
import pytest

from bmfm_targets.config import FieldInfo
from bmfm_targets.datasets.paired_view_collator import PanelSplitCollator
from bmfm_targets.tokenization import MultiFieldCollator, MultiFieldInstance
from bmfm_targets.training.masking.strategy import WCEDMasker
from bmfm_targets.training.panels import PanelSampler

from .helpers import load_test_tokenizer


def _cell(genes, expressions, name="c0"):
    return MultiFieldInstance(
        data={"genes": list(genes), "expressions": [float(e) for e in expressions]},
        metadata={"cell_name": name},
    )


def _nonzero_genes(mfi):
    return {
        g for g, e in zip(mfi["genes"], mfi["expressions"]) if float(e) != 0.0
    }


def _input_view_fields():
    return [
        FieldInfo("genes"),
        FieldInfo(
            "expressions",
            is_masked=True,
            tokenization_strategy="continuous_value_encoder",
            decode_modes={"regression": {}},
        ),
    ]


def _gene_tokens_in_row(tokenizer, gene_ids_row):
    """Decode a row of gene input_ids back to the set of (non-special) gene tokens."""
    vocab = tokenizer.get_field_vocab("genes")
    inv = {v: k for k, v in vocab.items()}
    specials = set(tokenizer.all_special_tokens)
    tokens = set()
    for i in gene_ids_row.tolist():
        tok = inv.get(int(i))
        if tok is not None and tok not in specials:
            tokens.add(tok)
    return tokens


# ---------------------------------------------------------------------------
# §7.1 PanelSampler: disjoint panels, floor / clamp / degrade
# ---------------------------------------------------------------------------
def test_panel_sampler_disjoint_and_respects_floor():
    """P >= 2*floor: two disjoint panels, each at least floor genes, drawn from pool."""
    pool = [f"g{i}" for i in range(20)]
    cells = [_cell(pool, [1.0] * len(pool))]
    floor = 3
    sampler = PanelSampler(
        panel_size_floor=floor, panel_size_sampling="uniform", seed=1
    )
    pool_set = set(pool)
    for _ in range(25):
        p1, p2 = sampler.sample_panels(cells)
        assert p1.isdisjoint(p2)
        assert len(p1) >= floor
        assert len(p2) >= floor
        assert p1 <= pool_set and p2 <= pool_set


def test_panel_sampler_log_uniform_disjoint():
    """Default log_uniform sizing also yields disjoint, floor-respecting panels."""
    pool = [f"g{i}" for i in range(64)]
    cells = [_cell(pool, [1.0] * len(pool))]
    sampler = PanelSampler(panel_size_floor=4, panel_size_sampling="log_uniform", seed=7)
    for _ in range(25):
        p1, p2 = sampler.sample_panels(cells)
        assert p1.isdisjoint(p2)
        assert len(p1) >= 4 and len(p2) >= 4


def test_panel_sampler_excludes_zero_expression_genes():
    """Pool = get_genes_expressed_in_batch, so zero-expression genes never appear."""
    genes = [f"g{i}" for i in range(10)]
    # even-indexed genes are expressed, odd-indexed are zero
    exprs = [1.0 if i % 2 == 0 else 0.0 for i in range(10)]
    cells = [_cell(genes, exprs)]
    zero_genes = {g for g, e in zip(genes, exprs) if e == 0.0}
    sampler = PanelSampler(panel_size_floor=1, panel_size_sampling="uniform", seed=0)
    for _ in range(10):
        p1, p2 = sampler.sample_panels(cells)
        assert (p1 | p2).isdisjoint(zero_genes)


def test_panel_sampler_small_pool_degrades_to_half_split():
    """2 <= P < 2*floor: split the pool in half, still disjoint, sub-floor allowed."""
    pool = [f"g{i}" for i in range(6)]
    cells = [_cell(pool, [1.0] * len(pool))]
    sampler = PanelSampler(panel_size_floor=400, seed=0)  # P=6 << 800
    p1, p2 = sampler.sample_panels(cells)
    assert p1.isdisjoint(p2)
    assert len(p1) == 3 and len(p2) == 3
    assert (p1 | p2) == set(pool)


def test_panel_sampler_singleton_pool_returns_whole_pool():
    """P < 2 (pathological): both panels are the whole pool."""
    cells = [_cell(["only_gene"], [1.0])]
    sampler = PanelSampler(panel_size_floor=400, seed=0)
    p1, p2 = sampler.sample_panels(cells)
    assert p1 == {"only_gene"}
    assert p2 == {"only_gene"}


# ---------------------------------------------------------------------------
# §7.1 PanelSplitCollator: each view limited to panel ∩ nonzero(cell)
# ---------------------------------------------------------------------------
class _FakeSampler:
    """Deterministic sampler that always returns the same fixed (panel1, panel2)."""

    def __init__(self, panel1, panel2):
        self.panel1 = set(panel1)
        self.panel2 = set(panel2)

    def sample_panels(self, examples):
        return self.panel1, self.panel2


def _make_input_collator(tokenizer):
    all_genes = [f"token{i}" for i in range(1, 9)]
    return MultiFieldCollator(
        tokenizer,
        fields=_input_view_fields(),
        sequence_order="sorted",
        pad_zero_expression_strategy={
            "strategy": "batch_wise",
            "expressed_sample_ratio": 1.0,
        },
        max_length=16,
        pad_to_multiple_of=8,
        limit_input_tokens=all_genes,
    )


def test_panel_split_collator_two_views_limited_to_panel_and_nonzero():
    tokenizer = load_test_tokenizer()
    # 4 cells; each expresses >=2 genes in EACH panel (so the pad-zero min-length-2
    # fallback never fires) and each carries some panel-member zeros that the input
    # zero-drop must remove.
    cells = [
        _cell(
            ["token1", "token2", "token3", "token5", "token6", "token7"],
            [3.0, 4.0, 0.0, 2.0, 5.0, 0.0],
            "a",
        ),
        _cell(
            ["token1", "token3", "token4", "token5", "token7", "token8"],
            [0.0, 1.0, 6.0, 0.0, 2.0, 3.0],
            "b",
        ),
        _cell(
            ["token1", "token2", "token4", "token6", "token8"],
            [0.0, 2.0, 3.0, 3.0, 4.0],
            "c",
        ),
        _cell(
            ["token1", "token3", "token2", "token5", "token7", "token6"],
            [5.0, 4.0, 0.0, 3.0, 2.0, 0.0],
            "d",
        ),
    ]
    panel1 = {"token1", "token2", "token3", "token4"}
    panel2 = {"token5", "token6", "token7", "token8"}
    n = len(cells)

    collator = PanelSplitCollator(
        _make_input_collator(tokenizer), _FakeSampler(panel1, panel2), n_views=2
    )
    batch = collator(cells)

    input_ids = batch["input_ids"]
    assert input_ids.shape[0] == 2 * n  # block layout [A_0..A_{n-1}, B_0..B_{n-1}]

    block_a_union, block_b_union = set(), set()
    for row in range(2 * n):
        cell = cells[row % n]
        panel = panel1 if row < n else panel2
        genes = _gene_tokens_in_row(tokenizer, input_ids[row, 0, :])
        allowed = panel & _nonzero_genes(cell)
        # each view's input genes are a subset of panel ∩ nonzero(cell)
        assert genes <= allowed, (row, genes, allowed)
        assert genes  # non-empty for these cells
        (block_a_union if row < n else block_b_union).update(genes)

    # all cells in block A drew panel1, all cells in block B drew panel2
    assert block_a_union <= panel1
    assert block_b_union <= panel2
    assert block_a_union.isdisjoint(block_b_union)


def test_panel_split_collator_single_view():
    tokenizer = load_test_tokenizer()
    cells = [
        _cell(["token1", "token2", "token3"], [3.0, 0.0, 4.0], "a"),
        _cell(["token1", "token4", "token2"], [1.0, 6.0, 0.0], "b"),
    ]
    panel1 = {"token1", "token2", "token3", "token4"}
    panel2 = {"token5", "token6"}
    n = len(cells)

    collator = PanelSplitCollator(
        _make_input_collator(tokenizer), _FakeSampler(panel1, panel2), n_views=1
    )
    batch = collator(cells)
    input_ids = batch["input_ids"]
    assert input_ids.shape[0] == n  # single view: N rows only

    for row in range(n):
        genes = _gene_tokens_in_row(tokenizer, input_ids[row, 0, :])
        allowed = panel1 & _nonzero_genes(cells[row])
        assert genes <= allowed
        assert genes


# ---------------------------------------------------------------------------
# §7.2 / §7.5 WCED target: whole-cell label, input/non_input split, real zeros
# ---------------------------------------------------------------------------
def test_wced_panel_split_target_zeros_and_split():
    tokenizer = load_test_tokenizer()
    # Whole cell (expose_zeros: all) with a genuine zero/nonzero mix.
    genes = ["token1", "token2", "token3", "token4", "token5", "token6"]
    exprs = [3.0, 0.0, 5.0, 0.0, 2.0, 0.0]  # nonzero: token1, token3, token5
    cell = _cell(genes, exprs)

    # Input panel; panel ∩ nonzero(cell) = {token1, token3} -> the input view.
    panel = {"token1", "token2", "token3"}

    masker = WCEDMasker(
        tokenizer=tokenizer,
        label_sets=["non_input", "input", "all"],
        sample_transforms_kwargs={
            "log_normalize_transform": False,
            "median_normalization": False,
        },
    )
    collator = MultiFieldCollator(
        tokenizer,
        masker=masker,
        fields=_input_view_fields(),
        sequence_order="sorted",
        pad_zero_expression_strategy={
            "strategy": "batch_wise",
            "expressed_sample_ratio": 1.0,
        },
        max_length=16,
        pad_to_multiple_of=8,
        limit_input_tokens=genes,
    )

    batch = collator([cell], input_token_panels=[panel])

    vocab = tokenizer.get_field_vocab("genes")
    labels = batch["labels"]["expressions"]
    inp = labels["input"][0]
    non = labels["non_input"][0]
    all_lbl = labels["all"][0]

    # --- input label set = seen panel nonzeros (token1, token3) ---
    assert inp[vocab["token1"]].item() == 3.0
    assert inp[vocab["token3"]].item() == 5.0
    for tok in ["token2", "token4", "token5", "token6"]:
        assert inp[vocab[tok]].item() == -100

    # --- non_input label set = everything else in the whole cell ---
    # panel zero (token2) + entire held-out transcriptome (token4/5/6)
    assert non[vocab["token2"]].item() == 0.0  # panel gene, zero in cell -> held out
    assert non[vocab["token4"]].item() == 0.0
    assert non[vocab["token6"]].item() == 0.0
    assert non[vocab["token5"]].item() == 2.0
    for tok in ["token1", "token3"]:
        assert non[vocab[tok]].item() == -100

    # --- §7.5 headline: non_input is a genuine zero/nonzero MIX (not all-nonzero) ---
    measured = non[non != -100]
    assert (measured == 0.0).any(), "non_input must carry real measured zeros"
    assert (measured > 0.0).any(), "non_input must carry real nonzeros"

    # --- all label set = whole cell, zeros kept ---
    assert all_lbl[vocab["token1"]].item() == 3.0
    assert all_lbl[vocab["token5"]].item() == 2.0
    assert all_lbl[vocab["token2"]].item() == 0.0
    assert all_lbl[vocab["token6"]].item() == 0.0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-x", "-q"]))
