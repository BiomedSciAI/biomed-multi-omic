"""
Tests for WP-B: configurable pseudobulk pairing in BasescRNA2ChIPDataset.

Runs in the *fast* CI job (no heavy data, pure in-memory synthetic AnnData).
"""

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as ss

from bmfm_targets.datasets.scRNA2ChIP.base_scrna2chip_dataset import (
    BasescRNA2ChIPDataset,
)

# ---------------------------------------------------------------------------
# Synthetic AnnData fixture
# ---------------------------------------------------------------------------


def _make_synthetic_scrna2chip_adata() -> sc.AnnData:
    """
    Build a small synthetic AnnData for scRNA2ChIP pairing tests.

    8 cells x 4 genes.
    - 4 scRNA cells: 2 T cell, 2 B cell.
    - 4 ChIP cells: 2 T cell (distinct values), 2 B cell (distinct values).

    T cell ChIP values: row0=[1,0,0,0], row1=[3,0,0,0]  -> mean=[2,0,0,0]
    B cell ChIP values: row0=[0,1,0,0], row1=[0,3,0,0]  -> mean=[0,2,0,0]

    The mean for each tissue is strictly distinct from any single draw, so
    tests can reliably distinguish pseudobulk from random pairing.
    """
    genes = ["gA", "gB", "gC", "gD"]

    scrna_X = np.array(
        [
            [1.0, 2.0, 0.0, 0.0],  # T cell scRNA 0
            [0.0, 1.0, 2.0, 0.0],  # T cell scRNA 1
            [0.0, 0.0, 1.0, 2.0],  # B cell scRNA 2
            [2.0, 0.0, 0.0, 1.0],  # B cell scRNA 3
        ]
    )
    chip_X = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # T cell ChIP 0
            [3.0, 0.0, 0.0, 0.0],  # T cell ChIP 1   -> mean T=[2,0,0,0]
            [0.0, 1.0, 0.0, 0.0],  # B cell ChIP 0
            [0.0, 3.0, 0.0, 0.0],  # B cell ChIP 1   -> mean B=[0,2,0,0]
        ]
    )
    X = np.vstack([scrna_X, chip_X])

    obs = pd.DataFrame(
        {
            "data_type": [
                "scRNA",
                "scRNA",
                "scRNA",
                "scRNA",
                "ChIP",
                "ChIP",
                "ChIP",
                "ChIP",
            ],
            "tissue_label": [
                "T cell",
                "T cell",
                "B cell",
                "B cell",
                "T cell",
                "T cell",
                "B cell",
                "B cell",
            ],
        },
        index=[f"cell_{i}" for i in range(8)],
    )
    var = pd.DataFrame(index=genes)
    return sc.AnnData(X=ss.csr_matrix(X), obs=obs, var=var)


def _make_dataset(**extra_kwargs) -> BasescRNA2ChIPDataset:
    """Construct BasescRNA2ChIPDataset from synthetic AnnData."""
    adata = _make_synthetic_scrna2chip_adata()
    return BasescRNA2ChIPDataset(
        processed_data_source=adata,
        new_field="label_expressions",
        expose_zeros="all",
        **extra_kwargs,
    )


# ---------------------------------------------------------------------------
# RED tests — these must fail before the implementation is written
# ---------------------------------------------------------------------------


def test_default_strategy_is_random():
    """Default dataset has chip_pairing_strategy == 'random'."""
    ds = _make_dataset()
    assert ds.chip_pairing_strategy == "random"


def test_pseudobulk_pairing_is_deterministic():
    """With 'pseudobulk', calling _get_item_by_index(i) twice yields identical label_expressions."""
    ds = _make_dataset(chip_pairing_strategy="pseudobulk")
    # Call twice on the same T cell index
    result_a = ds._get_item_by_index(0)
    result_b = ds._get_item_by_index(0)
    assert (
        result_a["label_expressions"] == result_b["label_expressions"]
    ), "Pseudobulk pairing must be deterministic — got different values on two calls"


def test_pseudobulk_equals_group_mean():
    """Pseudobulk label_expressions equal the independently computed mean of matching ChIP cells."""
    ds = _make_dataset(chip_pairing_strategy="pseudobulk")

    # T cell is scRNA cells 0 and 1
    result = ds._get_item_by_index(0)
    label_exprs = result["label_expressions"]

    # Compute expected mean independently from the raw ChIP data
    chip_tcell = ds.chipseq_cells[
        ds.chipseq_cells.obs.query("tissue_label == 'T cell'").index
    ]
    expected_mean = chip_tcell.X.toarray().mean(axis=0).tolist()

    # The merged MFI has genes in sorted order; map label_exprs to gene->value
    genes = result["genes"]
    label_map = dict(zip(genes, label_exprs))
    expected_map = dict(zip(ds.chipseq_cells.var_names, expected_mean))

    for gene in ds.chipseq_cells.var_names:
        assert (
            abs(label_map.get(gene, 0.0) - expected_map[gene]) < 1e-6
        ), f"Gene {gene}: got {label_map.get(gene)}, expected {expected_map[gene]}"


def test_random_pairing_unchanged():
    """With 'random' (default), paired label_expressions must be one of the candidate ChIP rows."""
    ds = _make_dataset()  # default chip_pairing_strategy="random"

    # For T cell scRNA cell at index 0, the two valid ChIP rows are:
    chip_tcell = ds.chipseq_cells[
        ds.chipseq_cells.obs.query("tissue_label == 'T cell'").index
    ]
    valid_rows = [chip_tcell[i].X.toarray().tolist()[0] for i in range(len(chip_tcell))]

    # Run several times to sample the random draw
    seen = set()
    for _ in range(20):
        result = ds._get_item_by_index(0)
        genes = result["genes"]
        label_exprs = result["label_expressions"]
        label_map = dict(zip(genes, label_exprs))
        row_vals = tuple(label_map.get(g, 0.0) for g in ds.chipseq_cells.var_names)
        seen.add(row_vals)

    valid_tuples = {tuple(r) for r in valid_rows}
    assert seen.issubset(
        valid_tuples
    ), f"Random pairing returned values not in candidate ChIP rows: {seen - valid_tuples}"


# ---------------------------------------------------------------------------
# Datamodule: chip_pairing_strategy parameter injection
# ---------------------------------------------------------------------------


def test_datamodule_stores_chip_pairing_strategy():
    """scRNA2ChIPDataModule __init__ signature has chip_pairing_strategy param."""
    import inspect

    from bmfm_targets.datasets.scRNA2ChIP.scrna2chip_data_module import (
        scRNA2ChIPDataModule,
    )

    sig = inspect.signature(scRNA2ChIPDataModule.__init__)
    assert (
        "chip_pairing_strategy" in sig.parameters
    ), "chip_pairing_strategy must be a parameter of scRNA2ChIPDataModule.__init__"
    assert (
        sig.parameters["chip_pairing_strategy"].default == "random"
    ), "chip_pairing_strategy default must be 'random'"


def _make_held_out_split_adata() -> sc.AnnData:
    """
    Synthetic AnnData with a held-out tissue (Liver) only in the test split.

    Train: T cell, B cell (scRNA + 2 ChIP each).  Test: Liver (scRNA + 2 ChIP).
    Per-tissue ChIP means: T=[2,0,0,0], B=[0,2,0,0], Liver=[0,0,6,0].
    """
    genes = ["gA", "gB", "gC", "gD"]
    rows, obs_rows = [], []

    def add(x, dt, tis, spl):
        rows.append(x)
        obs_rows.append((dt, tis, spl))

    add([1.0, 2.0, 0.0, 0.0], "scRNA", "T cell", "train")
    add([0.0, 1.0, 2.0, 0.0], "scRNA", "B cell", "train")
    add([0.0, 0.0, 1.0, 2.0], "scRNA", "Liver", "test")
    add([1.0, 0.0, 0.0, 0.0], "ChIP", "T cell", "train")
    add([3.0, 0.0, 0.0, 0.0], "ChIP", "T cell", "train")  # T mean [2,0,0,0]
    add([0.0, 1.0, 0.0, 0.0], "ChIP", "B cell", "train")
    add([0.0, 3.0, 0.0, 0.0], "ChIP", "B cell", "train")  # B mean [0,2,0,0]
    add([0.0, 0.0, 5.0, 0.0], "ChIP", "Liver", "test")
    add([0.0, 0.0, 7.0, 0.0], "ChIP", "Liver", "test")  # Liver mean [0,0,6,0]

    obs = pd.DataFrame(
        obs_rows,
        columns=["data_type", "tissue_label", "tissue_split"],
        index=[f"c{i}" for i in range(len(rows))],
    )
    return sc.AnnData(
        X=ss.csr_matrix(np.array(rows)), obs=obs, var=pd.DataFrame(index=genes)
    )


def test_group_means_span_all_splits_with_avg_baseline():
    """A train-split dataset's group_means cover held-out tissues + an avg-baseline row."""
    adata = _make_held_out_split_adata()
    ds = BasescRNA2ChIPDataset(
        processed_data_source=adata,
        new_field="label_expressions",
        expose_zeros="all",
        split="train",
        split_column_name="tissue_split",
        chip_pairing_strategy="pseudobulk",
    )

    gm_index = set(ds.group_means.obs_names)
    # metrics ground truth spans the held-out (test-split) tissue ...
    assert {"T cell", "B cell", "Liver"}.issubset(gm_index)
    # ... and carries the average-training-tissue baseline row
    assert "Average_Perturbation_Train" in gm_index

    # baseline = unweighted mean over TRAIN tissue means (Liver excluded)
    avg = ds.group_means["Average_Perturbation_Train"].X
    avg = avg.toarray()[0] if ss.issparse(avg) else np.asarray(avg)[0]
    np.testing.assert_allclose(avg, [1.0, 1.0, 0.0, 0.0], atol=1e-6)


def test_pseudobulk_pairing_target_is_split_restricted():
    """The pairing target uses only this split's ChIP — no held-out leakage, no avg row."""
    adata = _make_held_out_split_adata()
    ds = BasescRNA2ChIPDataset(
        processed_data_source=adata,
        new_field="label_expressions",
        expose_zeros="all",
        split="train",
        split_column_name="tissue_split",
        chip_pairing_strategy="pseudobulk",
    )
    pair_index = set(ds._chip_group_means.obs_names)
    assert pair_index == {
        "T cell",
        "B cell",
    }, f"Pairing target must be train-split only; got {pair_index}"


def test_datamodule_prepare_dataset_kwargs_injects_strategy():
    """_prepare_dataset_kwargs injects chip_pairing_strategy into the result dict."""
    from unittest.mock import patch

    from bmfm_targets.datasets.scRNA2ChIP.scrna2chip_data_module import (
        scRNA2ChIPDataModule,
    )

    dm = object.__new__(scRNA2ChIPDataModule)
    dm.perturbation_column_name = "perturbation"
    dm.use_ot_batching = False
    dm.celltype_column = "tissue_label"
    dm.chip_populations = {}
    dm.dataset_kwargs = {}
    dm.chip_pairing_strategy = "pseudobulk"

    with patch(
        "bmfm_targets.training.data_module.DataModule._prepare_dataset_kwargs",
        return_value={},
    ):
        result = scRNA2ChIPDataModule._prepare_dataset_kwargs(dm)

    assert "chip_pairing_strategy" in result
    assert result["chip_pairing_strategy"] == "pseudobulk"
