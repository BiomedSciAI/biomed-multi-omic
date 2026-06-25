"""
Tests for _ChipCollator and scRNA2ChIPDataModule OT-batching improvements.

TDD: these tests were written before the implementation was updated, so they
initially fail and then pass once the production code is fixed.
"""
import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from bmfm_targets.datasets.scRNA2ChIP.scrna2chip_data_module import (
    _ChipCollator,
    scRNA2ChIPDataModule,
)
from bmfm_targets.tokenization import MultiFieldInstance
from bmfm_targets.training.data_module import DataModule

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 8
CELLTYPE_COLUMN = "tissue_label"


def _make_examples(celltype: str, n: int = 3) -> list[MultiFieldInstance]:
    """Return n fake MultiFieldInstances all belonging to celltype."""
    return [
        MultiFieldInstance(
            data={"genes": ["g1", "g2"], "expressions": [1.0, 2.0]},
            metadata={CELLTYPE_COLUMN: celltype, "cell_name": f"cell_{i}"},
        )
        for i in range(n)
    ]


def _make_base_collate_with_labels():
    """Fake base_collate that returns a batch with an existing labels dict."""

    def _collate(examples):
        return {
            "input_ids": torch.zeros(len(examples), 4),
            "labels": {"gene_expression": torch.zeros(len(examples), 4)},
        }

    return _collate


def _make_base_collate_without_labels():
    """Fake base_collate that returns a batch WITHOUT a labels key."""

    def _collate(examples):
        return {
            "input_ids": torch.zeros(len(examples), 4),
        }

    return _collate


@pytest.fixture()
def chip_populations() -> dict[str, torch.Tensor]:
    return {
        "T cell": torch.arange(1, VOCAB_SIZE + 1, dtype=torch.float32)
        .unsqueeze(0)
        .expand(3, -1)
        .clone(),  # shape [3, VOCAB_SIZE]
        "B cell": torch.ones(2, VOCAB_SIZE) * 2.0,  # shape [2, VOCAB_SIZE]
    }


# ---------------------------------------------------------------------------
# _ChipCollator tests
# ---------------------------------------------------------------------------


class TestChipCollatorLabelsInjection:
    """Verify chip_population appears under batch['labels']."""

    def test_population_placed_under_labels_when_labels_present(self, chip_populations):
        collator = _ChipCollator(
            base_collate=_make_base_collate_with_labels(),
            celltype_column=CELLTYPE_COLUMN,
            chip_populations=chip_populations,
            vocab_size=VOCAB_SIZE,
        )
        batch = collator(_make_examples("T cell"))

        assert "labels" in batch
        assert "chip_population" in batch["labels"]
        expected = chip_populations["T cell"]
        assert batch["labels"]["chip_population"].shape == (3, VOCAB_SIZE)
        assert torch.allclose(batch["labels"]["chip_population"], expected)

    def test_population_placed_under_labels_when_labels_absent(self, chip_populations):
        """If base_collate doesn't include 'labels', collator must create it."""
        collator = _ChipCollator(
            base_collate=_make_base_collate_without_labels(),
            celltype_column=CELLTYPE_COLUMN,
            chip_populations=chip_populations,
            vocab_size=VOCAB_SIZE,
        )
        batch = collator(_make_examples("B cell"))

        assert "labels" in batch
        assert "chip_population" in batch["labels"]
        assert batch["labels"]["chip_population"].shape == (2, VOCAB_SIZE)

    def test_top_level_chip_population_still_set(self, chip_populations):
        """Backward compat: batch['chip_population'] at top level must be retained."""
        collator = _ChipCollator(
            base_collate=_make_base_collate_with_labels(),
            celltype_column=CELLTYPE_COLUMN,
            chip_populations=chip_populations,
            vocab_size=VOCAB_SIZE,
        )
        batch = collator(_make_examples("T cell"))

        assert "chip_population" in batch

    def test_both_references_are_the_same_tensor(self, chip_populations):
        """batch['chip_population'] and batch['labels']['chip_population'] are identical."""
        collator = _ChipCollator(
            base_collate=_make_base_collate_with_labels(),
            celltype_column=CELLTYPE_COLUMN,
            chip_populations=chip_populations,
            vocab_size=VOCAB_SIZE,
        )
        batch = collator(_make_examples("T cell"))
        assert batch["chip_population"] is batch["labels"]["chip_population"]


class TestChipCollatorMissingCelltype:
    """Verify warning and zeros fallback when celltype not in chip_populations."""

    def test_missing_celltype_warns_with_celltype_in_message(
        self, chip_populations, caplog
    ):
        collator = _ChipCollator(
            base_collate=_make_base_collate_with_labels(),
            celltype_column=CELLTYPE_COLUMN,
            chip_populations=chip_populations,
            vocab_size=VOCAB_SIZE,
        )
        with caplog.at_level(
            logging.WARNING, logger="bmfm_targets.training.data_module"
        ):
            collator(_make_examples("NK cell"))

        warning_messages = [
            r.message for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert any(
            "NK cell" in msg for msg in warning_messages
        ), f"Expected 'NK cell' in warning; got: {warning_messages}"

    def test_missing_celltype_uses_zeros_fallback(self, chip_populations):
        collator = _ChipCollator(
            base_collate=_make_base_collate_with_labels(),
            celltype_column=CELLTYPE_COLUMN,
            chip_populations=chip_populations,
            vocab_size=VOCAB_SIZE,
        )
        batch = collator(_make_examples("NK cell"))

        assert "chip_population" in batch["labels"]
        assert batch["labels"]["chip_population"].shape == (1, VOCAB_SIZE)
        assert torch.all(batch["labels"]["chip_population"] == 0.0)

    def test_missing_celltype_zeros_also_at_top_level(self, chip_populations):
        collator = _ChipCollator(
            base_collate=_make_base_collate_with_labels(),
            celltype_column=CELLTYPE_COLUMN,
            chip_populations=chip_populations,
            vocab_size=VOCAB_SIZE,
        )
        batch = collator(_make_examples("NK cell"))

        assert batch["chip_population"].shape == (1, VOCAB_SIZE)
        assert torch.all(batch["chip_population"] == 0.0)


# ---------------------------------------------------------------------------
# scRNA2ChIPDataModule.setup — validate stage
# ---------------------------------------------------------------------------


class TestSetupValidateBuildsChipPopulations:
    """setup('validate') must call _build_chip_populations when use_ot_batching."""

    def _make_dm(self, use_ot_batching: bool = True) -> scRNA2ChIPDataModule:
        """Return a minimal scRNA2ChIPDataModule instance without calling __init__."""
        dm = object.__new__(scRNA2ChIPDataModule)
        dm.use_ot_batching = use_ot_batching
        dm.chip_populations = {}
        return dm

    def test_setup_validate_calls_build_when_use_ot_batching(self):
        dm = self._make_dm(use_ot_batching=True)
        build_called = []
        dm._build_chip_populations = lambda: build_called.append(True)

        with patch.object(DataModule, "setup", return_value=None):
            dm.setup(stage="validate")

        assert (
            build_called
        ), "_build_chip_populations was not called for stage='validate'"

    def test_setup_validate_skips_build_when_already_populated(self):
        dm = self._make_dm(use_ot_batching=True)
        dm.chip_populations = {"T cell": torch.zeros(1, VOCAB_SIZE)}  # already built
        build_called = []
        dm._build_chip_populations = lambda: build_called.append(True)

        with patch.object(DataModule, "setup", return_value=None):
            dm.setup(stage="validate")

        assert (
            not build_called
        ), "_build_chip_populations should not be called when populations are already built"

    def test_setup_fit_calls_build(self):
        dm = self._make_dm(use_ot_batching=True)
        build_called = []
        dm._build_chip_populations = lambda: build_called.append(True)

        with patch.object(DataModule, "setup", return_value=None):
            dm.setup(stage="fit")

        assert build_called

    def test_setup_no_stage_calls_build(self):
        dm = self._make_dm(use_ot_batching=True)
        build_called = []
        dm._build_chip_populations = lambda: build_called.append(True)

        with patch.object(DataModule, "setup", return_value=None):
            dm.setup(stage=None)

        assert build_called

    def test_setup_validate_no_op_when_not_ot_batching(self):
        dm = self._make_dm(use_ot_batching=False)
        build_called = []
        dm._build_chip_populations = lambda: build_called.append(True)

        with patch.object(DataModule, "setup", return_value=None):
            dm.setup(stage="validate")

        assert not build_called


# ---------------------------------------------------------------------------
# scRNA2ChIPDataModule.val_dataloader — OT batching path
# ---------------------------------------------------------------------------


class _FakeDataset(Dataset):
    """Minimal torch Dataset with metadata attribute for sampler."""

    def __init__(self, examples: list[MultiFieldInstance], celltype_column: str):
        self._examples = examples
        self.metadata = pd.DataFrame(
            {celltype_column: [e.metadata[celltype_column] for e in examples]}
        )

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> MultiFieldInstance:
        return self._examples[idx]


class TestValDataloaderOtBatching:
    """val_dataloader with use_ot_batching=True must use _ChipCollator."""

    def _make_dm(
        self,
        examples: list[MultiFieldInstance],
        chip_populations: dict[str, torch.Tensor],
        vocab_size: int = VOCAB_SIZE,
        use_ot_batching: bool = True,
        batch_size: int = 3,
    ) -> scRNA2ChIPDataModule:
        """Build a stub scRNA2ChIPDataModule without calling __init__."""
        dm = object.__new__(scRNA2ChIPDataModule)
        dm.use_ot_batching = use_ot_batching
        dm.celltype_column = CELLTYPE_COLUMN
        dm.batch_size = batch_size
        dm.num_workers = 0
        dm.chip_populations = chip_populations

        dm.dev_dataset = _FakeDataset(examples, CELLTYPE_COLUMN)

        # Stub val_collate_fn to return a trivial collate function
        def _simple_collate(batch):
            return {
                "input_ids": torch.zeros(len(batch), 2),
                "labels": {},
            }

        dm.val_collate_fn = MagicMock(return_value=_simple_collate)

        # Stub tokenizer
        mock_gene_tok = MagicMock()
        mock_gene_tok.vocab = {f"g{i}": i for i in range(vocab_size)}
        dm.tokenizer = MagicMock()
        dm.tokenizer.get_field_tokenizer.return_value = mock_gene_tok

        return dm

    def test_val_dataloader_ot_returns_dataloader(self, chip_populations):
        examples = _make_examples("T cell", n=6)
        dm = self._make_dm(examples, chip_populations)

        loader = dm.val_dataloader()

        assert isinstance(loader, DataLoader)

    def test_val_dataloader_ot_uses_chip_collator(self, chip_populations):
        examples = _make_examples("T cell", n=6)
        dm = self._make_dm(examples, chip_populations)

        loader = dm.val_dataloader()

        assert isinstance(loader.collate_fn, _ChipCollator)

    def test_val_dataloader_ot_batches_have_chip_population_in_labels(
        self, chip_populations
    ):
        examples = _make_examples("T cell", n=6)
        dm = self._make_dm(examples, chip_populations, batch_size=3)

        loader = dm.val_dataloader()

        for batch in loader:
            assert "labels" in batch
            assert "chip_population" in batch["labels"]
            assert batch["labels"]["chip_population"].shape[1] == VOCAB_SIZE
            break  # one batch is enough

    def test_val_dataloader_no_ot_delegates_to_super(self, chip_populations):
        """When use_ot_batching=False, val_dataloader must call super().val_dataloader()."""
        examples = _make_examples("T cell", n=6)
        dm = self._make_dm(examples, chip_populations, use_ot_batching=False)

        sentinel = object()
        with patch.object(DataModule, "val_dataloader", return_value=sentinel):
            result = dm.val_dataloader()

        assert result is sentinel
