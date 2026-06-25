import pytest

from bmfm_targets.config import LabelColumnInfo
from bmfm_targets.config.main_config import SCBertMainConfig


@pytest.fixture()
def cell_type_col():
    return LabelColumnInfo(label_column_name="cell_type", n_unique_values=5)


class TestMergeLabelColumns:
    """Tests for SCBertMainConfig._merge_label_columns."""

    def test_empty_yaml_list_respected_during_training(self, cell_type_col):
        """
        label_columns: [] in YAML must suppress checkpoint label columns during training.

        Regression: before fix, [] was falsy so ckpt_cols were silently used,
        causing a KeyError when the dataset tried to load 'cell_type' from an
        h5ad that doesn't have that column.
        """
        result = SCBertMainConfig._merge_label_columns(
            None,  # self (static method pattern via direct call)
            ckpt_cols=[cell_type_col],
            yaml_cols=[],
            is_training=True,
        )
        assert (
            result == []
        ), "Explicit empty label_columns: [] should override checkpoint columns in training mode"

    def test_none_yaml_falls_back_to_ckpt_during_training(self, cell_type_col):
        """label_columns absent from YAML (None) should keep checkpoint columns."""
        result = SCBertMainConfig._merge_label_columns(
            None,
            ckpt_cols=[cell_type_col],
            yaml_cols=None,
            is_training=True,
        )
        assert result == [cell_type_col]

    def test_yaml_cols_override_ckpt_during_training(self, cell_type_col):
        new_col = LabelColumnInfo(label_column_name="tissue", n_unique_values=3)
        result = SCBertMainConfig._merge_label_columns(
            None,
            ckpt_cols=[cell_type_col],
            yaml_cols=[new_col],
            is_training=True,
        )
        assert result == [new_col]

    def test_empty_ckpt_returns_yaml(self, cell_type_col):
        result = SCBertMainConfig._merge_label_columns(
            None,
            ckpt_cols=[],
            yaml_cols=[cell_type_col],
            is_training=True,
        )
        assert result == [cell_type_col]

    def test_empty_yaml_list_respected_in_predict_mode(self, cell_type_col):
        """Explicit empty list in predict mode = embedding-only (cross-dataset)."""
        result = SCBertMainConfig._merge_label_columns(
            None,
            ckpt_cols=[cell_type_col],
            yaml_cols=[],
            is_training=False,
        )
        assert result == []

    def test_predict_mode_uses_ckpt_when_yaml_none(self, cell_type_col):
        result = SCBertMainConfig._merge_label_columns(
            None,
            ckpt_cols=[cell_type_col],
            yaml_cols=None,
            is_training=False,
        )
        assert result == [cell_type_col]
