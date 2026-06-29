import logging
from types import SimpleNamespace

import pytest
import torch

from bmfm_targets.config import FieldInfo, LabelColumnInfo
from bmfm_targets.config.main_config import SCBertMainConfig


def _make_config(fields, losses=None):
    """
    Build a bare SCBertMainConfig (no __init__) with just the attrs the
    decode-head reconciliation / param-accounting helpers read.
    """
    cfg = SCBertMainConfig.__new__(SCBertMainConfig)
    cfg.fields = fields
    cfg.trainer = SimpleNamespace(losses=losses) if losses is not None else None
    return cfg


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


class TestMergeFields:
    """
    Tests for SCBertMainConfig._merge_fields attribute-aware merge.

    Encoder attributes (vocab_size, pretrained_embedding, vocab_update_strategy,
    tokenization_strategy, num_special_tokens, encoder_kwargs, datastore_config)
    must come from the checkpoint to stay compatible with pretrained weights.

    Decoder/task attributes (decode_modes, is_masked, is_input) must come from the
    YAML so the current downstream task configuration is respected.
    """

    @pytest.fixture()
    def ckpt_expr_field(self):
        """Checkpoint 'expressions' field with wced decode mode (pretrained decoder head)."""
        return FieldInfo(
            field_name="expressions",
            vocab_size=20000,
            is_masked=True,
            decode_modes={"wced": {"hidden_size": 256}},
            encoder_kwargs={"kind": "scale_adapt", "n_sin_basis": 11},
        )

    @pytest.fixture()
    def yaml_expr_field(self):
        """YAML 'expressions' field for a fine-tuning task that does NOT want decoding."""
        return FieldInfo(
            field_name="expressions",
            vocab_size=None,  # not yet resolved
            is_masked=False,
            decode_modes=None,
        )

    @pytest.fixture()
    def ckpt_genes_field(self):
        """Checkpoint 'genes' field."""
        return FieldInfo(
            field_name="genes",
            vocab_size=5000,
            vocab_update_strategy="static",
            tokenization_strategy="tokenize",
            num_special_tokens=3,
        )

    def test_conflicting_field_checkpoint_decode_modes_yaml_none(
        self, ckpt_expr_field, yaml_expr_field
    ):
        """
        Regression: checkpoint wced decode_modes must NOT bleed into fine-tuning field.

        When YAML specifies decode_modes=None for a field, the merged result must also
        have decode_modes=None even if the checkpoint carried a wced decoder head.
        """
        result = SCBertMainConfig._merge_fields(
            None,
            ckpt_fields=[ckpt_expr_field],
            yaml_fields=[yaml_expr_field],
            is_training=True,
        )
        assert len(result) == 1
        merged = result[0]
        # Decoder attrs from YAML
        assert (
            merged.decode_modes is None
        ), "Checkpoint's wced decode_modes must NOT override YAML decode_modes=None"
        assert merged.is_masked is False
        assert merged.is_input is True
        # Encoder attrs from checkpoint
        assert merged.vocab_size == 20000
        assert merged.encoder_kwargs == {"kind": "scale_adapt", "n_sin_basis": 11}

    def test_conflicting_field_yaml_decode_modes_win_over_checkpoint(
        self, ckpt_expr_field
    ):
        """YAML decode_modes takes priority over a different checkpoint decode_modes."""
        yaml_field = FieldInfo(
            field_name="expressions",
            is_masked=True,
            decode_modes={"regression": {}},
        )
        result = SCBertMainConfig._merge_fields(
            None,
            ckpt_fields=[ckpt_expr_field],
            yaml_fields=[yaml_field],
            is_training=True,
        )
        assert len(result) == 1
        merged = result[0]
        # Decoder attrs from YAML
        assert merged.decode_modes == {"regression": {}}
        assert merged.is_masked is True
        # Encoder attrs from checkpoint
        assert merged.vocab_size == 20000

    def test_brand_new_yaml_field_appended(self, ckpt_genes_field):
        """A field present only in YAML (not in checkpoint) is appended as-is."""
        new_yaml_field = FieldInfo(
            field_name="perturbations",
            is_masked=False,
            decode_modes=None,
        )
        result = SCBertMainConfig._merge_fields(
            None,
            ckpt_fields=[ckpt_genes_field],
            yaml_fields=[new_yaml_field],
            is_training=True,
        )
        assert len(result) == 2
        names = [f.field_name for f in result]
        assert names == ["genes", "perturbations"]
        # Brand-new field is unchanged
        assert result[1] == new_yaml_field

    def test_checkpoint_only_field_kept_when_yaml_missing_it(
        self, ckpt_genes_field, ckpt_expr_field
    ):
        """A checkpoint field absent from YAML is retained unchanged."""
        yaml_field = FieldInfo(
            field_name="genes",
            is_masked=False,
            decode_modes=None,
        )
        result = SCBertMainConfig._merge_fields(
            None,
            ckpt_fields=[ckpt_genes_field, ckpt_expr_field],
            yaml_fields=[yaml_field],
            is_training=True,
        )
        # Both checkpoint fields present; 'expressions' not in YAML so kept as-is
        assert len(result) == 2
        expr_result = next(f for f in result if f.field_name == "expressions")
        assert expr_result == ckpt_expr_field

    def test_predict_mode_returns_checkpoint_fields_unchanged(
        self, ckpt_expr_field, yaml_expr_field
    ):
        """In test/predict mode the checkpoint fields are returned as-is."""
        result = SCBertMainConfig._merge_fields(
            None,
            ckpt_fields=[ckpt_expr_field],
            yaml_fields=[yaml_expr_field],
            is_training=False,
        )
        assert result == [ckpt_expr_field]

    def test_no_ckpt_fields_returns_yaml(self, yaml_expr_field):
        """When checkpoint has no fields, YAML fields are returned."""
        result = SCBertMainConfig._merge_fields(
            None,
            ckpt_fields=[],
            yaml_fields=[yaml_expr_field],
            is_training=True,
        )
        assert result == [yaml_expr_field]

    def test_no_yaml_fields_during_training_returns_ckpt(self, ckpt_expr_field):
        """When YAML specifies no fields during training, checkpoint fields are used."""
        result = SCBertMainConfig._merge_fields(
            None,
            ckpt_fields=[ckpt_expr_field],
            yaml_fields=None,
            is_training=True,
        )
        assert result == [ckpt_expr_field]

    def test_checkpoint_field_order_preserved(self, ckpt_genes_field, ckpt_expr_field):
        """Checkpoint field order is maintained in the merged result."""
        yaml_genes = FieldInfo(field_name="genes", is_masked=False, decode_modes=None)
        yaml_expr = FieldInfo(
            field_name="expressions", is_masked=False, decode_modes=None
        )
        result = SCBertMainConfig._merge_fields(
            None,
            ckpt_fields=[ckpt_genes_field, ckpt_expr_field],
            yaml_fields=[yaml_genes, yaml_expr],
            is_training=True,
        )
        assert [f.field_name for f in result] == ["genes", "expressions"]


class TestLossFieldName:
    """Tests for SCBertMainConfig._loss_field_name across loss config forms."""

    def test_dict_field_loss(self):
        assert (
            SCBertMainConfig._loss_field_name({"field_name": "expressions"})
            == "expressions"
        )

    def test_label_loss_returns_none(self):
        """Label-based losses (no field_name) are not field-decode references."""
        assert (
            SCBertMainConfig._loss_field_name({"label_column_name": "cell_type"})
            is None
        )

    def test_losstask_object_with_source(self):
        loss = SimpleNamespace(source=SimpleNamespace(field_name="genes"))
        assert SCBertMainConfig._loss_field_name(loss) == "genes"


class TestReconcileDecodeHeads:
    """Tests for SCBertMainConfig._reconcile_decode_heads_with_losses (#3 / #5)."""

    def test_ghost_head_pruned(self, caplog):
        """A decode head no active loss references is pruned in training, with a warning."""
        expr = FieldInfo(
            field_name="expressions",
            vocab_size=20000,
            decode_modes={"wced": {"vocab_field": "genes"}},
        )
        label = FieldInfo(
            field_name="label_expressions",
            is_input=False,
            decode_modes={"wced": {"vocab_field": "genes"}},
        )
        cfg = _make_config([expr, label], losses=[{"field_name": "label_expressions"}])
        with caplog.at_level(logging.WARNING):
            cfg._reconcile_decode_heads_with_losses(is_training=True)

        expr_after = next(f for f in cfg.fields if f.field_name == "expressions")
        label_after = next(f for f in cfg.fields if f.field_name == "label_expressions")
        assert expr_after.decode_modes is None  # ghost head pruned
        assert label_after.decode_modes == {"wced": {"vocab_field": "genes"}}  # kept
        assert "expressions_wced" in caplog.text

    def test_referenced_field_kept(self):
        """A field referenced by an active loss keeps its decode_modes."""
        expr = FieldInfo(
            field_name="expressions",
            vocab_size=20000,
            decode_modes={"wced": {"vocab_field": "genes"}},
        )
        cfg = _make_config([expr], losses=[{"field_name": "expressions"}])
        cfg._reconcile_decode_heads_with_losses(is_training=True)
        assert cfg.fields[0].decode_modes == {"wced": {"vocab_field": "genes"}}

    def test_masked_field_not_pruned(self, caplog):
        """A masked field with no referencing loss is left in place (only warned)."""
        expr = FieldInfo(
            field_name="expressions",
            vocab_size=20000,
            is_masked=True,
            decode_modes={"token_scores": {}},
        )
        cfg = _make_config([expr], losses=[{"field_name": "other_field"}])
        with caplog.at_level(logging.WARNING):
            cfg._reconcile_decode_heads_with_losses(is_training=True)
        assert cfg.fields[0].decode_modes == {"token_scores": {}}  # unchanged
        assert "masked field 'expressions'" in caplog.text

    def test_predict_mode_no_pruning(self):
        """No pruning in test/predict mode -- heads are needed to produce outputs."""
        expr = FieldInfo(
            field_name="expressions",
            vocab_size=20000,
            decode_modes={"wced": {"vocab_field": "genes"}},
        )
        cfg = _make_config([expr], losses=[{"field_name": "label_expressions"}])
        cfg._reconcile_decode_heads_with_losses(is_training=False)
        assert cfg.fields[0].decode_modes == {"wced": {"vocab_field": "genes"}}

    def test_losstask_object_source_keeps_head(self):
        """Field-name extraction from a LossTask-style object keeps the referenced head."""
        expr = FieldInfo(
            field_name="expressions",
            vocab_size=20000,
            decode_modes={"wced": {"vocab_field": "genes"}},
        )
        loss = SimpleNamespace(source=SimpleNamespace(field_name="expressions"))
        cfg = _make_config([expr], losses=[loss])
        cfg._reconcile_decode_heads_with_losses(is_training=True)
        assert cfg.fields[0].decode_modes == {"wced": {"vocab_field": "genes"}}

    def test_no_losses_is_noop(self):
        """With no losses configured, nothing is pruned."""
        expr = FieldInfo(
            field_name="expressions",
            vocab_size=20000,
            decode_modes={"wced": {"vocab_field": "genes"}},
        )
        cfg = _make_config([expr], losses=None)
        cfg._reconcile_decode_heads_with_losses(is_training=True)
        assert cfg.fields[0].decode_modes == {"wced": {"vocab_field": "genes"}}


class TestCheckpointParamAccounting:
    """Tests for SCBertMainConfig._log_checkpoint_param_accounting (#5)."""

    def test_total_params_logged(self, caplog):
        cfg = _make_config([FieldInfo(field_name="genes", vocab_size=5000)])
        state_dict = {"a": torch.zeros(10), "b": torch.zeros(5)}
        with caplog.at_level(logging.INFO):
            cfg._log_checkpoint_param_accounting(
                {"state_dict": state_dict}, SimpleNamespace(hidden_size=256)
            )
        assert "total=15" in caplog.text

    def test_dropped_head_logged(self, caplog):
        """A decode head in the checkpoint but not built is reported as dropped."""
        cfg = _make_config([FieldInfo(field_name="genes", vocab_size=5000)])
        prefix = "cls.predictions.predictions.decoder.field_decoders.expressions_wced"
        state_dict = {
            "scbert.encoder.weight": torch.zeros(10, 10),
            f"{prefix}.weight": torch.zeros(20000, 256),
            f"{prefix}.bias": torch.zeros(20000),
        }
        with caplog.at_level(logging.WARNING):
            cfg._log_checkpoint_param_accounting(
                {"state_dict": state_dict}, SimpleNamespace(hidden_size=256)
            )
        assert "DROPPED" in caplog.text
        assert "expressions_wced" in caplog.text

    def test_new_head_estimated(self, caplog):
        """A built head absent from the checkpoint is reported with an estimated size."""
        genes = FieldInfo(field_name="genes", vocab_size=5000)
        expr = FieldInfo(
            field_name="expressions",
            vocab_size=55,
            decode_modes={"wced": {"vocab_field": "genes"}},
        )
        cfg = _make_config([genes, expr])
        state_dict = {"scbert.encoder.weight": torch.zeros(10, 10)}
        with caplog.at_level(logging.WARNING):
            cfg._log_checkpoint_param_accounting(
                {"state_dict": state_dict}, SimpleNamespace(hidden_size=256)
            )
        assert "NEW" in caplog.text
        assert "expressions_wced" in caplog.text
        # wced over genes vocab (5000) at hidden 256: 5000*256 + 5000 = 1_285_000
        assert "1285000" in caplog.text

    def test_never_raises_on_bad_input(self):
        cfg = _make_config([FieldInfo(field_name="genes", vocab_size=5000)])
        # Must be a silent no-op, never raise.
        cfg._log_checkpoint_param_accounting({}, None)
        cfg._log_checkpoint_param_accounting(None, None)
