"""Tests for checkpoint migration from old model formats to multitask."""

from pathlib import Path
from typing import Any

import pytest
import torch

from bmfm_targets.config import FieldInfo, LabelColumnInfo, SCBertConfig
from bmfm_targets.models.predictive import scbert
from bmfm_targets.training.modules import MultiTaskTrainingModule
from run.migrate_checkpoints_to_multitask import (
    convert_mlm_to_multitask,
    convert_seqcls_to_multitask,
    convert_seqlabel_to_multitask,
    detect_checkpoint_type,
)


@pytest.fixture()
def minimal_mlm_config():
    """Create minimal MLM config for testing."""
    return SCBertConfig(
        fields=[
            FieldInfo(field_name="genes", vocab_size=100),
            FieldInfo(
                field_name="expressions",
                vocab_size=100,
                is_masked=True,
                decode_modes={"token_scores": {}},
            ),
        ],
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
    )


@pytest.fixture()
def minimal_seqcls_config():
    """Create minimal SeqCls config for testing."""
    return SCBertConfig(
        fields=[FieldInfo(field_name="genes", vocab_size=100)],
        label_columns=[
            LabelColumnInfo(label_column_name="cell_type", n_unique_values=10)
        ],
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
    )


@pytest.fixture()
def minimal_multitask_config():
    """Create minimal multitask config for testing."""
    return SCBertConfig(
        fields=[
            FieldInfo(field_name="genes", vocab_size=100),
            FieldInfo(
                field_name="expressions",
                vocab_size=100,
                is_masked=True,
                decode_modes={"token_scores": {}},
            ),
        ],
        label_columns=[
            LabelColumnInfo(label_column_name="cell_type", n_unique_values=10)
        ],
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
    )


def test_detect_mlm_checkpoint(minimal_mlm_config):
    """Test detection of MLM checkpoint type."""
    model = scbert.SCBertForMaskedLM(minimal_mlm_config)
    ckpt_type, label = detect_checkpoint_type(model.state_dict())
    assert ckpt_type == "mlm_or_seqlabel"
    assert label is None


def test_detect_seqcls_checkpoint(minimal_seqcls_config):
    """Test detection of SeqCls checkpoint type."""
    model = scbert.SCBertForSequenceClassification(minimal_seqcls_config)
    ckpt_type, label = detect_checkpoint_type(model.state_dict())
    assert ckpt_type == "sequence_classification"
    assert label is None


def test_detect_seqlabel_checkpoint(minimal_mlm_config):
    """Test detection of SeqLabel checkpoint type."""
    model = scbert.SCBertForSequenceLabeling(minimal_mlm_config)
    ckpt_type, label = detect_checkpoint_type(model.state_dict())
    assert ckpt_type == "mlm_or_seqlabel"
    assert label is None


def test_detect_multitask_checkpoint(minimal_multitask_config):
    """Test detection of already-multitask checkpoint."""
    model = scbert.SCBertForMultiTaskModeling(minimal_multitask_config)
    ckpt_type, label = detect_checkpoint_type(model.state_dict())
    assert ckpt_type == "multitask"
    assert label is None


def test_convert_mlm_to_multitask(minimal_mlm_config):
    """Test MLM → Multitask conversion preserves weights."""
    mlm_state = scbert.SCBertForMaskedLM(minimal_mlm_config).state_dict()
    multitask_state = convert_mlm_to_multitask(mlm_state)

    # Verify key structure changes
    assert any("cls.predictions.predictions" in k for k in multitask_state)
    assert not any(
        "cls.predictions" in k and "cls.predictions.predictions" not in k
        for k in multitask_state
        if "cls" in k
    )

    # Verify weights preserved
    for old_key, val in mlm_state.items():
        if "cls.predictions" in old_key:
            new_key = old_key.replace("cls.predictions", "cls.predictions.predictions")
            assert torch.allclose(val, multitask_state[new_key])
        else:
            assert torch.allclose(val, multitask_state[old_key])


def test_convert_mlm_loadable_into_multitask(minimal_multitask_config):
    """Test converted MLM checkpoint can be loaded into multitask model."""
    mlm_state = scbert.SCBertForMaskedLM(minimal_multitask_config).state_dict()
    multitask_state = convert_mlm_to_multitask(mlm_state)

    model = scbert.SCBertForMultiTaskModeling(minimal_multitask_config)
    missing, unexpected = model.load_state_dict(multitask_state, strict=False)

    assert len(unexpected) == 0
    # After migration, label_predictions are missing (not in MLM checkpoint)
    assert any("label_predictions" in k for k in missing)
    # Pooler is NOT in missing - it's randomly initialized in multitask model
    # Users must use pooling_method="first_token" for migrated MLM checkpoints
    assert not any("pooler" in k for k in missing)


def test_convert_seqcls_to_multitask_with_label_name(minimal_seqcls_config):
    """Test SeqCls → Multitask conversion with explicit label name."""
    seqcls_state = scbert.SCBertForSequenceClassification(
        minimal_seqcls_config
    ).state_dict()
    multitask_state = convert_seqcls_to_multitask(seqcls_state, label_name="cell_type")

    assert any(
        "cls.label_predictions.predictions.label_decoders.cell_type.decoder" in k
        for k in multitask_state
    )
    assert not any("classifier" in k for k in multitask_state)


def test_convert_seqcls_requires_label_name(minimal_seqcls_config):
    """Test SeqCls conversion fails without label name."""
    seqcls_state = scbert.SCBertForSequenceClassification(
        minimal_seqcls_config
    ).state_dict()
    with pytest.raises(ValueError, match="label_name required"):
        convert_seqcls_to_multitask(seqcls_state, label_name=None)


def test_convert_seqlabel_to_multitask(minimal_mlm_config):
    """Test SeqLabel → Multitask conversion."""
    seqlabel_state = scbert.SCBertForSequenceLabeling(minimal_mlm_config).state_dict()
    multitask_state = convert_seqlabel_to_multitask(seqlabel_state)
    assert any("cls.predictions.predictions" in k for k in multitask_state)


def test_mlm_conversion_preserves_encoder_weights(minimal_mlm_config):
    """Test that encoder weights are unchanged after conversion."""
    mlm_state = scbert.SCBertForMaskedLM(minimal_mlm_config).state_dict()
    multitask_state = convert_mlm_to_multitask(mlm_state)

    encoder_keys = [k for k in mlm_state if "scbert.encoder" in k]
    for key in encoder_keys:
        assert torch.allclose(mlm_state[key], multitask_state[key])


def test_conversion_preserves_pooler(minimal_seqcls_config):
    """Test that pooler weights are preserved in SeqCls conversion."""
    seqcls_state = scbert.SCBertForSequenceClassification(
        minimal_seqcls_config
    ).state_dict()
    multitask_state = convert_seqcls_to_multitask(seqcls_state, label_name="cell_type")

    pooler_keys = [k for k in seqcls_state if "pooler" in k]
    for key in pooler_keys:
        assert torch.allclose(seqcls_state[key], multitask_state[key])


# --- Real Checkpoint Tests ---


def _load_checkpoint(ckpt_path: Path) -> tuple[dict, Any]:
    """
    Load checkpoint and return (cleaned_state_dict, config_from_hyperparams).

    Config can be SCBertConfig or LlamaConfig depending on checkpoint type.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # Remove "model." prefix from PyTorch Lightning checkpoints
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k[6:] if k.startswith("model.") else k] = v

    # Extract config from hyperparameters if available
    config = None
    if isinstance(ckpt, dict) and "hyper_parameters" in ckpt:
        hparams = ckpt["hyper_parameters"]
        if "model_config" in hparams:
            config = hparams["model_config"]
        elif "config" in hparams:
            config = hparams["config"]

    return cleaned, config


def _create_test_input(config: Any, batch_size: int = 2, seq_len: int = 8) -> tuple:
    """Create test input with correct shape for the model."""
    assert config.fields is not None, "Config must have fields"
    num_fields = len(config.fields)

    # Create input_ids with correct vocab size for each field
    input_ids = torch.zeros((batch_size, num_fields, seq_len), dtype=torch.long)
    for i, field in enumerate(config.fields):
        vocab_size = field.vocab_size or 100
        input_ids[:, i, :] = torch.randint(0, vocab_size, (batch_size, seq_len))

    attention_mask = torch.ones((batch_size, seq_len))
    return input_ids, attention_mask


@pytest.mark.parametrize(
    ("ckpt_dir", "ckpt_name", "conversion_fn"),
    [
        ("scbert", "mlm.ckpt", convert_mlm_to_multitask),
        ("scbert", "seq_lab.ckpt", convert_seqlabel_to_multitask),
        ("scbert", "seq_cls.ckpt", convert_seqcls_to_multitask),
        ("llama", "llama_mlm.ckpt", convert_mlm_to_multitask),
        ("llama", "llama_seq_cls.ckpt", convert_seqcls_to_multitask),
    ],
)
def test_real_checkpoint_migration(ckpt_dir, ckpt_name, conversion_fn):
    """
    Test migration produces identical embeddings and logits.

    Model-agnostic: works with SCBert and Llama using config from checkpoint.
    """
    ckpt_path = Path(__file__).parent / "resources" / "ckpts" / ckpt_dir / ckpt_name
    if not ckpt_path.exists():
        pytest.skip(f"Checkpoint not found: {ckpt_path}")

    # Load checkpoint
    raw_state_dict, config = _load_checkpoint(ckpt_path)
    if config is None:
        pytest.skip(f"No config in hyperparameters: {ckpt_path}")

    # Detect type and label
    from run.migrate_checkpoints_to_multitask import detect_checkpoint_type

    ckpt_type, label_name = detect_checkpoint_type(raw_state_dict)
    is_seqcls = "seq_cls" in ckpt_name

    # Get old model class
    if hasattr(config, "model_type") and config.model_type == "scllama":
        from bmfm_targets.models.predictive.llama.model import (
            LlamaForMaskedLMModel,
            LlamaForMultiTaskModel,
            LlamaForSequenceClassificationModel,
        )

        old_model_class = (
            LlamaForSequenceClassificationModel if is_seqcls else LlamaForMaskedLMModel
        )
        new_model_class = LlamaForMultiTaskModel
    else:
        old_model_class = (
            scbert.SCBertForSequenceClassification
            if is_seqcls
            else scbert.SCBertForSequenceLabeling
            if "seq_lab" in ckpt_name
            else scbert.SCBertForMaskedLM
        )
        new_model_class = scbert.SCBertForMultiTaskModeling

    # Remap for old model
    # Llama: base_model.core.* -> core.*, classifiers.X.* -> (no classifier in old llama seqcls)
    # SCBert: base_model.* -> scbert.*, classifiers.X.* -> classifier.*
    is_llama = hasattr(config, "model_type") and config.model_type == "scllama"
    state_for_old = {}
    for k, v in raw_state_dict.items():
        if k.startswith("base_model.core.") and is_llama:
            # Llama: base_model.core.* -> core.*
            state_for_old[k.replace("base_model.", "")] = v
        elif k.startswith("base_model."):
            # SCBert: base_model.* -> scbert.*
            state_for_old[k.replace("base_model.", "scbert.")] = v
        elif k.startswith("classifiers.") and label_name and label_name in k:
            # SCBert only: classifiers.X.* -> classifier.*
            state_for_old[k.replace(f"classifiers.{label_name}.", "classifier.")] = v
        else:
            state_for_old[k] = v

    # Load and run old model
    old_model = old_model_class(config)
    old_model.load_state_dict(state_for_old, strict=False)
    old_model.eval()

    input_ids, mask = _create_test_input(config)
    with torch.no_grad():
        out_old = old_model(input_ids, attention_mask=mask, output_hidden_states=True)

    # Convert and load new model
    converted = (
        conversion_fn(raw_state_dict, label_name=label_name)
        if label_name
        else conversion_fn(raw_state_dict)
    )
    new_model = new_model_class(config)
    missing, unexpected = new_model.load_state_dict(converted, strict=False)
    assert not any(
        "encoder" in k for k in missing
    ), f"Lost encoder: {[k for k in missing if 'encoder' in k]}"
    new_model.eval()

    with torch.no_grad():
        out_new = new_model(input_ids, attention_mask=mask, output_hidden_states=True)

    # Compare embeddings
    if is_seqcls:
        # SeqCls: compare pooled embeddings
        old_emb = (
            out_old.embeddings
            if hasattr(out_old, "embeddings")
            else out_old.pooler_output
        )
        new_emb = out_new.embeddings
    else:
        # MLM: compare last hidden state
        # Old model returns last_hidden_state directly, new returns tuple/list of all layers
        old_emb = (
            out_old.hidden_states
            if not isinstance(out_old.hidden_states, tuple | list)
            else out_old.hidden_states[-1]
        )
        new_emb = (
            out_new.hidden_states[-1]
            if isinstance(out_new.hidden_states, tuple | list)
            else out_new.hidden_states
        )

    assert torch.allclose(
        old_emb, new_emb, atol=1e-5
    ), f"Embedding mismatch: {(old_emb - new_emb).abs().max():.6f}"

    # Compare logits (skip for llama seq_cls which has no classifier head in old model)
    is_llama = hasattr(config, "model_type") and config.model_type == "scllama"
    if is_seqcls and is_llama:
        # Old llama seq_cls has no classifier head, only encoder - skip logits comparison
        pass
    elif is_seqcls:
        old_logits = (
            out_old.logits
            if isinstance(out_old.logits, torch.Tensor)
            else out_old.logits.get(label_name, list(out_old.logits.values())[0])
        )
        new_logits = (
            out_new.logits[label_name]
            if label_name
            else list(out_new.logits.values())[0]
        )
        assert torch.allclose(
            old_logits, new_logits, atol=1e-4
        ), f"Logit mismatch: max={(old_logits - new_logits).abs().max():.6f}"
    else:
        old_logits = (
            out_old.logits
            if isinstance(out_old.logits, torch.Tensor)
            else list(out_old.logits.values())[0]
        )
        new_logits = list(out_new.logits.values())[0]
        assert torch.allclose(
            old_logits, new_logits, atol=1e-4
        ), f"Logit mismatch: max={(old_logits - new_logits).abs().max():.6f}"


# --- Automatic Migration Tests ---


@pytest.mark.parametrize("ckpt_name", ["mlm.ckpt", "seq_cls.ckpt"])
def test_automatic_migration_via_model_config(ckpt_name):
    """Test checkpoint auto-migration via model_config.checkpoint."""
    from bmfm_targets.config import TrainerConfig
    from bmfm_targets.training.modules.base import BaseTrainingModule

    ckpt_path = Path(__file__).parent / "resources" / "ckpts" / "scbert" / ckpt_name
    if not ckpt_path.exists():
        pytest.skip(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "hyper_parameters" not in ckpt or "model_config" not in ckpt["hyper_parameters"]:
        pytest.skip(f"No config in checkpoint: {ckpt_path}")

    model_config = ckpt["hyper_parameters"]["model_config"]
    model_config.checkpoint = str(ckpt_path)

    module = BaseTrainingModule(
        model_config=model_config,
        trainer_config=TrainerConfig(losses=[]),
        tokenizer=None,
        label_dict=None,
    )

    assert module.model is not None
    assert module.model_config.checkpoint is None  # Cleared after loading


def test_automatic_migration_via_lightning_hook():
    """Test checkpoint auto-migration via Lightning's on_load_checkpoint hook."""
    from bmfm_targets.config import TrainerConfig
    from bmfm_targets.training.modules.base import BaseTrainingModule

    ckpt_path = Path(__file__).parent / "resources" / "ckpts" / "scbert" / "mlm.ckpt"
    if not ckpt_path.exists():
        pytest.skip(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "hyper_parameters" not in ckpt or "model_config" not in ckpt["hyper_parameters"]:
        pytest.skip(f"No config in checkpoint: {ckpt_path}")

    model_config = ckpt["hyper_parameters"]["model_config"]
    model_config.checkpoint = None

    module = BaseTrainingModule(
        model_config=model_config,
        trainer_config=TrainerConfig(losses=[]),
        tokenizer=None,
        label_dict=None,
    )

    # Simulate Lightning's checkpoint loading
    checkpoint_for_lightning = {
        "state_dict": {f"model.{k}": v for k, v in ckpt["state_dict"].items()}
    }

    module.on_load_checkpoint(checkpoint_for_lightning)

    # Verify migration happened
    cleaned = {
        k[6:] if k.startswith("model.") else k: v
        for k, v in checkpoint_for_lightning["state_dict"].items()
    }
    assert any("cls.predictions.predictions" in k for k in cleaned.keys())


@pytest.mark.parametrize(
    ("ckpt_name", "pooling_method"),
    [("mlm.ckpt", "first_token"), ("seq_cls.ckpt", "pooling_layer")],
)
def test_pooler_embeddings_consistency_after_migration(
    ckpt_name, pooling_method, pl_data_module_zheng68k_seq_cls
):
    """Test that migrated checkpoints work with correct pooling_method."""
    from bmfm_targets import config
    from bmfm_targets.models.predictive.layers import get_embeddings_from_outputs

    ckpt_path = Path(__file__).parent / "resources" / "ckpts" / "scbert" / ckpt_name
    if not ckpt_path.exists():
        pytest.skip(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    data_module = pl_data_module_zheng68k_seq_cls
    data_module.setup("predict")
    batch = next(iter(data_module.predict_dataloader()))

    # Automatic migration via model_config.checkpoint
    model_config = ckpt["hyper_parameters"]["model_config"]
    model_config.checkpoint = str(ckpt_path)
    migrated_module = MultiTaskTrainingModule(
        model_config, config.TrainerConfig(losses=[]), label_dict=data_module.label_dict
    )
    migrated_module.eval()

    with torch.no_grad():
        outputs = migrated_module.model(
            batch["input_ids"], batch["attention_mask"], output_hidden_states=True
        )
        embeddings = get_embeddings_from_outputs(
            outputs, batch["attention_mask"], pooling_method
        )

    # Verify embeddings are extracted correctly with explicit pooling_method
    assert embeddings is not None
    assert embeddings.shape[0] == batch["input_ids"].shape[0]


# Made with Bob
