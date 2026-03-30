import tempfile
from pathlib import Path

import pytest
import scanpy as sc
from pytorch_lightning.callbacks import ModelCheckpoint

import bmfm_targets as bmfm
from bmfm_targets import config
from bmfm_targets.tasks.task_utils import make_trainer_for_task
from bmfm_targets.tokenization import load_tokenizer
from bmfm_targets.training.losses import (
    CrossEntropyObjective,
    LossTask,
    MSEObjective,
    WCEDFieldSource,
)
from bmfm_targets.training.modules import MultiTaskTrainingModule


@pytest.fixture(scope="module")
def wced_multitask_fields():
    """Create WCED fields for testing."""
    field_dicts = [
        {
            "field_name": "genes",
            "is_masked": False,
            "vocab_update_strategy": "static",
        },
        {
            "field_name": "expressions",
            "is_masked": True,
            "decode_modes": {
                "wced": {
                    "vocab_field": "genes",
                    "logit_outputs": ["mse", "is_zero_bce"],
                }
            },
            "tokenization_strategy": "continuous_value_encoder",
            "vocab_update_strategy": "static",
            "encoder_kwargs": {
                "kind": "mlp_with_special_token_embedding",
                "zero_as_special_token": True,
            },
        },
    ]

    fields = [config.FieldInfo(**fd) for fd in field_dicts]
    tokenizer = load_tokenizer("all_genes")
    for field in fields:
        if field.is_input:
            field.update_vocab_size(tokenizer)
    return fields


@pytest.fixture(scope="module")
def pbmc3k_test_data():
    """Load and preprocess pbmc3k dataset for testing."""
    adata = sc.datasets.pbmc3k()
    adata = adata[:5, :].copy()
    sc.pp.log1p(adata)
    return adata


@pytest.fixture(scope="module")
def small_wced_multitask_checkpoint(wced_multitask_fields):
    """Create a small WCED multitask checkpoint without downloading from HuggingFace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create small model config with WCED
        label_columns = [
            config.LabelColumnInfo(label_column_name="cell_type", n_unique_values=5),
            config.LabelColumnInfo(label_column_name="tissue", n_unique_values=3),
            config.LabelColumnInfo(
                label_column_name="tissue_general", n_unique_values=2
            ),
        ]

        model_config = config.SCBertConfig(
            fields=wced_multitask_fields,
            label_columns=label_columns,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            hidden_size=32,  # Small hidden size for fast testing
        )

        # Create WCED losses + label losses (true multitask)
        trainer_config = config.TrainerConfig(
            losses=[
                LossTask(
                    WCEDFieldSource(
                        field_name="expressions",
                        wced_target="non_input_genes",
                    ),
                    MSEObjective(),
                ),
                LossTask.from_label("cell_type", objective=CrossEntropyObjective()),
                LossTask.from_label("tissue", objective=CrossEntropyObjective()),
                LossTask.from_label(
                    "tissue_general", objective=CrossEntropyObjective()
                ),
            ]
        )

        # Save tokenizer
        tokenizer = load_tokenizer("all_genes")
        tokenizer.save_pretrained(tmpdir, legacy_format=not tokenizer.is_fast)

        # Create training module (no actual training needed for inference test)
        training_module = MultiTaskTrainingModule(
            model_config, trainer_config, tokenizer=tokenizer
        )

        # Save checkpoint
        task_config = config.TrainingTaskConfig(
            default_root_dir=tmpdir,
            max_epochs=1,
            max_steps=1,
            precision="32",
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[
                ModelCheckpoint(
                    dirpath=Path(tmpdir),
                    save_last=True,
                    save_top_k=0,
                    filename="test",
                    auto_insert_metric_name=False,
                )
            ],
        )

        pl_trainer = make_trainer_for_task(task_config)
        # Save checkpoint without training (just initialize)
        pl_trainer.strategy.connect(training_module)
        pl_trainer.save_checkpoint(f"{tmpdir}/last.ckpt")

        ckpt_path = f"{tmpdir}/last.ckpt"
        yield ckpt_path


def test_bmfm_adata_inference_with_small_wced_model(
    small_wced_multitask_checkpoint, pbmc3k_test_data
):
    """Test inference with a small locally-created WCED multitask model."""
    adata = pbmc3k_test_data.copy()

    bmfm.inference(adata, checkpoint=small_wced_multitask_checkpoint)

    # Check that predictions were added
    assert "bmfm_pred_cell_type" in adata.obs.columns
    assert "bmfm_pred_tissue" in adata.obs.columns
    assert "bmfm_pred_tissue_general" in adata.obs.columns
    assert "X_bmfm" in adata.obsm
    assert adata.obsm["X_bmfm"].shape == (5, 32)  # hidden_size=32


def test_bmfm_adata_inference_with_pooling_method(
    small_wced_multitask_checkpoint, pbmc3k_test_data
):
    """Test that pooling_method parameter correctly overrides checkpoint default."""
    adata = pbmc3k_test_data.copy()

    adata_default = adata.copy()
    bmfm.inference(
        adata_default,
        checkpoint=small_wced_multitask_checkpoint,
        embedding_key="X_bmfm_default",
    )

    adata_first_token = adata.copy()
    bmfm.inference(
        adata_first_token,
        checkpoint=small_wced_multitask_checkpoint,
        pooling_method="first_token",
        embedding_key="X_bmfm_first",
    )

    adata_mean = adata.copy()
    bmfm.inference(
        adata_mean,
        checkpoint=small_wced_multitask_checkpoint,
        pooling_method="mean_pooling",
        embedding_key="X_bmfm_mean",
    )

    adata_pos0 = adata.copy()
    bmfm.inference(
        adata_pos0,
        checkpoint=small_wced_multitask_checkpoint,
        pooling_method=0,
        embedding_key="X_bmfm_pos0",
    )

    adata_pos1 = adata.copy()
    bmfm.inference(
        adata_pos1,
        checkpoint=small_wced_multitask_checkpoint,
        pooling_method=1,
        embedding_key="X_bmfm_pos1",
    )

    adata_multi = adata.copy()
    bmfm.inference(
        adata_multi,
        checkpoint=small_wced_multitask_checkpoint,
        pooling_method=[0, 1],
        embedding_key="X_bmfm_multi",
    )

    assert "X_bmfm_default" in adata_default.obsm
    assert "X_bmfm_first" in adata_first_token.obsm
    assert "X_bmfm_mean" in adata_mean.obsm
    assert "X_bmfm_pos0" in adata_pos0.obsm
    assert "X_bmfm_pos1" in adata_pos1.obsm
    assert "X_bmfm_multi" in adata_multi.obsm

    assert adata_default.obsm["X_bmfm_default"].shape == (5, 32)
    assert adata_first_token.obsm["X_bmfm_first"].shape == (5, 32)
    assert adata_mean.obsm["X_bmfm_mean"].shape == (5, 32)
    assert adata_pos0.obsm["X_bmfm_pos0"].shape == (5, 32)
    assert adata_pos1.obsm["X_bmfm_pos1"].shape == (5, 32)
    assert adata_multi.obsm["X_bmfm_multi"].shape == (5, 32 * 2)

    assert "bmfm_pred_cell_type" in adata_first_token.obs.columns
    assert "bmfm_pred_cell_type" in adata_mean.obs.columns
    assert "bmfm_pred_cell_type" in adata_pos1.obs.columns
    assert "bmfm_pred_cell_type" in adata_multi.obs.columns


def test_bmfm_adata_inference_with_mixed_pooling_methods(
    small_wced_multitask_checkpoint, pbmc3k_test_data
):
    """Test inference with mixed pooling methods (strings and integers)."""
    adata = pbmc3k_test_data.copy()

    # Test mixing string pooling methods
    adata_mixed_strings = adata.copy()
    bmfm.inference(
        adata_mixed_strings,
        checkpoint=small_wced_multitask_checkpoint,
        pooling_method=["pooling_layer", "mean_pooling"],
        embedding_key="X_bmfm_mixed_strings",
    )
    assert "X_bmfm_mixed_strings" in adata_mixed_strings.obsm
    assert adata_mixed_strings.obsm["X_bmfm_mixed_strings"].shape == (
        5,
        32 * 2,
    )  # 2 methods

    # Test mixing positions and string methods
    adata_mixed_pos_str = adata.copy()
    bmfm.inference(
        adata_mixed_pos_str,
        checkpoint=small_wced_multitask_checkpoint,
        pooling_method=[0, "pooling_layer", 1],
        embedding_key="X_bmfm_mixed_pos_str",
    )
    assert "X_bmfm_mixed_pos_str" in adata_mixed_pos_str.obsm
    assert adata_mixed_pos_str.obsm["X_bmfm_mixed_pos_str"].shape == (
        5,
        32 * 3,
    )  # 3 methods

    # Test all pooling methods combined
    adata_all_methods = adata.copy()
    bmfm.inference(
        adata_all_methods,
        checkpoint=small_wced_multitask_checkpoint,
        pooling_method=["first_token", "pooling_layer", "mean_pooling"],
        embedding_key="X_bmfm_all_methods",
    )
    assert "X_bmfm_all_methods" in adata_all_methods.obsm
    assert adata_all_methods.obsm["X_bmfm_all_methods"].shape == (
        5,
        32 * 3,
    )  # 3 methods

    # Test complex combination
    adata_complex = adata.copy()
    bmfm.inference(
        adata_complex,
        checkpoint=small_wced_multitask_checkpoint,
        pooling_method=[0, "pooling_layer", 1, "mean_pooling", 2],
        embedding_key="X_bmfm_complex",
    )
    assert "X_bmfm_complex" in adata_complex.obsm
    assert adata_complex.obsm["X_bmfm_complex"].shape == (5, 32 * 5)  # 5 methods

    # Verify predictions still work
    assert "bmfm_pred_cell_type" in adata_mixed_strings.obs.columns
    assert "bmfm_pred_cell_type" in adata_mixed_pos_str.obs.columns
    assert "bmfm_pred_cell_type" in adata_all_methods.obs.columns
    assert "bmfm_pred_cell_type" in adata_complex.obs.columns
