"""Test LoRA configuration with layers_to_transform and layers_pattern."""

import tempfile

import pytest

from bmfm_targets import config
from bmfm_targets.config.training_config import LoraConfigWrapper
from bmfm_targets.models.predictive.llama.config import LlamaForSequenceClassification
from bmfm_targets.training.losses import CrossEntropyObjective, LabelSource, LossTask
from bmfm_targets.training.modules.sequence_classification import (
    SequenceClassificationTrainingModule,
)


def test_lora_config_wrapper_with_layers():
    """Test that LoraConfigWrapper correctly passes layers_to_transform and layers_pattern."""
    lora_config = LoraConfigWrapper(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "proj"],
        layers_to_transform=[0, 1, 2],
        layers_pattern="blocks",
    )

    peft_config = lora_config.to_peft_config()

    assert peft_config.r == 8
    assert peft_config.lora_alpha == 16
    # PEFT converts target_modules list to set internally
    assert set(peft_config.target_modules) == {"c_attn", "proj"}
    assert peft_config.layers_to_transform == [0, 1, 2]
    assert peft_config.layers_pattern == "blocks"


def test_lora_config_wrapper_without_layers():
    """Test that LoraConfigWrapper works without layers_to_transform (applies to all layers)."""
    lora_config = LoraConfigWrapper(
        r=16,
        lora_alpha=32,
        target_modules=["c_attn", "proj", "c_fc1", "c_fc2", "c_proj"],
    )

    peft_config = lora_config.to_peft_config()

    assert peft_config.r == 16
    assert peft_config.lora_alpha == 32
    assert peft_config.layers_to_transform is None
    assert peft_config.layers_pattern is None


def test_lora_with_limited_layers_reduces_params(
    pl_data_module_zheng68k_seq_cls,
    gene2vec_fields: list[config.FieldInfo],
):
    """Test that using layers_to_transform actually reduces trainable parameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create model config
        model_config = LlamaForSequenceClassification(
            batch_size=pl_data_module_zheng68k_seq_cls.batch_size,
            label_columns=pl_data_module_zheng68k_seq_cls.label_columns,
            fields=gene2vec_fields,
            num_attention_heads=2,
            num_hidden_layers=6,  # 6 layers total
            intermediate_size=128,
        )

        # Test 1: LoRA on all layers
        trainer_config_all = config.TrainerConfig(
            losses=[
                LossTask(
                    source=LabelSource(label_name="celltype"),
                    objective=CrossEntropyObjective(),
                )
            ],
            lora_config=LoraConfigWrapper(
                r=8,
                lora_alpha=16,
                target_modules=["c_attn", "proj"],
                # No layers_to_transform - applies to all layers
            ),
        )

        module_all = SequenceClassificationTrainingModule(
            model_config,
            trainer_config_all,
            label_dict=pl_data_module_zheng68k_seq_cls.label_dict,
        )

        # Count trainable params for all layers
        trainable_params_all = sum(
            p.numel() for p in module_all.parameters() if p.requires_grad
        )

        # Test 2: LoRA on first 3 layers only
        trainer_config_limited = config.TrainerConfig(
            losses=[
                LossTask(
                    source=LabelSource(label_name="celltype"),
                    objective=CrossEntropyObjective(),
                )
            ],
            lora_config=LoraConfigWrapper(
                r=8,
                lora_alpha=16,
                target_modules=["c_attn", "proj"],
                layers_to_transform=[0, 1, 2],  # Only first 3 layers
                layers_pattern="blocks",
            ),
        )

        module_limited = SequenceClassificationTrainingModule(
            model_config,
            trainer_config_limited,
            label_dict=pl_data_module_zheng68k_seq_cls.label_dict,
        )

        # Count trainable params for limited layers
        trainable_params_limited = sum(
            p.numel() for p in module_limited.parameters() if p.requires_grad
        )

        # Assert that limited layers has fewer trainable params
        assert trainable_params_limited < trainable_params_all, (
            f"Limited layers ({trainable_params_limited}) should have fewer "
            f"trainable params than all layers ({trainable_params_all})"
        )

        # The ratio should be approximately 3/6 = 0.5 (since we're using 3 out of 6 layers)
        # Allow some tolerance for the classification head and other parameters
        ratio = trainable_params_limited / trainable_params_all
        assert (
            0.4 < ratio < 0.7
        ), f"Expected ratio around 0.5 (3/6 layers), got {ratio:.2f}"


def test_trainer_config_get_lora_config_with_wrapper():
    """Test that TrainerConfig.get_lora_config() returns the wrapper correctly."""
    lora_wrapper = LoraConfigWrapper(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],
        layers_to_transform=[0, 1, 2, 3, 4, 5],
        layers_pattern="blocks",
    )

    trainer_config = config.TrainerConfig(
        losses=[],
        lora_config=lora_wrapper,
    )

    retrieved_config = trainer_config.get_lora_config()

    assert retrieved_config.r == 8
    assert retrieved_config.lora_alpha == 16
    assert retrieved_config.layers_to_transform == [0, 1, 2, 3, 4, 5]
    assert retrieved_config.layers_pattern == "blocks"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
