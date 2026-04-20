"""Unit tests for LoRA configuration."""


from bmfm_targets.config import TrainerConfig
from bmfm_targets.config.training_config import LoraConfigWrapper
from bmfm_targets.models.predictive.llama.model import LlamaForMultiTaskModel


def test_lora_config_wrapper_with_custom_r():
    """Test that LoraConfigWrapper accepts custom r value."""
    config = LoraConfigWrapper(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "proj"],
    )

    assert config.r == 8
    assert config.lora_alpha == 16
    peft_config = config.to_peft_config()
    assert peft_config.r == 8


def test_trainer_config_lora_presets():
    """Test that string presets return correct target_modules."""
    trainer_config = TrainerConfig(lora_config="llama")
    lora_config = trainer_config.get_lora_config()

    assert lora_config is not None
    assert lora_config.target_modules == ["c_attn", "proj", "c_fc1", "c_fc2", "c_proj"]


def test_llama_model_accepts_output_attentions():
    """Test that LlamaForMultiTaskModel.forward() accepts output_attentions and return_dict."""
    import inspect

    # Check the forward method signature
    sig = inspect.signature(LlamaForMultiTaskModel.forward)
    params = list(sig.parameters.keys())

    # Verify the parameters we added are present
    assert "output_attentions" in params, "output_attentions parameter missing"
    assert "return_dict" in params, "return_dict parameter missing"
    assert "kwargs" in params, "**kwargs parameter missing"


# Made with Bob
