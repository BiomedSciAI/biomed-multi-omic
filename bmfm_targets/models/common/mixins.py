import torch
from transformers.utils import logging

from bmfm_targets.training.serialization import prepare_model_dict_from_checkpoint

logger = logging.get_logger(__name__)


class InitWeightsMixin:
    def _init_weights(self, module):
        pass


class CheckpointMixin:
    def load_checkpoint(self):
        if self.config.checkpoint:
            logger.info("Loading model from checkpoint " + str(self.config.checkpoint))
            model_dict = prepare_model_dict_from_checkpoint(self.config.checkpoint)
            key_report = self.load_state_dict(model_dict, strict=False)
            logger.info(f"Loading complete. {len(model_dict)} layers in ckpt.")
            logger.info(f"Unexpected keys: {key_report.unexpected_keys}")
            logger.info(f"Missing keys: {key_report.missing_keys}")

    @classmethod
    def _from_config(cls, config, **kwargs):
        torch_dtype = kwargs.pop("torch_dtype", config.torch_dtype)
        if isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype)

        dtype_orig = None
        if torch_dtype is not None:
            dtype_orig = cls._set_default_torch_dtype(torch_dtype)

        if "attn_implementation" in kwargs:
            config._attn_implementation = kwargs.pop("attn_implementation")

        model = cls(config, **kwargs)
        if dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)

        return model
