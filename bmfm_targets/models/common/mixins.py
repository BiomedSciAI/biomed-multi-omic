import torch
from transformers.utils import logging

from bmfm_targets.training.serialization import prepare_model_dict_from_checkpoint

logger = logging.get_logger(__name__)


class InitWeightsMixin:
    def _init_weights(self, module):
        pass


class AttentionMaskMixin:
    """Provides get_extended_attention_mask and invert_attention_mask removed in transformers >= 5.17."""

    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: tuple[int, ...] | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """
        Makes broadcastable attention mask so that masked tokens are ignored.

        Args:
        ----
            attention_mask (torch.Tensor): Mask with 1 for tokens to attend to, 0 for tokens to ignore.
            input_shape (tuple[int, ...], optional): Shape of the input tensor.
            dtype (torch.dtype, optional): Target dtype, defaults to self.dtype.

        Returns:
        -------
            torch.Tensor: The extended attention mask with 0.0 for valid tokens and -inf/min for masked tokens.
        """
        if dtype is None:
            try:
                dtype = self.dtype
            except (AttributeError, StopIteration):
                dtype = torch.float32

        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            if getattr(getattr(self, "config", None), "is_decoder", False):
                batch_size, seq_length = attention_mask.shape
                seq_ids = torch.arange(seq_length, device=attention_mask.device)
                causal_mask = (
                    seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
                    <= seq_ids[None, :, None]
                )
                causal_mask = causal_mask.to(attention_mask.dtype)
                extended_attention_mask = (causal_mask * attention_mask[:, None, :])[
                    :, None, :, :
                ]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for attention_mask (shape {attention_mask.shape})"
            )

        extended_attention_mask = extended_attention_mask.to(dtype=dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(
            dtype
        ).min
        return extended_attention_mask

    def invert_attention_mask(
        self, encoder_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Invert an attention mask (switches 0 and 1) and makes it broadcastable.

        Args:
        ----
            encoder_attention_mask (torch.Tensor): Attention mask.

        Returns:
        -------
            torch.Tensor: The inverted attention mask.
        """
        try:
            dtype = self.dtype
        except (AttributeError, StopIteration):
            dtype = torch.float32

        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for encoder_attention_mask (shape {encoder_attention_mask.shape})"
            )

        encoder_extended_attention_mask = encoder_extended_attention_mask.to(
            dtype=dtype
        )
        encoder_extended_attention_mask = (
            1.0 - encoder_extended_attention_mask
        ) * torch.finfo(dtype).min
        return encoder_extended_attention_mask


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
