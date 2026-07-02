import warnings

import torch
from torch import nn
from torch.nn import functional as F
from transformers.configuration_utils import PretrainedConfig
from transformers.pytorch_utils import (
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

from .layers import SCSelfOutput


def attention_factory(
    attention: str | None, config: PretrainedConfig, default_attention_class
):
    """
    Factory class that creates instances of custom attention.

    Args:
    ----
        attention (str): Name of attention to create (torch, ...). If None default_attention_class is created.
        config (PretrainedConfig): Parameters of attention.
        default_attention_class: default attention class.
    """
    if not attention:
        return default_attention_class(config)

    if attention == "torch":
        return SCBertCustomAttention(config, self_attention=SelfTorchAttention(config))
    else:
        warnings.warn(
            f"Not supported attention {attention}, reverting to default. Select from torch, ..."
        )
        return default_attention_class(config)


class SelfTorchAttention(nn.Module):
    """
    Attention implementation based on Torch attention.

    Torch attention is multiple different implementations of attention
    that PyTorch automatically selects depending on available kernels
    when you call torch.nn.functional.scaled_dot_product_attention.
    For instance, flash attention is compatible only with Ampere architecture or newer.
    If you run this code on A100, it will use flash attention.
    On V100, torch could select between Memory efficient attention or C++ fused kernel.
    Please refer to torch.nn.functional.scaled_dot_product_attention
    (https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) for more info.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout_rate = config.attention_probs_dropout_prob
        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_value: (
            tuple[torch.Tensor, torch.Tensor]
            | tuple[torch.FloatTensor, torch.FloatTensor, tuple[torch.FloatTensor]]
            | None
        ) = None,
        output_attentions: bool | None = False,
    ) -> tuple[torch.Tensor, ...]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None
        key_layer: torch.Tensor | tuple[torch.Tensor] | None = None
        value_layer: torch.Tensor | tuple[torch.FloatTensor] | None = None
        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.

        if head_mask is not None:
            raise ValueError("Head mask is not supported by custom attention class.")

        if output_attentions:
            raise ValueError(
                "Custom attention class does not support output of attention probs."
            )

        dropout_p = self.dropout_rate if self.training else 0.0
        context_layer = F.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)  # type: ignore
        return outputs


class SCBertCustomAttention(nn.Module):
    """Alternative implementation of SCBertAttention that uses custom attention instead of vanilla implementation."""

    def __init__(self, config: PretrainedConfig, self_attention):
        """
        Args:
        ----
            self_attention: instance of a class with custom implementation of self attention (e.g., SelfTorchAttention).
        """
        super().__init__()
        self.self = self_attention
        self.output = SCSelfOutput(config)
        self.pruned_heads: set[int] = set()

    def prune_heads(self, heads: list[int]):
        """
        Prune heads of the layer.

        Args:
        ----
            heads (list[int]): A list of heads to prune.
        """
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_value: tuple[tuple[torch.FloatTensor]] | None = None,
        output_attentions: bool | None = False,
    ) -> tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs
