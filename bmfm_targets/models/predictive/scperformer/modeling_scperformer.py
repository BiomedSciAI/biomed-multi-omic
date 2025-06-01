"""SCPerformer model was adapted from PerformerModel in transformers library."""

import math
from collections.abc import Callable
from contextlib import contextmanager
from functools import partial

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import repeat
from torch import nn
from torch.cuda.amp import autocast
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging

from bmfm_targets.config.model_config import (
    FeatureGenerationAlgorithm,
    PerformerKernel,
    SCPerformerConfig,
)
from bmfm_targets.models.model_utils import (
    MaskedLMOutputWithEmbeddings,
    SequenceClassifierOutputWithEmbeddings,
)
from bmfm_targets.models.predictive.layers import (
    SCEmbeddingsLayer,
    SCIntermediate,
    SCMultiTaskHead,
    SCOnlyMLMHead,
    SCOutput,
    SCPooler,
    SCSelfOutput,
    SCSequenceLabelingHead,
)
from bmfm_targets.training.serialization import prepare_model_dict_from_checkpoint

try:
    from apex import amp

    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False


@contextmanager
def null_context():
    yield


# from .modeling_utils import find_pruneable_heads_and_indices, prune_linear_layer


def exists(val):
    return val is not None


logger = logging.get_logger(__name__)

KERNEL_CALLABLES = {
    PerformerKernel.cosh: lambda x, h: torch.cat(
        (torch.exp(h + x), torch.exp(h - x)), dim=-1
    ),
    PerformerKernel.exp: lambda x, h: torch.exp(h + x),  # Default
    PerformerKernel.elu: lambda x: F.elu(x) + 1,
    PerformerKernel.relu: F.relu,
}


def resolve_enum(enum_class, value):
    return enum_class[value] if isinstance(value, str) else value


def softmax_kernel(
    data,
    *,
    projection_matrix,
    is_query,
    kernel_fn,
    kernel_epsilon=0.001,
    normalize_output=True,
):
    if projection_matrix is None:
        raise ValueError("projection_matrix cannot be None for softmax kernel")
    b, h, *_ = data.shape
    data_normalizer = (data.shape[-1] ** -0.25) if normalize_output else 1.0
    data_dash = torch.einsum(
        "...id,...jd->...ij", (data_normalizer * data), projection_matrix
    )
    diag_data = -torch.sum(data**2, dim=-1, keepdim=True) / 2.0

    if is_query:
        q_indices = diag_data.argmax(-1).unsqueeze(-1)
        q_stabilizer = diag_data.gather(-1, q_indices)
        q_kernel_output = kernel_fn(data_dash - q_stabilizer, diag_data)
        normalizing_constant = q_kernel_output.shape[-1] ** -0.5
        return normalizing_constant * (q_kernel_output + kernel_epsilon)

    else:
        k_stabilizer = torch.max(diag_data)
        k_kernel_output = kernel_fn(data_dash - k_stabilizer, diag_data)
        normalizing_constant = k_kernel_output.shape[-1] ** -0.5
        return normalizing_constant * (k_kernel_output + kernel_epsilon)


def generalized_kernel(
    data, *, projection_matrix, kernel_fn, kernel_epsilon, normalize_output
):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_output else 1.0

    if projection_matrix is None:
        raise ValueError("projection_matrix cannot be None for generalized kernel")

    data_dash = torch.einsum(
        "...id,...jd->...ij", (data_normalizer * data), projection_matrix
    )

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)


# non-causal linear attention
def linear_attention(q, k, v):
    k_cumsum = k.sum(dim=-2)
    D_inv = 1.0 / torch.einsum("...nd,...d->...n", q, k_cumsum.type_as(q))
    context = torch.einsum("...nd,...ne->...de", k, v)
    out = torch.einsum("...de,...nd,...n->...ne", context, q, D_inv)
    return out


# efficient causal linear attention, created by EPFL
# TODO: rewrite EPFL's CUDA kernel to do mixed precision and remove half to float conversion and back
def causal_linear_attention(q, k, v, eps=1e-6):
    from fast_transformers.causal_product import CausalDotProduct

    autocast_enabled = torch.is_autocast_enabled()
    is_half = isinstance(q, torch.cuda.HalfTensor)
    assert (
        not is_half or APEX_AVAILABLE
    ), "half tensors can only be used if nvidia apex is available"
    cuda_context = (
        null_context if not autocast_enabled else partial(autocast, enabled=False)
    )

    causal_dot_product_fn = (
        amp.float_function(CausalDotProduct.apply)
        if is_half
        else CausalDotProduct.apply
    )

    k_cumsum = k.cumsum(dim=-2) + eps
    D_inv = 1.0 / torch.einsum("...nd,...nd->...n", q, k_cumsum.type_as(q))

    with cuda_context():
        if autocast_enabled:
            q, k, v = (t.float() for t in (q, k, v))

        out = causal_dot_product_fn(q, k, v)

    out = torch.einsum("...nd,...n->...nd", out, D_inv)
    return out


# inefficient causal linear attention, without cuda code, for reader's reference
# not being used
def causal_linear_attention_noncuda(q, k, v, chunk_size=128, eps=1e-6):
    last_k_cumsum = 0
    last_context_cumsum = 0
    outs = []

    for q, k, v in zip(*(t.chunk(chunk_size, dim=-2) for t in (q, k, v))):
        k_cumsum = last_k_cumsum + k.cumsum(dim=-2)

        D_inv = 1.0 / torch.einsum("...nd,...nd->...n", q, k_cumsum.type_as(q) + eps)
        context = torch.einsum("...nd,...ne->...nde", k, v)
        context_cumsum = last_context_cumsum + context.cumsum(dim=-3)
        out = torch.einsum("...nde,...nd,...n->...ne", context_cumsum, q, D_inv)

        last_k_cumsum = k_cumsum[:, :, -1:]
        last_context_cumsum = context_cumsum[:, :, -1:]
        outs.append(out)

    return torch.cat(outs, dim=-2)


class SCPerformerSelfAttention(nn.Module):
    def __init__(self, config: dict | SCPerformerConfig | None = None, **kwargs):
        super().__init__()

        if isinstance(config, dict):
            config = SCPerformerConfig(**config)
        else:
            config = config or SCPerformerConfig()

        # kwargs take precedence over the default values that might be stored in the config object
        for k, v in kwargs.items():
            assert hasattr(config, k), f"'{k}' is an invalid config parameter"
            setattr(config, k, v)

        # assert self.num_attention_heads and self.hidden_size, "num_attention_heads and hidden_size must be non-None"
        assert (
            config.hidden_size % config.num_attention_heads == 0
        ), "num_attention_heads must divide hidden_size evenly"
        assert (
            config.hidden_size > config.num_attention_heads
        ), "Number of dimensions per head must be greater than 1"

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = config.num_attention_heads * self.attention_head_size
        self.num_attention_heads = config.num_attention_heads
        self.feature_redraw_interval = config.feature_redraw_interval
        calls_since_last_redraw = torch.tensor(0, requires_grad=False)
        self.register_buffer(
            "calls_since_last_redraw", calls_since_last_redraw, persistent=False
        )
        self.feature_generation_algorithm: FeatureGenerationAlgorithm = resolve_enum(
            FeatureGenerationAlgorithm, config.feature_generation_algorithm
        )

        if self.feature_generation_algorithm == FeatureGenerationAlgorithm.auto:
            self.feature_generation_algorithm = FeatureGenerationAlgorithm.qr

        self.num_features = (
            config.num_random_features
            if config.num_random_features is not None
            else int(self.attention_head_size * math.log(self.attention_head_size))
        )
        self.create_projection = partial(
            self.generate_projection_matrix,
            head_size=self.num_attention_heads,
            num_rows=self.num_features,
            num_columns=self.attention_head_size,
            feature_generation_algorithm=self.feature_generation_algorithm,
        )

        if isinstance(config.kernel_type, Callable):
            self.kernel_fn = config.kernel_type  # Allow for custom kernel types
        else:
            self.kernel_type = resolve_enum(PerformerKernel, config.kernel_type)
            self.kernel_fn = KERNEL_CALLABLES[self.kernel_type]

        self.kernel_epsilon = config.kernel_epsilon
        self.normalize_output = config.normalize_output
        self.causal = config.causal
        self.is_decoder = config.is_decoder
        self.use_recurrent_decoding = config.use_recurrent_decoding
        self.normalization_stabilizer = config.normalization_stabilizer

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)
        self.pruned_heads: set[int] = set()

        if self.causal:
            try:
                self.causal_linear_fn = partial(causal_linear_attention)
            except ImportError:
                print(
                    "unable to import cuda code for auto-regressive Performer. will default to the memory inefficient non-cuda version"
                )
                self.causal_linear_fn = causal_linear_attention_noncuda
        # Recurrent state, taken from 'Transformers are RNNs' Katharopoulos et al. 2020 paper
        if self.use_recurrent_decoding:
            self.s = None  # Numerator
            self.z = None  # Denominator
        else:
            assert not self.use_recurrent_decoding

    @torch.no_grad()
    def redraw_projection_matrix(self, hidden_states):
        batch_size = hidden_states.shape[0]
        if not hasattr(self, "projection_matrix"):
            logger.info("drawing projection matrix for the first time")
            projections = self.create_projection(batch_size=batch_size).type_as(
                hidden_states
            )
            self.register_buffer("projection_matrix", projections, persistent=False)

        elif batch_size != self.projection_matrix.shape[0]:
            logger.info(
                "redrawing projection matrix at %d steps because of batch size mismatch",
                self.calls_since_last_redraw,
            )
            projections = self.create_projection(batch_size=batch_size).type_as(
                hidden_states
            )
            self.projection_matrix = projections.type_as(self.projection_matrix)
            self.calls_since_last_redraw.zero_()
            del projections

        elif (
            exists(self.feature_redraw_interval)
            and self.calls_since_last_redraw >= self.feature_redraw_interval
            and self.training
        ):
            logger.info(
                "redrawing projection matrix at %d steps", self.calls_since_last_redraw
            )
            projections = self.create_projection(batch_size=batch_size).type_as(
                hidden_states
            )
            self.projection_matrix.copy_(projections)
            self.calls_since_last_redraw.zero_()
            del projections
        else:
            self.calls_since_last_redraw += 1

    def generate_projection_matrix(
        self, batch_size, head_size, num_rows, num_columns, feature_generation_algorithm
    ):
        if feature_generation_algorithm == FeatureGenerationAlgorithm.guassian:
            output_tensor = torch.randn(num_rows, num_columns)

        elif feature_generation_algorithm == FeatureGenerationAlgorithm.qr:
            output_tensor = self.gaussian_orthogonal_random_matrix(
                num_rows, num_columns
            )

        elif feature_generation_algorithm == FeatureGenerationAlgorithm.kacs:
            # throw not implemented error
            raise NotImplementedError("Kacs algorithm is not implemented yet.")
        repeat_projection_matrix = repeat(
            output_tensor, "j d -> b h j d", b=batch_size, h=head_size
        )
        return repeat_projection_matrix

    def orthogonal_matrix_chunk(self, cols):
        unstructured_block = torch.randn((cols, cols))
        q, r = torch.linalg.qr(unstructured_block, "reduced")
        return q.t()

    def gaussian_orthogonal_random_matrix(
        self, nb_rows, nb_columns, regularize_feature_norms=True
    ):
        nb_full_blocks = int(nb_rows / nb_columns)
        block_list = []

        for _ in range(nb_full_blocks):
            q = self.orthogonal_matrix_chunk(nb_columns)
            block_list.append(q)

        remaining_rows = nb_rows - nb_full_blocks * nb_columns

        if remaining_rows > 0:
            q = self.orthogonal_matrix_chunk(nb_columns)
            block_list.append(q[:remaining_rows])

        final_matrix = torch.cat(block_list)

        if not regularize_feature_norms:
            multiplier = torch.randn((nb_rows, nb_columns)).norm(dim=1)
        else:
            multiplier = math.sqrt(float(nb_columns)) * torch.ones((nb_rows,))

        return torch.diag(multiplier) @ final_matrix

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
        self.redraw_projection_matrix(hidden_states)

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

        # Get the transformed values of Q and K

        if (
            self.kernel_type == PerformerKernel.exp
            or self.kernel_type == PerformerKernel.cosh
        ):
            create_kernel = partial(
                softmax_kernel,
                projection_matrix=self.projection_matrix,
                kernel_fn=self.kernel_fn,
                kernel_epsilon=self.kernel_epsilon,
                normalize_output=self.normalize_output,
            )
            q = create_kernel(query_layer, is_query=True)
            k = create_kernel(key_layer, is_query=False)
        else:
            create_kernel = partial(
                generalized_kernel,
                projection_matrix=self.projection_matrix,
                kernel_fn=self.kernel_fn,
                kernel_epsilon=self.kernel_epsilon,
                normalize_output=self.normalize_output,
            )
            q, k = map(create_kernel, (query_layer, key_layer))

        mask = attention_mask[:, None, :, None]
        value_layer = value_layer * mask
        attention_output = linear_attention(q, k, value_layer)
        return self._finalize_attention_output(attention_output, head_mask)
        # return self.compute_attention_with_projected_queries_and_keys(q, k, value_layer, mask=attention_mask, head_mask=head_mask, out_layer=self.out)

    def compute_attention_with_projected_queries_and_keys(
        self, q_prime, k_prime, v, mask=None, head_mask=None, out_layer=None
    ):
        """
        Computes the attention output given Q' and K' from the above get_projected_queries_and_keys method.

        Parameters
        ----------
            q_prime: torch.tensor(bs, seq_length, num_features)
            k_prime: torch.tensor(bs, seq_length, num_features)
            v: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)

        Returns
        -------
            V': torch.tensor(bs, seq_length, dim).

        """
        # Apply the padding mask to K'. Also applying it to Q' would be redundant.
        k_prime_t = k_prime.transpose(-2, -1)
        output = self._numerator_for_projected_queries_and_keys(
            q_prime, k_prime_t, v, mask
        )

        if self.normalize_output:
            output /= self._denominator_for_projected_queries_and_keys(
                q_prime, k_prime_t
            )

        return self._finalize_attention_output(output, head_mask)

    def _numerator_for_projected_queries_and_keys(
        self, q_prime, k_prime_t, v, mask=None
    ):
        mask = mask[:, None, :, None]
        v = v * mask
        # Noncausal
        if not self.causal:
            return q_prime @ (k_prime_t @ v)

        # Causal, during training
        if not self.use_recurrent_decoding:
            return self.causal_numerator_fn(q_prime, k_prime_t, v)

        # Causal, at inference time- recurrent autoregressive decoding
        s_delta = k_prime_t @ v
        self.s = s_delta if self.s is None else self.s + s_delta

        return q_prime @ self.s

    def _denominator_for_projected_queries_and_keys(self, q_prime, k_prime_t):
        # Noncausal
        if not self.causal:
            denom = q_prime @ k_prime_t.sum(dim=-1, keepdim=True)  # Sum over positions

        # Causal, during training
        elif not self.use_recurrent_decoding:
            prefix_sums = k_prime_t.cumsum(dim=-1)  # Cumsum over positions
            denom = torch.einsum("bhlm,bhml->bhl", q_prime, prefix_sums)
            denom.unsqueeze_(-1)

        # Causal, at inference time- recurrent autoregressive decoding
        else:
            self.z = (
                k_prime_t if self.z is None else self.z + k_prime_t
            )  # Incrementally sum over positions
            denom = q_prime @ self.z

        # Avoid dividing by very small numbers
        return denom + 2 * self.normalization_stabilizer * (
            torch.abs(denom) <= self.normalization_stabilizer
        )

    def _finalize_attention_output(
        self, context, head_mask=None, att_map_to_output=None
    ):
        def unshape(x):
            return x.transpose(1, 2).reshape(x.shape[0], -1, x.shape[1] * x.shape[-1])

        # Mask heads if we want to
        if head_mask is not None:
            context *= head_mask

        context = unshape(context)  # (bs, q_length, dim)
        context = self.out(context)  # (bs, q_length, dim)

        if att_map_to_output:
            return context, att_map_to_output
        else:
            return (context,)

    def reset_recurrent_state(self):
        """Resets the recurrent state kept by the model when use_recurrent_decoding == True."""
        self.s = None
        self.z = None

    def prune_heads(self, heads):
        if len(heads) == 0:
            return

        attention_head_size = self.hidden_size // self.num_attention_heads
        heads, index = find_pruneable_heads_and_indices(
            heads, self.num_attention_heads, attention_head_size, self.pruned_heads
        )

        # Update hyper params
        self.num_attention_heads -= len(heads)
        self.hidden_size = attention_head_size * self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)


class SCPerformerAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = SCPerformerSelfAttention(config)
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


class SCPerformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = SCPerformerAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(
                    f"{self} should be used as a decoder model if cross attention is added"
                )
            self.crossattention = SCPerformerAttention(config)
        self.intermediate = SCIntermediate(config)
        self.output = SCOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_value: tuple[tuple[torch.FloatTensor]] | None = None,
        output_attentions: bool | None = False,
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, ...]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[
                1:
            ]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = (
                past_key_value[-2:] if past_key_value is not None else None
            )
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = (
                outputs + cross_attention_outputs[1:-1]
            )  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class SCPerformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [SCPerformerLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_values: tuple[tuple[torch.FloatTensor]] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = False,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        all_hidden_states: tuple | None = () if output_hidden_states else None
        all_self_attentions: tuple | None = () if output_attentions else None
        all_cross_attentions: tuple | None = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache: tuple | None = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                assert all_hidden_states is not None
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                assert next_decoder_cache is not None
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                assert all_self_attentions is not None
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    assert all_cross_attentions is not None
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            assert all_hidden_states is not None
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class SCPerformerPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class SCPerformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple
    interface for downloading and loading pretrained models.
    """

    config_class = SCPerformerConfig
    base_model_prefix = "scperformer"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, SCPerformerEncoder):
            module.gradient_checkpointing = value


class SCPerformerModel(SCPerformerPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as
    a decoder, in which case a layer of cross-attention is added between the
    self-attention layers, following the architecture described in [Attention
    is all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam
    Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    `is_decoder` argument of the configuration set to `True`. To be used
    in a Seq2Seq model, the model needs to initialized with both
    `is_decoder` argument and `add_cross_attention` set to `True`; an
    `encoder_hidden_states` is then expected as an input to the forward
    pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = SCEmbeddingsLayer(config)
        self.encoder = SCPerformerEncoder(config)

        self.pooler = SCPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    # def get_input_embeddings(self):
    #    return [self.embeddings.gene_embeddings, self.embeddings.expression_embeddings]

    # def set_input_embeddings(self, value):
    #    self.embeddings.gene_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model.

        heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPoolingAndCrossAttentions:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            # self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, num_fields, seq_length = input_shape

        device = None
        if input_ids is not None:
            device = input_ids.device
        elif inputs_embeds is not None:
            device = inputs_embeds.device

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # extended_attention_mask: torch.Tensor = attention_mask

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_attention_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_attention_heads] or [num_hidden_layers x num_attention_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_attention_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            # inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class SCPerformerForMaskedLM(SCPerformerPreTrainedModel):
    """
    Performer model with masked language modeling head.

    Attributes
    ----------
        config (:obj:`PretrainedConfig`): Model configuration class with all the parameters of the model.

    """

    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config: SCPerformerConfig):
        """
        Initializes the model.

        Args:
        ----
            config (:obj:`PretrainedConfig`): Model configuration class with all the parameters of the model.

        """
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `PerformerForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.scperformer = SCPerformerModel(config, add_pooling_layer=False)
        self.cls = SCOnlyMLMHead(config)
        self.post_init()

        # Initialize weights and apply final processing
        if self.config.checkpoint:
            logger.info("Loading model from checkpoint " + str(self.config.checkpoint))
            model_dict = prepare_model_dict_from_checkpoint(self.config.checkpoint)
            key_report = self.load_state_dict(model_dict, strict=False)
            logger.info(f"Loading complete. {len(model_dict)} layers in ckpt.")
            logger.info(f"Unexpected keys: {key_report.unexpected_keys}")
            logger.info(f"Missing keys: {key_report.missing_keys}")

    def get_output_embeddings(self):
        logger.warning(
            "Tie weights not supported for this model. This is used for tying weights. If you need to use tie weights fix it"
        )
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        logger.warning(
            "Tie weights not supported for this model. This is used for tying weights. If you need to use tie weights fix it"
        )
        self.cls.predictions.decoder = new_embeddings

    def tie_weights(self):
        logger.warning("Tie weights not supported for this model")
        return

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> MaskedLMOutputWithEmbeddings:
        """
        Forward pass on the model.

        Args:
        ----
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_fields, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_fields, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are NOT MASKED,
                - 0 for tokens that are MASKED.
            head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_hidden_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_fields, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert :obj:`input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_fields, sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                if the model is configured as a decoder.
            encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_fields, sequence_length)`, `optional`):
                Mask to avoid performing attention on the padding token indices of the encoder input. This mask
                is used in the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are NOT MASKED,
                - 0 for tokens that are MASKED.
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_fields, sequence_length)`, `optional`):
                Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
                config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored

        """
        # You can do a for loop over the fields but it's not efficient
        outputs = self.scperformer(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        cls_embeddings = (
            outputs.pooler_output
            if outputs.pooler_output is not None
            else outputs.last_hidden_state[:, 0, :]
        )
        mvc_query_embeddings = {}
        mvc_field_names = {
            decoder_name.split("_")[0]
            for decoder_name in self.cls.predictions.decoder.field_decoders.keys()
            if "mvc" in decoder_name
        }
        input_fields = [field for field in self.config.fields if field.is_input]
        for i, field in enumerate(input_fields):
            if field.field_name in mvc_field_names:
                embeds = self.scbert.embeddings.calculate_field_embedding(
                    input_ids, i, field
                )
                mvc_query_embeddings[field.field_name] = embeds
        if len(mvc_query_embeddings) == 0:
            field_logits = self.cls(outputs.last_hidden_state)
        else:
            field_logits = self.cls(
                outputs.last_hidden_state, cls_embeddings, mvc_query_embeddings
            )
        return MaskedLMOutputWithEmbeddings(
            logits=field_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            embeddings=cls_embeddings,
        )


class SCPerformerForSequenceClassification(SCPerformerPreTrainedModel):
    """
    Performer model for sequence classification.

    Attributes
    ----------
        config (:obj:`PretrainedConfig`): Model configuration class with all the parameters of the model.

    """

    def __init__(self, config: SCPerformerConfig):
        """
        Initializes the model.

        Args:
        ----
            config (:obj:`PretrainedConfig`): Model configuration class with all the parameters of the model.

        """
        super().__init__(config)
        self.config = config

        self.scperformer = SCPerformerModel(config)

        self.dropout = nn.Dropout(config.classifier_dropout)
        self.label_column = self.config.label_columns[0]
        self.classifier = nn.Linear(config.hidden_size, self.label_column.output_size)

        # Initialize weights and apply final processing
        self.post_init()

        if self.config.checkpoint:
            logger.info("Loading model from checkpoint " + str(self.config.checkpoint))
            model_dict = prepare_model_dict_from_checkpoint(self.config.checkpoint)
            key_report = self.load_state_dict(model_dict, strict=False)
            logger.info(f"Loading complete. {len(model_dict)} layers in ckpt.")
            logger.info(f"Unexpected keys: {key_report.unexpected_keys}")
            logger.info(f"Missing keys: {key_report.missing_keys}")

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> SequenceClassifierOutputWithEmbeddings:
        """
        Forward pass on the model.

        Args:
        ----
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_fields, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_fields, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are NOT MASKED,
                - 0 for tokens that are MASKED.
            head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_hidden_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_fields, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert :obj:`input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_fields, sequence_length)`, `optional`):
                Labels for computing the sequence classification/regression loss. Indices should be in ``[0, ...,
                config.output_size - 1]``.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for more detail.

        """
        outputs = self.scperformer(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooler_output = self.dropout(outputs.pooler_output)
        logits = {self.label_column.label_column_name: self.classifier(pooler_output)}
        return SequenceClassifierOutputWithEmbeddings(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            embeddings=outputs.pooler_output,
        )


class SCPerformerForSequenceLabeling(SCPerformerPreTrainedModel):
    """
    Performer model with sequence labeling head.

    Attributes
    ----------
        config (:obj:`PretrainedConfig`): Model configuration class with all the parameters of the model.

    """

    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config: SCPerformerConfig):
        """
        Initializes the model.

        Args:
        ----
            config (:obj:`PretrainedConfig`): Model configuration class with all the parameters of the model.

        """
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `PerformerForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.scperformer = SCPerformerModel(config, add_pooling_layer=False)
        self.cls = SCSequenceLabelingHead(config)

        # Initialize weights and apply final processing
        self.post_init()

        if self.config.checkpoint:
            logger.info("Loading model from checkpoint " + str(self.config.checkpoint))
            model_dict = prepare_model_dict_from_checkpoint(
                self.config.checkpoint, self.base_model_prefix
            )
            key_report = self.load_state_dict(model_dict, strict=False)
            logger.info(f"Loading complete. {len(model_dict)} layers in ckpt.")
            logger.info(f"Unexpected keys: {key_report.unexpected_keys}")
            logger.info(f"Missing keys: {key_report.missing_keys}")

    def get_output_embeddings(self):
        logger.warning(
            "Tie weights not supported for this model. This is used for tying weights. If you need to use tie weights fix it"
        )
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        logger.warning(
            "Tie weights not supported for this model. This is used for tying weights. If you need to use tie weights fix it"
        )
        self.cls.predictions.decoder = new_embeddings

    def tie_weights(self):
        logger.warning("Tie weights not supported for this model")
        return

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> TokenClassifierOutput:
        """
        Forward pass on the model.

        Args:
        ----
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_fields, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_fields, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are NOT MASKED,
                - 0 for tokens that are MASKED.
            head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_hidden_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_fields, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert :obj:`input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_fields, sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                if the model is configured as a decoder.
            encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_fields, sequence_length)`, `optional`):
                Mask to avoid performing attention on the padding token indices of the encoder input. This mask
                is used in the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are NOT MASKED,
                - 0 for tokens that are MASKED.
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_fields, sequence_length)`, `optional`):
                Labels for computing the sequence labeling. Indices should be in ``[-100, 0, ...,
                config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored

        """
        # You can do a for loop over the fields but it's not efficient
        outputs = self.scperformer(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        field_logits = self.cls(outputs.last_hidden_state)

        return TokenClassifierOutput(
            logits=field_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class SCPerformerForMultiTaskModeling(SCPerformerPreTrainedModel):
    """
    Performer model with masked language modeling head and sequence classification tasks.

    Attributes
    ----------
        config (:obj:`PretrainedConfig`): Model configuration class with all the parameters of the model.

    """

    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config: SCPerformerConfig):
        """
        Initializes the model.

        Args:
        ----
            config (:obj:`PretrainedConfig`): Model configuration class with all the parameters of the model.

        """
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `PerformerForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.dropout = nn.Dropout(config.classifier_dropout)
        self.scperformer = SCPerformerModel(config)
        self.cls = SCMultiTaskHead(config)
        # Initialize weights and apply final processing
        self.post_init()
        if self.config.checkpoint:
            logger.info("Loading model from checkpoint " + str(self.config.checkpoint))
            model_dict = prepare_model_dict_from_checkpoint(self.config.checkpoint)
            key_report = self.load_state_dict(model_dict, strict=False)
            logger.info(f"Loading complete. {len(model_dict)} layers in ckpt.")
            logger.info(f"Unexpected keys: {key_report.unexpected_keys}")
            logger.info(f"Missing keys: {key_report.missing_keys}")

    def get_output_embeddings(self):
        logger.warning(
            "Tie weights not supported for this model. This is used for tying weights. If you need to use tie weights fix it"
        )
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        logger.warning(
            "Tie weights not supported for this model. This is used for tying weights. If you need to use tie weights fix it"
        )
        self.cls.predictions.decoder = new_embeddings

    def tie_weights(self):
        logger.warning("Tie weights not supported for this model")
        return

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> SequenceClassifierOutputWithEmbeddings:
        """
        Forward pass on the model.

        Args:
        ----
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_fields, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_fields, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are NOT MASKED,
                - 0 for tokens that are MASKED.
            head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_hidden_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_fields, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert :obj:`input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_fields, sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                if the model is configured as a decoder.
            encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_fields, sequence_length)`, `optional`):
                Mask to avoid performing attention on the padding token indices of the encoder input. This mask
                is used in the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are NOT MASKED,
                - 0 for tokens that are MASKED.
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_fields, sequence_length)`, `optional`):
                Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
                config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored

        """
        # You can do a for loop over the fields but it's not efficient
        outputs = self.scperformer(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        cls_embeddings = (
            outputs.pooler_output
            if outputs.pooler_output is not None
            else outputs.last_hidden_state[:, 0, :]
        )
        pooler_output = self.dropout(outputs.pooler_output)

        mvc_query_embeddings = {}
        mvc_field_names = {
            decoder_name.split("_")[0]
            for decoder_name in self.cls.predictions.predictions.decoder.field_decoders.keys()
            if "mvc" in decoder_name
        }
        input_fields = [field for field in self.config.fields if field.is_input]
        for i, field in enumerate(input_fields):
            if field.field_name in mvc_field_names:
                embeds = self.scbert.embeddings.calculate_field_embedding(
                    input_ids, i, field
                )
                mvc_query_embeddings[field.field_name] = embeds.squeeze(1)
        if len(mvc_query_embeddings) == 0:
            logits = self.cls(outputs.last_hidden_state, cls_embeddings)
        else:
            logits = self.cls(
                outputs.last_hidden_state, cls_embeddings, mvc_query_embeddings
            )

        return SequenceClassifierOutputWithEmbeddings(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            embeddings=outputs.pooler_output,
        )
