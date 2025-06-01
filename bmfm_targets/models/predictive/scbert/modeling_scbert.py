"""SCBert model was adapted from BertModel in transformers library."""

import math

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.configuration_utils import PretrainedConfig
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

from bmfm_targets.config import SCBertConfig
from bmfm_targets.models.model_utils import (
    MaskedLMOutputWithEmbeddings,
    SequenceClassifierOutputWithEmbeddings,
)
from bmfm_targets.models.predictive.attentions import attention_factory
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

logger = logging.get_logger(__name__)


class SCBertSelfAttention(nn.Module):
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

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
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
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in SCBertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs: torch.Tensor = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer: torch.Tensor = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        if self.is_decoder:
            outputs = outputs + (past_key_value,)  # type: ignore
        return outputs


class SCBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = SCBertSelfAttention(config)
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


class SCBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = attention_factory(
            getattr(config, "attention", None), config, SCBertAttention
        )
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(
                    f"{self} should be used as a decoder model if cross attention is added"
                )
            self.crossattention = SCBertAttention(config)
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
    ) -> tuple[torch.Tensor]:
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


class SCBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [SCBertLayer(config) for _ in range(config.num_hidden_layers)]
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


class SCBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple
    interface for downloading and loading pretrained models.
    """

    config_class = SCBertConfig
    base_model_prefix = "scbert"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights. For the embedding layers - the initialization is in layers.py."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, SCBertEncoder):
            module.gradient_checkpointing = value


class SCBertModel(SCBertPreTrainedModel):
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

    def __init__(self, config: SCBertConfig, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings = SCEmbeddingsLayer(config)
        self.encoder = SCBertEncoder(config)

        self.pooler = SCPooler(config) if add_pooling_layer else None
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
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:
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
            batch_size, num_fields, seq_length = input_shape
            # self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

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
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
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
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs.last_hidden_state
        pooler_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooler_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class SCBertForMaskedLM(SCBertPreTrainedModel):
    """
    Bert model with masked language modeling head.

    Attributes
    ----------
        config (:obj:`PretrainedConfig`): Model configuration class with all the parameters of the model.

    """

    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config: SCBertConfig):
        """
        Initializes the model.

        Args:
        ----
            config (:obj:`PretrainedConfig`): Model configuration class with all the parameters of the model.

        """
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
        self.scbert = SCBertModel(config, add_pooling_layer=False)
        self.cls = SCOnlyMLMHead(config)
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
        outputs = self.scbert(
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


class SCBertForSequenceClassification(SCBertPreTrainedModel):
    """
    Bert model for sequence classification.

    Attributes
    ----------
        config (:obj:`PretrainedConfig`): Model configuration class with all the parameters of the model.

    """

    def __init__(self, config: SCBertConfig):
        """
        Initializes the model.

        Args:
        ----
            config (:obj:`PretrainedConfig`): Model configuration class with all the parameters of the model.

        """
        super().__init__(config)
        self.config = config

        self.scbert = SCBertModel(config)

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
        outputs = self.scbert(
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


class SCBertForSequenceLabeling(SCBertPreTrainedModel):
    """
    Bert model with sequence labeling head.

    Attributes
    ----------
        config (:obj:`PretrainedConfig`): Model configuration class with all the parameters of the model.

    """

    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config: SCBertConfig):
        """
        Initializes the model.

        Args:
        ----
            config (:obj:`PretrainedConfig`): Model configuration class with all the parameters of the model.

        """
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.scbert = SCBertModel(config, add_pooling_layer=False)
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
        outputs = self.scbert(
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


class SCBertForMultiTaskModeling(SCBertPreTrainedModel):
    """
    Bert model with masked language modeling head and sequence classification tasks.

    Attributes
    ----------
        config (:obj:`PretrainedConfig`): Model configuration class with all the parameters of the model.

    """

    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config: SCBertConfig):
        """
        Initializes the model.

        Args:
        ----
            config (:obj:`PretrainedConfig`): Model configuration class with all the parameters of the model.

        """
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.dropout = nn.Dropout(config.classifier_dropout)
        self.scbert = SCBertModel(config)
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
        outputs = self.scbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooler_output = self.dropout(outputs.pooler_output)

        cls_embeddings = (
            outputs.pooler_output
            if outputs.pooler_output is not None
            else outputs.last_hidden_state[:, 0, :]
        )

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
                mvc_query_embeddings[field.field_name] = embeds

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
