"""PyTorch Nystromformer model."""

import math

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
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

from bmfm_targets.config import SCNystromformerConfig
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

logger = logging.get_logger(__name__)

# _CHECKPOINT_FOR_DOC = "uw-madison/nystromformer-512"
# _CONFIG_FOR_DOC = "NystromformerConfig"


class SCNystromformerSelfAttention(nn.Module):
    def __init__(self, config):
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

        self.num_landmarks = config.num_landmarks
        self.seq_len = config.max_position_embeddings
        self.conv_kernel_size = config.conv_kernel_size
        self.inverse_method = getattr(config, "inverse_method", "original")
        self.inverse_n_iter = getattr(config, "inverse_n_iter", 6)

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        if self.conv_kernel_size is not None:
            self.conv = nn.Conv2d(
                in_channels=self.num_attention_heads,
                out_channels=self.num_attention_heads,
                kernel_size=(self.conv_kernel_size, 1),
                padding=(self.conv_kernel_size // 2, 0),
                bias=False,
                groups=self.num_attention_heads,
            )

    @staticmethod
    def iterative_inv(
        mat: torch.Tensor, n_iter: int = 6, inverse_method: str = "original"
    ) -> torch.Tensor:
        """
        Function to approximate Moore-Penrose inverse via the iterative method.

        Parameters
        ----------
        mat : torch.Tensor
            Tidle landmark and tidle query matrices calculate the pseudoinverse on
        n_iter : int, optional
            Number of iterations for convergence, by default 6
        inverse_method : str, optional
            Iterative method to use. Must be "original", "newton" or "chebyshev", by default "original"

        Returns
        -------
        torch.Tensor
            Pinv mat

        """

        def _lambda_max_upper(a: torch.Tensor) -> torch.Tensor:
            # See "Bounds for Elgenvalues Using Traces", Wolkowicz and Styan, (1980)
            a_abs = torch.abs(a)
            r = torch.max(torch.sum(a_abs, -1))
            c = torch.max(torch.sum(a_abs, -2))
            return min(r, c)

        def _pinv_newton(a: torch.Tensor, n_iter: int = 6) -> torch.Tensor:
            alpha = 2 / _lambda_max_upper(torch.transpose(a, -1, -2) @ a)
            ainv = alpha * torch.transpose(a, -1, -2)
            two_eye = 2 * torch.eye(a.shape[-1], device=a.device)
            for _ in range(n_iter):
                ainv = ainv @ (two_eye - a @ ainv)
            return ainv

        def _pinv_chebyshev(a: torch.Tensor, n_iter: int = 6) -> torch.Tensor:
            alpha = 2 / _lambda_max_upper(a.transpose(-1, -2) @ a)
            ainv = alpha * a.transpose(-1, -2)
            three_eye = 3 * torch.eye(a.size(-1), device=a.device)
            for _ in range(n_iter):
                a_ainv = a @ ainv
                ainv = ainv @ (three_eye - a_ainv @ (three_eye - a_ainv))
            return ainv

        def _pinv_original_nystrom(a: torch.Tensor, n_iter: int = 6) -> torch.Tensor:
            identity = torch.eye(a.size(-1), device=a.device)
            key = a
            value = 1 / torch.max(torch.sum(key, dim=-2)) * key.transpose(-1, -2)
            for _ in range(n_iter):
                key_value = torch.matmul(key, value)
                value = torch.matmul(
                    0.25 * value,
                    13 * identity
                    - torch.matmul(
                        key_value,
                        15 * identity
                        - torch.matmul(key_value, 7 * identity - key_value),
                    ),
                )
            return value

        match inverse_method:
            case "original":
                pinv = _pinv_original_nystrom(a=mat, n_iter=n_iter)
            case "newton":
                pinv = _pinv_newton(a=mat, n_iter=n_iter)
            case "chebyshev":
                pinv = _pinv_chebyshev(a=mat, n_iter=n_iter)
            case _:
                raise NotImplementedError(
                    f"{inverse_method} must be original, newton or chebyshev"
                )

        return pinv

    def transpose_for_scores(self, layer):
        new_layer_shape = layer.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        layer = layer.view(*new_layer_shape)
        return layer.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        query_layer = query_layer / math.sqrt(math.sqrt(self.attention_head_size))
        key_layer = key_layer / math.sqrt(math.sqrt(self.attention_head_size))

        if self.num_landmarks == self.seq_len:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

            if attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in NystromformerModel forward() function)
                attention_scores = attention_scores + attention_mask

            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            context_layer = torch.matmul(attention_probs, value_layer)

        else:
            q_landmarks = query_layer.reshape(
                -1,
                self.num_attention_heads,
                self.num_landmarks,
                self.seq_len // self.num_landmarks,
                self.attention_head_size,
            ).mean(dim=-2)
            k_landmarks = key_layer.reshape(
                -1,
                self.num_attention_heads,
                self.num_landmarks,
                self.seq_len // self.num_landmarks,
                self.attention_head_size,
            ).mean(dim=-2)

            kernel_1 = torch.nn.functional.softmax(
                torch.matmul(query_layer, k_landmarks.transpose(-1, -2)), dim=-1
            )
            kernel_2 = torch.nn.functional.softmax(
                torch.matmul(q_landmarks, k_landmarks.transpose(-1, -2)), dim=-1
            )

            attention_scores = torch.matmul(q_landmarks, key_layer.transpose(-1, -2))

            if attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in NystromformerModel forward() function)
                attention_scores = attention_scores + attention_mask

            kernel_3 = nn.functional.softmax(attention_scores, dim=-1)
            attention_probs = torch.matmul(
                kernel_1,
                self.iterative_inv(
                    kernel_2,
                    inverse_method=self.inverse_method,
                    n_iter=self.inverse_n_iter,
                ),
            )
            new_value_layer = torch.matmul(kernel_3, value_layer)
            context_layer = torch.matmul(attention_probs, new_value_layer)

        if self.conv_kernel_size is not None:
            context_layer += self.conv(value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs


class SCNystromformerAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = SCNystromformerSelfAttention(config)
        self.output = SCSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
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

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        self_outputs = self.self(hidden_states, attention_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class SCNystromformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = SCNystromformerAttention(config)
        self.add_cross_attention = config.add_cross_attention
        self.intermediate = SCIntermediate(config)
        self.output = SCOutput(config)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, output_attentions=output_attentions
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class SCNystromformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [SCNystromformerLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, attention_mask, output_attentions
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class SCNystromformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SCNystromformerConfig
    base_model_prefix = "scnystromformer"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear | nn.Conv2d):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
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
        if isinstance(module, SCNystromformerEncoder):
            module.gradient_checkpointing = value


class SCNystromformerModel(SCNystromformerPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = SCEmbeddingsLayer(config)
        self.encoder = SCNystromformerEncoder(config)

        self.pooler = SCPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing

        self.post_init()

    # def get_input_embeddings(self):
    #     return self.embeddings.word_embeddings

    # def set_input_embeddings(self, value):
    #     self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel.
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
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
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            ###            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        ### added num_fiellds
        batch_size, num_fields, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
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


class SCNystromformerForMaskedLM(SCNystromformerPreTrainedModel):
    """
    Nystromformer model with masked language modeling head.

    Attributes
    ----------
        config (:obj:`PretrainedConfig`): Model configuration class with all the parameters of the model.

    """

    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config: SCNystromformerConfig):
        """
        Initializes the model.

        Args:
        ----
            config (:obj:`PretrainedConfig`): Model configuration class with all the parameters of the model.

        """
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `NystromformerForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
        self.scnystromformer = SCNystromformerModel(config, add_pooling_layer=False)
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
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_fields, sequence_length)`, `optional`):
                Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
                config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored

        """
        # You can do a for loop over the fields but it's not efficient
        outputs = self.scnystromformer(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
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


class SCNystromformerClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.label_column = config.label_columns[0]
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, self.label_column.output_size)

        self.config = config

    def forward(self, x, **kwargs):
        x = self.dense(x)
        x = ACT2FN[self.config.hidden_act](x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class SCNystromformerForSequenceClassification(SCNystromformerPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.dropout = nn.Dropout(config.classifier_dropout)
        self.use_pooling_layer = add_pooling_layer

        self.scnystromformer = SCNystromformerModel(
            config, add_pooling_layer=self.use_pooling_layer
        )
        self.dropout = nn.Dropout(config.classifier_dropout)

        self.label_column = self.config.label_columns[0]
        if not self.use_pooling_layer:
            self.classifier = SCNystromformerClassificationHead(config)
        else:
            self.classifier = nn.Linear(
                config.hidden_size, self.label_column.output_size
            )

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
    ) -> tuple[torch.Tensor] | SequenceClassifierOutputWithEmbeddings:
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
        outputs = self.scnystromformer(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooler_output = (
            outputs.pooler_output
            if self.use_pooling_layer
            else outputs.last_hidden_state[:, 0, :]
        )

        pooler_output = self.dropout(pooler_output)
        logits = {self.label_column.label_column_name: self.classifier(pooler_output)}

        return SequenceClassifierOutputWithEmbeddings(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            embeddings=pooler_output,
        )


class SCNystromformerForSequenceLabeling(SCNystromformerPreTrainedModel):
    """
    Nystromformer model with sequence labeling head.

    Attributes
    ----------
        config (:obj:`PretrainedConfig`): Model configuration class with all the parameters of the model.

    """

    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config: SCNystromformerConfig):
        """
        Initializes the model.

        Args:
        ----
            config (:obj:`PretrainedConfig`): Model configuration class with all the parameters of the model.

        """
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `NystromformerForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.scnystromformer = SCNystromformerModel(config, add_pooling_layer=False)
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
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_fields, sequence_length)`, `optional`):
                Labels for computing the sequence labeling. Indices should be in ``[-100, 0, ...,
                config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored

        """
        # You can do a for loop over the fields but it's not efficient
        outputs = self.scnystromformer(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        field_logits = self.cls(outputs.last_hidden_state)

        return TokenClassifierOutput(
            logits=field_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class SCNystromformerForMultiTaskModeling(SCNystromformerPreTrainedModel):
    """
    Nystromformer model with masked language modeling head and sequence classification tasks.

    Attributes
    ----------
        config (:obj:`PretrainedConfig`): Model configuration class with all the parameters of the model.

    """

    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config: SCNystromformerConfig):
        """
        Initializes the model.

        Args:
        ----
            config (:obj:`PretrainedConfig`): Model configuration class with all the parameters of the model.

        """
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `NystromformerForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.dropout = nn.Dropout(config.classifier_dropout)
        self.scnystromformer = SCNystromformerModel(config)
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
                Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
                config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored

        """
        # You can do a for loop over the fields but it's not efficient
        outputs = self.scnystromformer(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
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
