from functools import partial

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.utils import logging

from bmfm_targets.models.common.hf_registration import (
    register_autoconfig,
    register_automodel_for_maskedLM,
    register_automodel_for_sequence_classification,
)
from bmfm_targets.models.common.llama import LlamaEncoder, LlamaParams
from bmfm_targets.models.common.mixins import CheckpointMixin, InitWeightsMixin
from bmfm_targets.models.model_utils import (
    MaskedLMOutputWithEmbeddings,
    SequenceClassifierOutputWithEmbeddings,
)
from bmfm_targets.models.predictive.layers import (
    SCEmbeddingsLayer,
    SCMultiTaskHead,
    SCOnlyMLMHead,
    SCPooler,
)

from .config import (
    LlamaConfig,
    LlamaForMaskedLMConfig,
    LlamaForMultiTaskConfig,
    LlamaForSequenceClassification,
    LlamaWithPoolingHeadConfig,
)

logger = logging.get_logger(__name__)


class LlamaEncoderWithInputEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        pars = LlamaParams(
            n_layer=config.num_hidden_layers,
            n_embd=config.hidden_size,
            n_head=config.num_attention_heads,
            n_aux_tokens=config.num_memory_tokens,
            attention=config.attention,
        )

        embed_constructor = partial(SCEmbeddingsLayer, config)
        self.encoder = LlamaEncoder(embed_constructor, pars)

    def _init_weights(self, module):
        ...

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
    ):
        h: BaseModelOutputWithPoolingAndCrossAttentions = self.encoder(
            input_ids,
            attention_mask,
            inputs_embeds,
            head_mask=head_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        return h


class LlamaEncoderWithPoolingHead(LlamaEncoderWithInputEmbedding):
    def __init__(self, config: LlamaWithPoolingHeadConfig):
        super().__init__(config)
        self.pooler = SCPooler(config)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
    ):
        encoder_output: BaseModelOutputWithPoolingAndCrossAttentions = super().forward(
            input_ids,
            attention_mask,
            inputs_embeds,
            head_mask=head_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        pooled_h = self.pooler(
            hidden_states=encoder_output.last_hidden_state,
            attention_mask=attention_mask,
        )
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=encoder_output.last_hidden_state,
            hidden_states=encoder_output.hidden_states,
            pooler_output=pooled_h,
        )


class LlamaForMaskedLMModel(nn.Module, CheckpointMixin):
    def __init__(self, config: LlamaForMaskedLMConfig):
        super().__init__()

        self.config = config
        self.core = LlamaEncoderWithInputEmbedding(config)
        self.cls = SCOnlyMLMHead(config)
        self.load_checkpoint()

    def get_base_model(self):
        return self.core

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> SequenceClassifierOutputWithEmbeddings:
        outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.core(
            input_ids, attention_mask, inputs_embeds
        )
        cls_embeddings = outputs.last_hidden_state[:, 0, :]

        mvc_query_embeddings = {}
        mvc_field_names = {
            decoder_name.split("_")[0]
            for decoder_name in self.cls.predictions.decoder.field_decoders.keys()
            if "mvc" in decoder_name
        }
        input_fields = [field for field in self.config.fields if field.is_input]
        for i, field in enumerate(input_fields):
            if field.field_name in mvc_field_names:
                embeds = self.core.encoder.embeddings.calculate_field_embedding(
                    input_ids, i, field
                )
                mvc_query_embeddings[field.field_name] = embeds

        if len(mvc_query_embeddings) == 0:
            logits = self.cls(outputs.last_hidden_state)
        else:
            logits = self.cls(
                outputs.last_hidden_state, cls_embeddings, mvc_query_embeddings
            )

        return MaskedLMOutputWithEmbeddings(
            logits=logits,
            hidden_states=outputs.last_hidden_state,
            embeddings=cls_embeddings,
        )


class LlamaForMultiTaskModel(nn.Module, CheckpointMixin):
    def __init__(self, config: LlamaForMultiTaskConfig):
        super().__init__()

        self.config = config
        self.core = LlamaEncoderWithPoolingHead(config)
        self.cls = SCMultiTaskHead(config)
        self.load_checkpoint()

    def get_base_model(self):
        return self.core

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> SequenceClassifierOutputWithEmbeddings:
        outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.core(
            input_ids, attention_mask, inputs_embeds
        )
        cls_embeddings = outputs.pooler_output

        mvc_query_embeddings = {}
        mvc_field_names = {
            decoder_name.split("_")[0]
            for decoder_name in self.cls.predictions.predictions.decoder.field_decoders.keys()
            if "mvc" in decoder_name
        }
        input_fields = [field for field in self.config.fields if field.is_input]
        for i, field in enumerate(input_fields):
            if field.field_name in mvc_field_names:
                embeds = self.core.encoder.embeddings.calculate_field_embedding(
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
            embeddings=outputs.pooler_output,
        )


class LlamaForSequenceClassificationModel(nn.Module, CheckpointMixin, InitWeightsMixin):
    def __init__(self, config: LlamaForMultiTaskConfig):
        super().__init__()

        self.config = config
        self.core = LlamaEncoderWithPoolingHead(config)
        self.load_checkpoint()

    def get_base_model(self):
        return self.core

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> SequenceClassifierOutputWithEmbeddings:
        if head_mask is not None:
            raise ValueError("LLama model does not support head mask ...")
        if output_attentions:
            raise ValueError("LLama model does not support output attentions ...")

        outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.core(
            input_ids, attention_mask, inputs_embeds
        )
        pooler_output = outputs.pooler_output

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            pooler_output=pooler_output,
        )


register_autoconfig(config_classes=[LlamaForMaskedLMConfig])
register_automodel_for_maskedLM(
    config_classes=[LlamaForMaskedLMConfig], lm_classes=[LlamaForMaskedLMModel]
)

register_automodel_for_sequence_classification(
    config_classes=[LlamaForSequenceClassification],
    seq_classes=[LlamaForSequenceClassificationModel],
)
