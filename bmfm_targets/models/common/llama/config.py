from pydantic import TypeAdapter

from bmfm_targets.config import (
    FieldInfo,
    LabelColumnInfo,
    ModelingStrategy,
    SCModelConfigBase,
)
from bmfm_targets.models.common.hf_registration import (
    register_automodel_for_sequence_classification,
)

from .constants import AttentionKind


class LlamaConfig(SCModelConfigBase):
    model_type = "scllama"

    def __init__(
        self,
        fields: list[FieldInfo] | None = None,
        label_columns: list[LabelColumnInfo] | None = None,
        checkpoint: str | None = None,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        hidden_size: int = 768,
        hidden_dropout_prob: float = 0.0,
        attention: AttentionKind = AttentionKind.TORCH,
        pad_token_id: int = 0,
        max_position_embeddings: int = 512,
        position_embedding_type: str | None = None,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.fields = fields
        self.label_columns = label_columns
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.position_embedding_type = position_embedding_type
        self.checkpoint = checkpoint
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention = TypeAdapter(AttentionKind).validate_python(attention)

        from bmfm_targets.models.predictive.llama.model import (
            LlamaForSequenceClassificationModel,
        )

        register_automodel_for_sequence_classification(
            config_classes=[self.__class__],
            seq_classes=[LlamaForSequenceClassificationModel],
        )

    def build_model(self, strategy: ModelingStrategy):
        if strategy == ModelingStrategy.MULTITASK:
            from bmfm_targets.models.predictive.llama.model import (
                LlamaForMultiTaskModel,
            )

            return LlamaForMultiTaskModel(config=self)

        if strategy == ModelingStrategy.MLM:
            from bmfm_targets.models.predictive.llama.model import (
                LlamaForMaskedLMModel,
            )

            return LlamaForMaskedLMModel(config=self)

        if strategy == ModelingStrategy.SEQUENCE_CLASSIFICATION:
            from bmfm_targets.models.predictive.llama.model import (
                LlamaForSequenceClassificationModel,
            )

            return LlamaForSequenceClassificationModel(config=self)

        raise ValueError(f"Strategy {strategy} is not supported by LLama model.")
