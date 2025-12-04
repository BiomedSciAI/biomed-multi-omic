from typing import Literal

from bmfm_targets.models.common.hf_registration import (
    register_automodel_for_maskedLM,
)
from bmfm_targets.models.common.llama import LlamaConfig


class LlamaWithPoolingHeadConfig(LlamaConfig):
    def __init__(
        self,
        classifier_pooling: Literal["cls", "mean"] = "cls",
        num_memory_tokens: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.classifier_pooling = classifier_pooling
        self.num_memory_tokens = num_memory_tokens


class LlamaForMaskedLMConfig(LlamaConfig):
    def __init__(
        self,
        hidden_act="gelu",
        layer_norm_eps: float = 1e-12,
        num_memory_tokens: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.num_memory_tokens = num_memory_tokens
        from bmfm_targets.models.predictive.llama.model import (
            LlamaForMaskedLMModel,
        )

        register_automodel_for_maskedLM(
            config_classes=[self.__class__], lm_classes=[LlamaForMaskedLMModel]
        )


class LlamaForMultiTaskConfig(LlamaWithPoolingHeadConfig):
    def __init__(
        self,
        hidden_act="gelu",
        layer_norm_eps: float = 1e-12,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps


class LlamaForSequenceClassification(LlamaWithPoolingHeadConfig):
    def __init__(
        self,
        hidden_act="gelu",
        layer_norm_eps: float = 1e-12,
        classifier_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.classifier_dropout = classifier_dropout
