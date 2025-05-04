from dataclasses import dataclass

import torch
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.auto import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
)

from bmfm_targets import config


def register_configs_and_models():
    from bmfm_targets.models.predictive import (
        scbert,
        scmodernbert,
        scnystromformer,
        scperformer,
    )

    _config_maps = [
        (
            config.SCBertConfig,
            scbert.SCBertForMaskedLM,
            scbert.SCBertForSequenceClassification,
        ),
        (
            config.SCPerformerConfig,
            scperformer.SCPerformerForMaskedLM,
            scperformer.SCPerformerForSequenceClassification,
        ),
        (
            config.SCNystromformerConfig,
            scnystromformer.SCNystromformerForMaskedLM,
            scnystromformer.SCNystromformerForSequenceClassification,
        ),
        (
            config.SCModernBertConfig,
            scmodernbert.SCModernBertForMaskedLM,
            scmodernbert.SCModernBertForSequenceClassification,
        ),
    ]

    for config_class, lm_class, seqcls_class in _config_maps:
        AutoConfig.register(config_class.model_type, config_class)
        AutoModelForMaskedLM.register(config_class, lm_class)
        AutoModelForSequenceClassification.register(config_class, seqcls_class)


def get_model_from_config(
    model_config: config.SCModelConfigBase, modeling_strategy: str
):
    from bmfm_targets.models.predictive import (
        scbert,
        scmodernbert,
        scnystromformer,
        scperformer,
    )

    # scbert
    if isinstance(model_config, config.SCBertConfig):
        if modeling_strategy == "mlm":
            return scbert.SCBertForMaskedLM(model_config)
        if modeling_strategy == "sequence_classification":
            return scbert.SCBertForSequenceClassification(model_config)
        if modeling_strategy == "sequence_labeling":
            return scbert.SCBertForSequenceLabeling(model_config)
        if modeling_strategy == "multitask":
            return scbert.SCBertForMultiTaskModeling(model_config)
    # scperformer
    if isinstance(model_config, config.SCPerformerConfig):
        if modeling_strategy == "mlm":
            return scperformer.SCPerformerForMaskedLM(model_config)
        if modeling_strategy == "sequence_classification":
            return scperformer.SCPerformerForSequenceClassification(model_config)
        if modeling_strategy == "sequence_labeling":
            return scperformer.SCPerformerForSequenceLabeling(model_config)
        if modeling_strategy == "multitask":
            return scperformer.SCPerformerForMultiTaskModeling(model_config)
    # scnystromformer
    if isinstance(model_config, config.SCNystromformerConfig):
        if modeling_strategy == "mlm":
            return scnystromformer.SCNystromformerForMaskedLM(model_config)
        if modeling_strategy == "sequence_classification":
            return scnystromformer.SCNystromformerForSequenceClassification(
                model_config
            )
        if modeling_strategy == "sequence_labeling":
            return scnystromformer.SCNystromformerForSequenceLabeling(model_config)
        if modeling_strategy == "multitask":
            return scnystromformer.SCNystromformerForMultiTaskModeling(model_config)
    # SCModernBert
    if isinstance(model_config, config.SCModernBertConfig):
        if modeling_strategy == "mlm":
            return scmodernbert.SCModernBertForMaskedLM(model_config)
        if modeling_strategy == "sequence_classification":
            return scmodernbert.SCModernBertForSequenceClassification(model_config)
        if modeling_strategy == "sequence_labeling":
            return scmodernbert.SCModernBertForSequenceLabeling(model_config)
        if modeling_strategy == "multitask":
            return scmodernbert.SCModernBertForMultiTaskModeling(model_config)

    raise ValueError(f"Unknown model_config type {type(model_config)}")


def get_base_model_from_config(model_config: config.SCModelConfigBase):
    from bmfm_targets.models.predictive import (
        scbert,
        scmodernbert,
        scnystromformer,
        scperformer,
    )

    if isinstance(model_config, config.SCBertConfig):
        base_model = scbert.modeling_scbert.SCBertModel(
            model_config, add_pooling_layer=True
        )
    elif isinstance(model_config, config.SCPerformerConfig):
        base_model = scperformer.modeling_scperformer.SCPerformerModel(
            model_config, add_pooling_layer=True
        )
    elif isinstance(model_config, config.SCNystromformerConfig):
        base_model = scnystromformer.modeling_scnystromformer.SCNystromformerModel(
            model_config, add_pooling_layer=True
        )
    elif isinstance(model_config, config.SCModernBertConfig):
        base_model = scmodernbert.modeling_scmodernbert.SCModernBertModel(model_config)

    return base_model


@dataclass
class SequenceClassifierOutputWithEmbeddings(SequenceClassifierOutput):
    embeddings: torch.FloatTensor | None = None


@dataclass
class MaskedLMOutputWithEmbeddings(SequenceClassifierOutput):
    embeddings: torch.FloatTensor | None = None


def instantiate_classification_model(model_config, loss_tasks):
    from bmfm_targets.models.predictive import MultiTaskClassifier

    if loss_tasks:
        if model_config.checkpoint:
            return MultiTaskClassifier.from_ckpt(
                model_config.checkpoint,
                loss_tasks=loss_tasks,
                model_config=model_config,
            )
        else:
            base_model = get_base_model_from_config(model_config)
            return MultiTaskClassifier(base_model, loss_tasks)

    return get_model_from_config(
        model_config, modeling_strategy="sequence_classification"
    )
