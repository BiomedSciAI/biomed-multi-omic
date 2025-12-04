from transformers.models.auto import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
)


def register_autoconfig(config_classes: list[object]):
    for config_class in config_classes:
        AutoConfig.register(config_class.model_type, config_class)


def register_automodel_for_maskedLM(
    config_classes: list[object], lm_classes: list[object]
):
    for config_class, lm_class in zip(config_classes, lm_classes):
        AutoModelForMaskedLM.register(config_class, lm_class, exist_ok=True)


def register_automodel_for_sequence_classification(
    config_classes: list[object], seq_classes: list[object]
):
    for config_class, seq_class in zip(config_classes, seq_classes):
        AutoModelForSequenceClassification.register(
            config_class, seq_class, exist_ok=True
        )
