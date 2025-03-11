from bmfm_targets.training.modules.base import BaseTrainingModule


class MultiTaskTrainingModule(BaseTrainingModule):
    DEFAULT_METRICS = {"accuracy", "confusion_matrix", "nonzero_confusion_matrix"}
    PERPLEXITY_LOGGING = True
    MODELING_STRATEGY = "multitask"
