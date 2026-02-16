from bmfm_targets.training.modules.base import BaseTrainingModule


class MultiTaskTrainingModule(BaseTrainingModule):
    MODELING_STRATEGY = "multitask"
