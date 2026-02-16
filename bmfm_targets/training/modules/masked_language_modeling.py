from bmfm_targets.training.modules.base import BaseTrainingModule


class MLMTrainingModule(BaseTrainingModule):
    MODELING_STRATEGY = "mlm"
