import logging

from bmfm_targets.training.modules.base import BaseTrainingModule

logger = logging.getLogger(__name__)


class SequenceClassificationTrainingModule(BaseTrainingModule):
    MODELING_STRATEGY = "sequence_classification"
