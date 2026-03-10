import warnings

from bmfm_targets.training.modules.base import BaseTrainingModule


class MLMTrainingModule(BaseTrainingModule):
    """
    DEPRECATED: Use MultiTaskTrainingModule instead.

    This module is kept for backwards compatibility only.
    All masked language modeling tasks should now use MultiTaskTrainingModule.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "MLMTrainingModule is deprecated. "
            "Use MultiTaskTrainingModule instead. "
            "This class will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
