"""PyTorch Lightning Training Modules for use with the bmfm-targets package."""
import logging

from bmfm_targets.training.data_module import DataModule
from .sequence_classification import SequenceClassificationTrainingModule
from .masked_language_modeling import MLMTrainingModule
from .sequence_labeling import SequenceLabelingTrainingModule
from .multitask_modeling import MultiTaskTrainingModule

__all__ = [
    "SequenceClassificationTrainingModule",
    "MLMTrainingModule",
    "SequenceLabelingTrainingModule",
    "MultiTaskTrainingModule",
]


def get_training_module_class_for_data_module(data_module: DataModule):
    """
    Select correct training module based on the data module.

    If the data module is for MLM it requires a different training module, due to the
    collation differences for classifying vs masked language. This is controlled by the
    `mlm` parameter.

    Args:
    ----
      data_module (pl.LightningDataModule): an instantiated data module with the attr
          `mlm` defined.

    Returns:
    -------
       MLMTrainingModule | SequenceClassificationTrainingModule : the correct
         constructor function (not an instance).
    """
    if data_module.collation_strategy == "multitask":
        return MultiTaskTrainingModule
    elif data_module.collation_strategy == "language_modeling":
        if not data_module.mlm:
            logging.warning(
                "Requested language_modeling without `mlm`! "
                "Model can be used for predict only."
            )
        return MLMTrainingModule
    elif (
        data_module.collation_strategy == "sequence_classification_with_lm"
        and data_module.mlm
    ):
        ## TODO here we need to return another training module; will modify in the next PR
        return MLMTrainingModule
    elif data_module.collation_strategy in ["sequence_classification", "multilabel"]:
        return SequenceClassificationTrainingModule
    elif data_module.collation_strategy == "sequence_labeling":
        return SequenceLabelingTrainingModule
    else:
        raise ValueError(f"Unknown model type {data_module.collation_strategy}")
