"""
Loss handling package for bmfm_targets.

This package provides a composition-based architecture for loss handling,
separating data extraction (DataSource) from loss computation (Objective).

Public API:
-----------
- LossTask: Main container composing DataSource + Objective
- DataSource, Objective: Abstract base classes
- FieldSource, LabelSource: Concrete data sources
- CrossEntropyObjective, MSEObjective, etc.: Concrete objectives
- get_loss_tasks: Backward compatibility function
- loss_dict_to_task: Convert old dict configs to LossTask
- calculate_losses, calculate_predictions: Helper functions
"""

# Abstract base classes
from .base import DataSource, Objective

# Backward compatibility
from .compat import loss_dict_to_task

# Concrete implementations
from .objectives import (
    BCEWithLogitsObjective,
    CrossEntropyObjective,
    FocalObjective,
    HCEObjective,
    IsZeroBCEObjective,
    IsZeroFocalObjective,
    MAEObjective,
    MSEObjective,
    TokenValueObjective,
)
from .sources import FieldSource, LabelSource, WCEDFieldSource

# Main container
from .task import LossTask, lookup_wced_output_index

# Utility functions
from .utils import (
    adapt_hce_prediction_and_labels_to_metrics_entries,
    calculate_losses,
    calculate_predictions,
    combine_partial_predictions,
    get_loss_tasks,
)

__all__ = [
    # Abstract base classes
    "DataSource",
    "Objective",
    # Data sources
    "FieldSource",
    "LabelSource",
    "WCEDFieldSource",
    "lookup_wced_output_index",
    # Objectives
    "CrossEntropyObjective",
    "FocalObjective",
    "MSEObjective",
    "MAEObjective",
    "TokenValueObjective",
    "IsZeroBCEObjective",
    "IsZeroFocalObjective",
    "BCEWithLogitsObjective",
    "HCEObjective",
    # Main container
    "LossTask",
    # Utility functions
    "get_loss_tasks",
    "loss_dict_to_task",
    "calculate_losses",
    "calculate_predictions",
    "combine_partial_predictions",
    "adapt_hce_prediction_and_labels_to_metrics_entries",
]

# Made with Bob
