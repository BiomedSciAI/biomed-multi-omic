"""Evaluation utilities for trained models."""

from .embeddings import (
    evaluate_clusters,
    generate_clusters,
    load_predictions,
    load_prediction_data_to_anndata,
    calculate_metrics_embedding,
)
from .plots import (
    plot_cell_type_counts,
    plot_embeddings,
    plot_logits,
)
from .predictions import get_general_cell_type, merge_bmfm_adata
from .clearml_utils import get_folder_of_checkpoints, get_model_path, get_root_folder
from .utils import (
    check_gpu,
    get_label_dict,
)


__all__ = [
    "calculate_metrics_embedding",
    "check_gpu",
    "convert_ids_to_label_names",
    "create_results_obs",
    "evaluate_clusters",
    "generate_clusters",
    "get_folder_of_checkpoints",
    "get_general_cell_type",
    "get_label_dict",
    "get_label_map",
    "get_model_path",
    "get_root_folder",
    "load_prediction_data_to_anndata",
    "load_predictions",
    "merge_bmfm_adata",
    "plot_cell_type_counts",
    "plot_embeddings",
    "plot_logits",
]
