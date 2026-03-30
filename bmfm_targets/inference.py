"""
Simple inference API for BMFM models.

Provides a scanpy-style interface: bmfm.inference(adata)
"""

import logging
import os
import warnings
from contextlib import contextmanager
from typing import Literal

import torch
from anndata import AnnData

from bmfm_targets import config
from bmfm_targets.models import download_ckpt_from_huggingface
from bmfm_targets.tasks import task_utils
from bmfm_targets.tokenization import load_tokenizer
from bmfm_targets.training.data_module import DataModule

logger = logging.getLogger(__name__)


def _setup_logging():
    """Configure logging for clean CLI output."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning.utilities.migration").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning.trainer.connectors.data_connector").setLevel(
        logging.ERROR
    )

    warnings.filterwarnings("ignore", message=".*Lightning v.*")
    warnings.filterwarnings("ignore", message=".*Tie weights not supported.*")
    warnings.filterwarnings("ignore", message=".*does not have many workers.*")


@contextmanager
def use_layer(adata, layer):
    """
    Temporarily swap adata.X with a specified layer.

    Parameters
    ----------
    adata : AnnData
        Annotated data object
    layer : str or None
        Layer name to use as X. If None, no swap occurs.

    Yields
    ------
    AnnData
        The AnnData object with swapped layer
    """
    if layer is None:
        yield adata
        return

    if layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' not found in adata.layers")

    original_X = adata.X
    try:
        logger.info(f"Switching inference to layer='{layer}'")
        adata.X = adata.layers[layer]
        yield adata
    finally:
        adata.X = original_X


def inference(
    adata: AnnData,
    checkpoint: str | None = "ibm-research/biomed.rna.bert.110m.wced.multitask.v1",
    layer: str | None = None,
    embedding_key: str = "X_bmfm",
    prediction_key_prefix: str = "bmfm_pred_",
    batch_size: int = 8,
    max_length: int = 1024,
    limit_genes: Literal["tokenizer", "protein_coding"] | None = "protein_coding",
    device: str = "auto",
    copy: bool = False,
    output_dir: str | None = None,
    pooling_method: str | int | list[int | str] | None = None,
    log_normalize_transform: bool = True,
    **kwargs,
) -> AnnData:
    """
    Run BMFM zero-shot inference on an AnnData object.

    This function provides a simple, scanpy-style API for running BMFM model
    predictions on single-cell RNA-seq data. All model configuration is loaded
    directly from the checkpoint, making it fully self-contained.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object containing single-cell expression data
    checkpoint_path : str
        Path to model checkpoint (.ckpt file). The checkpoint contains all
        necessary configuration for inference.
    layer : str, optional
        Layer to use for expression values. If None, uses adata.X
    embedding_key : str, optional
        Key to store embeddings in adata.obsm. Default is 'X_bmfm'
    prediction_key_prefix : str, optional
        Prefix for prediction columns in adata.obs. Default is 'bmfm_pred_'
    batch_size : int, optional
        Batch size for inference. Default is 8
    max_length : int, optional
        Maximum sequence length for tokenization. Default is 1024
    limit_genes : str, optional
        Gene filtering strategy. Default is 'protein_coding'. Options include:
        - 'protein_coding': Only use protein-coding genes
        - None: Use all genes
    device : str, optional
        Device to run inference on ('cpu', 'cuda', 'auto'). Default is 'auto'
    copy : bool, optional
        Whether to return a copy of adata or modify in place. Default is False (in-place).
    output_dir : str, optional
        Directory for temporary outputs. If None, uses current directory.
    pooling_method : str | int | list[int | str], optional
        Pooling method for extracting embeddings. If None, uses the method from the
        checkpoint's trainer_config. Options:
        - "first_token": Use CLS token at position 0 (safe default for MLM checkpoints)
        - "pooling_layer": Use trained pooler layer (only for checkpoints with trained pooler)
        - "mean_pooling": Average all tokens except CLS
        - int: Use token at specific position (e.g., 0 for first CLS, 1 for second position)
        - list[int | str]: Concatenate embeddings from multiple methods
            Examples:
            * [0, 1, 2] - Concatenate positions 0, 1, and 2
            * ["pooling_layer", "mean_pooling"] - Concatenate pooler and mean pooling
            * [0, "pooling_layer", 1] - Mix positions and pooling methods
        Default is None (use checkpoint's setting).
    log_normalize_transform : bool, optional
        Whether to apply log normalization transform to the embeddings. Default is False.
    **kwargs
        Additional arguments (reserved for future use)

    Returns
    -------
    AnnData
        Modified AnnData object with:
        - Embeddings in adata.obsm[embedding_key]
        - Predictions in adata.obs[prediction_key_prefix + label_name]
        If copy=True, returns a new AnnData object. If copy=False, modifies in place and returns the same object.

    Examples
    --------
    >>> import bmfm_targets as bmfm
    >>> import scanpy as sc
    >>>
    >>> # Load data
    >>> adata = sc.read_h5ad("data.h5ad")
    >>>
    >>> # Run inference with a trained checkpoint
    >>> adata = bmfm.inference(adata, checkpoint_path="path/to/model.ckpt")
    >>>
    >>> # Use embeddings for downstream analysis
    >>> sc.pp.neighbors(adata, use_rep='X_bmfm')
    >>> sc.tl.umap(adata)
    >>> sc.pl.umap(adata, color='bmfm_pred_cell_type')

    Notes
    -----
    - This performs zero-shot inference using pre-trained models
    - All configuration is loaded from the checkpoint's hyper_parameters
    - The checkpoint is fully self-contained and portable
    """
    _setup_logging()

    if copy:
        adata = adata.copy()

    logger.info(
        f"Running BMFM inference on {adata.n_obs} cells with {adata.n_vars} genes"
    )

    if checkpoint and not os.path.isfile(checkpoint):
        checkpoint = download_ckpt_from_huggingface(checkpoint)

    assert checkpoint is not None, "Checkpoint path should not be None after validation"

    if device == "auto":
        if torch.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    tokenizer = load_tokenizer(os.path.dirname(checkpoint))
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ckpt_hyper = ckpt["hyper_parameters"]
    fields = ckpt_hyper.get("model_config").fields

    with use_layer(adata, layer):
        dataset_kwargs = {"processed_data_source": adata}

        data_module = DataModule(
            tokenizer=tokenizer,
            fields=fields,
            label_columns=[],
            batch_size=batch_size,
            max_length=max_length,
            dataset_kwargs=dataset_kwargs,
            limit_genes=limit_genes,
            mlm=False,
            data_dir=None,
            transform_datasets=False,
            log_normalize_transform=log_normalize_transform,
        )
        data_module.setup("predict")

        task_config = config.PredictTaskConfig(
            checkpoint=checkpoint,
            output_embeddings=True,
            output_predictions=True,
            accelerator="gpu" if device in ["cuda", "mps"] else "cpu",
            precision="16-mixed" if device == ["cuda", "mps"] else "32",
            default_root_dir=output_dir or ".",
            enable_progress_bar=True,
            enable_model_summary=False,
            callbacks=[],
        )

        pl_trainer = task_utils.make_trainer_for_task(task_config)
        pl_module = task_utils.instantiate_module_from_checkpoint(
            task_config, data_module, trainer_config=None
        )

        # Override pooling_method if user specified one
        if pooling_method is not None:
            logger.info(f"Using user-specified pooling_method: {pooling_method}")
            pl_module.trainer_config.pooling_method = pooling_method

        logger.info("Running inference...")
        results = task_utils.predict(pl_trainer, pl_module, data_module)

        logger.info("Adding to Anndata object...")
        _add_results_to_adata(
            adata, results, pl_module, embedding_key, prediction_key_prefix
        )

    logger.info("Inference complete!")
    return adata


def _add_results_to_adata(
    adata: AnnData,
    results: dict,
    pl_module,
    embedding_key: str,
    prediction_key_prefix: str,
    debug: bool = False,
) -> None:
    """
    Add prediction results to AnnData object.

    Separated for clarity and following predict_run's result handling pattern.
    """
    if "embeddings" in results:
        adata.obsm[embedding_key] = results["embeddings"]
        logger.info(
            f"Added embeddings to adata.obsm['{embedding_key}'] with shape {results['embeddings'].shape}"
        )

    for key, values in results.items():
        if "predictions" not in key:
            continue

        if key in ["donor_id_predictions", "sex_predictions"] and not debug:
            continue

        print(f"{key=}")
        label_name = key.replace("_predictions", "")
        obs_key = f"{prediction_key_prefix}{label_name}"

        if (
            hasattr(pl_module, "label_dict")
            and pl_module.label_dict is not None
            and label_name in pl_module.label_dict
        ):
            id_to_label = {v: k for k, v in pl_module.label_dict[label_name].items()}
            if len(id_to_label) > 1:
                adata.obs[obs_key] = [id_to_label.get(int(v), str(v)) for v in values]
            else:
                adata.obs[obs_key] = values
        else:
            adata.obs[obs_key] = values

        logger.info(f"Added predictions to adata.obs['{obs_key}']")
