import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import list_repo_files, snapshot_download
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
)
from transformers.models.auto import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
)

from bmfm_targets import config

logger = logging.getLogger(__name__)


def register_configs_and_models():
    from bmfm_targets.models.predictive import (
        scbert,
        scmodernbert,
        scnystromformer,
        scperformer,
    )

    _config_maps = [
        (
            config.SCBertConfig,
            scbert.SCBertForMaskedLM,
            scbert.SCBertForSequenceClassification,
        ),
        (
            config.SCPerformerConfig,
            scperformer.SCPerformerForMaskedLM,
            scperformer.SCPerformerForSequenceClassification,
        ),
        (
            config.SCNystromformerConfig,
            scnystromformer.SCNystromformerForMaskedLM,
            scnystromformer.SCNystromformerForSequenceClassification,
        ),
        (
            config.SCModernBertConfig,
            scmodernbert.SCModernBertForMaskedLM,
            scmodernbert.SCModernBertForSequenceClassification,
        ),
    ]

    for config_class, lm_class, seqcls_class in _config_maps:
        AutoConfig.register(config_class.model_type, config_class)
        AutoModelForMaskedLM.register(config_class, lm_class)
        AutoModelForSequenceClassification.register(config_class, seqcls_class)


def get_model_from_config(
    model_config: config.SCModelConfigBase, modeling_strategy: str
):
    from bmfm_targets.models.predictive import (
        scbert,
        scmodernbert,
        scnystromformer,
        scperformer,
    )

    # LLaMa
    if hasattr(model_config, "build_model"):
        return model_config.build_model(modeling_strategy)

    # scbert
    if isinstance(model_config, config.SCBertConfig):
        if modeling_strategy == "mlm":
            return scbert.SCBertForMaskedLM(model_config)
        if modeling_strategy == "sequence_classification":
            return scbert.SCBertForSequenceClassification(model_config)
        if modeling_strategy == "sequence_labeling":
            return scbert.SCBertForSequenceLabeling(model_config)
        if modeling_strategy == "multitask":
            return scbert.SCBertForMultiTaskModeling(model_config)
    # scperformer
    if isinstance(model_config, config.SCPerformerConfig):
        if modeling_strategy == "mlm":
            return scperformer.SCPerformerForMaskedLM(model_config)
        if modeling_strategy == "sequence_classification":
            return scperformer.SCPerformerForSequenceClassification(model_config)
        if modeling_strategy == "sequence_labeling":
            return scperformer.SCPerformerForSequenceLabeling(model_config)
        if modeling_strategy == "multitask":
            return scperformer.SCPerformerForMultiTaskModeling(model_config)
    # scnystromformer
    if isinstance(model_config, config.SCNystromformerConfig):
        if modeling_strategy == "mlm":
            return scnystromformer.SCNystromformerForMaskedLM(model_config)
        if modeling_strategy == "sequence_classification":
            return scnystromformer.SCNystromformerForSequenceClassification(
                model_config
            )
        if modeling_strategy == "sequence_labeling":
            return scnystromformer.SCNystromformerForSequenceLabeling(model_config)
        if modeling_strategy == "multitask":
            return scnystromformer.SCNystromformerForMultiTaskModeling(model_config)
    # SCModernBert
    if isinstance(model_config, config.SCModernBertConfig):
        if modeling_strategy == "mlm":
            return scmodernbert.SCModernBertForMaskedLM(model_config)
        if modeling_strategy == "sequence_classification":
            return scmodernbert.SCModernBertForSequenceClassification(model_config)
        if modeling_strategy == "sequence_labeling":
            return scmodernbert.SCModernBertForSequenceLabeling(model_config)
        if modeling_strategy == "multitask":
            return scmodernbert.SCModernBertForMultiTaskModeling(model_config)

    raise ValueError(f"Unknown model_config type {type(model_config)}")


def get_base_model_from_config(model_config: config.SCModelConfigBase):
    from bmfm_targets.models.predictive import (
        scbert,
        scmodernbert,
        scnystromformer,
        scperformer,
    )

    # LLaMa
    # TODO Unclear why get_base_model_from_config is called only by sequence classification
    # Why not to use get_model_from_config
    # Also bmfm_targets/models/predictive/multitask.py:88: in post_init
    # self.apply(self.base_model._init_weights) executed only for classification models
    # This function should never be called anyway, init_weights should be called instead
    if hasattr(model_config, "build_model"):
        return model_config.build_model(strategy="sequence_classification")

    if isinstance(model_config, config.SCBertConfig):
        base_model = scbert.modeling_scbert.SCBertModel(
            model_config, add_pooling_layer=True
        )
    elif isinstance(model_config, config.SCPerformerConfig):
        base_model = scperformer.modeling_scperformer.SCPerformerModel(
            model_config, add_pooling_layer=True
        )
    elif isinstance(model_config, config.SCNystromformerConfig):
        base_model = scnystromformer.modeling_scnystromformer.SCNystromformerModel(
            model_config, add_pooling_layer=True
        )
    elif isinstance(model_config, config.SCModernBertConfig):
        base_model = scmodernbert.modeling_scmodernbert.SCModernBertModel(model_config)

    return base_model


@dataclass
class SequenceClassifierOutputWithEmbeddings(SequenceClassifierOutput):
    embeddings: torch.FloatTensor | None = None


@dataclass
class MaskedLMOutputWithEmbeddings(SequenceClassifierOutput):
    embeddings: torch.FloatTensor | None = None


def instantiate_classification_model(model_config, loss_tasks, device=None):
    from bmfm_targets.models.predictive import MultiTaskClassifier

    if loss_tasks:
        if model_config.checkpoint:
            return MultiTaskClassifier.from_ckpt(
                model_config.checkpoint,
                loss_tasks=loss_tasks,
                model_config=model_config,
                device=device,
            )
        else:
            base_model = get_base_model_from_config(model_config)
            return MultiTaskClassifier(base_model, loss_tasks)

    return get_model_from_config(
        model_config, modeling_strategy="sequence_classification"
    )


def download_ckpt_from_huggingface(hf_repo) -> str:
    """
    Uses snapshot_download from huggingface_hub to download the files at
    hf_repo.
    Returns the path of the new .ckpt file.
    """
    local_hf_repo_path = snapshot_download(hf_repo, ignore_patterns=["*.git*", "*.md*"])
    logger.info(
        f"Downloaded checkpoint from HuggingFace: {hf_repo} - "
        f"Local path: {local_hf_repo_path}"
    )

    # Find the ckpt file in the downloaded directory
    ckpt_files = list(Path(local_hf_repo_path).glob("*.ckpt"))
    if not ckpt_files:
        logger.error(
            f"No .ckpt files found in the downloaded directory: {local_hf_repo_path}"
        )
        sys.exit(1)
    if len(ckpt_files) > 1:
        logger.warning(
            f"Multiple .ckpt files found in the directory: {local_hf_repo_path}. Using {ckpt_files[0]}."
        )
    checkpoint = ckpt_files[0]

    logger.info(f"Downloaded HF checkpoint to: {checkpoint}")

    return str(checkpoint)


def download_tokenizer_from_huggingface(hf_repo) -> None:
    """
    Uses snapshot_download from huggingface_hub to download the
    tokenizer-specific files from an hf repo.
    """
    hf_repo_files = list_repo_files(hf_repo)
    base_level_hf_repo_files = [f.split("/")[0] for f in hf_repo_files]
    if "tokenizers" in base_level_hf_repo_files:
        local_hf_repo_path = snapshot_download(
            repo_id=hf_repo, allow_patterns=["tokenizers*"]
        )
        logger.info(
            f"Downloaded tokenizer from HuggingFace: {hf_repo} - "
            f"Local path: {local_hf_repo_path}"
        )
        return local_hf_repo_path
    else:
        logger.warning(f"Tokenizer not found in HuggingFace repo: {hf_repo}")
        return hf_repo
