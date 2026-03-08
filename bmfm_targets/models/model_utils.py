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
    """
    Register model configs and multitask models with HuggingFace AutoModel.

    Multitask models are registered for BOTH AutoModelForMaskedLM and
    AutoModelForSequenceClassification since they handle both tasks.
    This maintains HF API compatibility while using only multitask models internally.
    """
    from bmfm_targets.models.predictive import (
        scbert,
        scmodernbert,
        scnystromformer,
        scperformer,
    )

    _config_maps = [
        (
            config.SCBertConfig,
            scbert.SCBertForMultiTaskModeling,
        ),
        (
            config.SCPerformerConfig,
            scperformer.SCPerformerForMultiTaskModeling,
        ),
        (
            config.SCNystromformerConfig,
            scnystromformer.SCNystromformerForMultiTaskModeling,
        ),
        (
            config.SCModernBertConfig,
            scmodernbert.SCModernBertForMultiTaskModeling,
        ),
    ]

    for config_class, multitask_class in _config_maps:
        AutoConfig.register(config_class.model_type, config_class)
        # Register multitask model for both MLM and SeqCls
        # This is correct since multitask models handle both tasks
        AutoModelForMaskedLM.register(config_class, multitask_class, exist_ok=True)
        AutoModelForSequenceClassification.register(
            config_class, multitask_class, exist_ok=True
        )


def get_model_from_config(model_config: config.SCModelConfigBase):
    """
    Get model from config. Always returns multitask model.

    Args:
    ----
        model_config: Model configuration

    Returns:
    -------
        Multitask model instance
    """
    from bmfm_targets.models.predictive import (
        scbert,
        scmodernbert,
        scnystromformer,
        scperformer,
    )

    # LLaMa
    if hasattr(model_config, "build_model"):
        return model_config.build_model()

    # Always return multitask models
    if isinstance(model_config, config.SCBertConfig):
        return scbert.SCBertForMultiTaskModeling(model_config)
    elif isinstance(model_config, config.SCPerformerConfig):
        return scperformer.SCPerformerForMultiTaskModeling(model_config)
    elif isinstance(model_config, config.SCNystromformerConfig):
        return scnystromformer.SCNystromformerForMultiTaskModeling(model_config)
    elif isinstance(model_config, config.SCModernBertConfig):
        return scmodernbert.SCModernBertForMultiTaskModeling(model_config)
    else:
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


def migrate_checkpoint_if_needed(
    checkpoint_path: str, label_name: str | None = None
) -> dict:
    """Load and migrate old checkpoint formats to multitask."""
    from run.migrate_checkpoints_to_multitask import (
        convert_mlm_to_multitask,
        convert_seqcls_to_multitask,
        detect_checkpoint_type,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    # Remove "model." prefix for detection
    cleaned = {k[6:] if k.startswith("model.") else k: v for k, v in state_dict.items()}

    ckpt_type, detected_label = detect_checkpoint_type(cleaned)
    if ckpt_type == "multitask":
        return ckpt

    logger.info(f"Migrating {ckpt_type} checkpoint")
    label_name = detected_label or label_name

    if ckpt_type == "mlm_or_seqlabel":
        migrated = convert_mlm_to_multitask(cleaned)
    elif ckpt_type in ("sequence_classification", "multitask_classifier"):
        if not label_name:
            raise ValueError("SeqCls checkpoint requires label_name")
        migrated = convert_seqcls_to_multitask(cleaned, label_name=label_name)
    else:
        raise ValueError(f"Unknown checkpoint type: {ckpt_type}")

    if "state_dict" in ckpt:
        ckpt["state_dict"] = migrated
    else:
        ckpt = migrated
    return ckpt
