import functools
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

from bmfm_targets.config import (
    FieldInfo,
    LabelColumnInfo,
    PredictTaskConfig,
    TokenizerConfig,
    TrainerConfig,
    TrainingTaskConfig,
)
from bmfm_targets.config.model_config import SCModelConfigBase
from bmfm_targets.config.training_config import BaseTaskConfig

logger = logging.getLogger(__name__)


@dataclass
class SCBertMainHydraConfigSchema:
    # target TokenizerConfig
    tokenizer: dict | None = None
    # target a DataModule
    data_module: dict | None = None
    # list of label_columns to LabelColumnInfo
    label_columns: list[dict] | None = None
    # list of targets to FieldInfo
    fields: list[dict] | None = None
    # target TaskConfig
    task: Any = None
    # target TrainerConfig
    trainer: dict | None = None
    # target SCBertConfig
    model: dict | None = None
    # no target. track_clearml follows kwargs for clearml.Task.init
    track_clearml: dict | None = None
    # no target. a simple dict with key `seed_value`
    seed: dict | None = None


def get_label_output_size_for_model_config(
    data_module: pl.LightningDataModule,
    partial_model_config: functools.partial | None = None,
):
    if (
        partial_model_config is not None
        and partial_model_config.label_columns is not None
    ):
        return partial_model_config.label_columns[0].output_size
    if data_module.label_columns:
        return data_module.label_columns[0].output_size
    return None


@dataclass
class SCBertMainConfig:
    data_module: pl.LightningDataModule
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    fields: list[FieldInfo] | None = None
    label_columns: list[LabelColumnInfo] | None = None
    task: BaseTaskConfig | None = None
    trainer: TrainerConfig | None = None
    model: SCModelConfigBase | None = None
    track_clearml: dict | None = None
    seed: dict | None = None

    @classmethod
    def from_omegaconf_config_schema(cls, dict_config: DictConfig):
        new_kwargs = {}
        for k, v in dict_config.items():
            if v is not None and k in cls.__annotations__:
                new_kwargs[k] = cls._instantiate_recursive(v)
        return cls(**new_kwargs)

    @classmethod
    def _instantiate_recursive(cls, val):
        if OmegaConf.is_config(val) or OmegaConf.is_dict(val):
            return instantiate(val, _convert_="object")
        if OmegaConf.is_list(val):
            return [cls._instantiate_recursive(x) for x in val]
        if isinstance(val, functools.partial):
            return functools.partial(
                val.func, **instantiate(val.keywords, _convert_="all")
            )
        return val

    def complete_config(self):
        self._merge_configs_from_checkpoint()
        tokenizer = self._load_tokenizer_from_cfg()

        if isinstance(self.data_module, functools.partial):
            self.data_module = self._setup_data_module(tokenizer)

        self.save_tokenizer(tokenizer, self.task.default_root_dir)
        self._update_fields(tokenizer)
        self._update_label_columns()

        if isinstance(self.model, functools.partial):
            self.model = self.model(
                fields=self.fields,
                label_columns=self.label_columns,
                pad_token_id=self.data_module.tokenizer.pad_token_id,
            )
        elif self.model is not None:
            # Model was loaded from checkpoint (no model section in YAML) — sync updated values
            self.model.fields = self.fields
            self.model.label_columns = self.label_columns

        if (
            isinstance(self.task, TrainingTaskConfig)
            and self.task.enable_checkpointing is not False
        ):
            self.add_checkpointing_callbacks()

    def _setup_data_module(self, tokenizer):
        dm = self.data_module(
            tokenizer=tokenizer, fields=self.fields, label_columns=self.label_columns
        )
        if getattr(dm, "dataset_kwargs", None) and OmegaConf.is_config(
            dm.dataset_kwargs
        ):
            dm.dataset_kwargs = OmegaConf.to_container(dm.dataset_kwargs)
        dm.prepare_data()
        dm.transform_datasets = False
        dm.setup(self.task.setup_stage)
        return dm

    def _update_fields(self, tokenizer):
        for f in self.fields:
            f.update_vocab_size(tokenizer)
            if f.pretrained_embedding:
                f.update_pretrained_embedding_indices(tokenizer)

    def _update_label_columns(self):
        if not self.label_columns:
            return
        for lc in self.label_columns:
            if (
                self.data_module.label_dict
                and lc.label_column_name in self.data_module.label_dict
                and lc.n_unique_values is None
            ):
                lc.update_n_unique_values(self.data_module.label_dict)

    def _merge_fields(
        self,
        ckpt_fields: list[FieldInfo],
        yaml_fields: list[FieldInfo] | None,
        is_training: bool,
    ) -> list[FieldInfo]:
        """Merge fields: checkpoint wins for existing, new YAML fields added in training."""
        logger.info(
            msg=f"Merging fields: checkpoint fields: {ckpt_fields} and {yaml_fields}"
        )
        if not ckpt_fields:
            return yaml_fields if yaml_fields else []

        # Test/predict: checkpoint only
        if not is_training:
            if yaml_fields and yaml_fields != ckpt_fields:
                logger.warning(
                    "fields: Ignoring YAML config in test/predict mode. "
                    "Using checkpoint fields to match trained model."
                )
            return ckpt_fields

        # If no YAML fields during training, just use checkpoint fields
        if not yaml_fields:
            logger.info("fields: No YAML fields specified, using checkpoint fields")
            return ckpt_fields

        # Training: merge - checkpoint fields win, new YAML fields added
        ckpt_field_names = {f.field_name for f in ckpt_fields}
        yaml_field_names = {f.field_name for f in yaml_fields}

        merged = list(ckpt_fields)
        new_fields = [f for f in yaml_fields if f.field_name not in ckpt_field_names]
        if new_fields:
            logger.info(
                f"fields: Adding {len(new_fields)} new fields from YAML: "
                f"{[f.field_name for f in new_fields]}"
            )
            merged.extend(new_fields)

        conflicting = yaml_field_names & ckpt_field_names
        if conflicting:
            logger.info(
                f"fields: Using checkpoint config for existing fields: {sorted(conflicting)}"
            )

        return merged

    def _merge_label_columns(
        self,
        ckpt_cols: list[LabelColumnInfo],
        yaml_cols: list[LabelColumnInfo] | None,
        is_training: bool,
    ) -> list[LabelColumnInfo]:
        """Merge label_columns: checkpoint-authoritative for test/predict, YAML for training."""
        if not ckpt_cols:
            return yaml_cols if yaml_cols is not None else []

        if not is_training:
            # Explicit empty list in YAML = embedding-only mode (cross-dataset)
            if yaml_cols is not None and len(yaml_cols) == 0:
                logger.info(
                    "label_columns: Using empty list from YAML (embedding-only mode)"
                )
                return yaml_cols

            # Use checkpoint columns (normal same-dataset prediction)
            if yaml_cols and yaml_cols != ckpt_cols:
                logger.warning(
                    "label_columns: Ignoring YAML config in test/predict mode. "
                    "Using checkpoint label_columns to match trained model."
                )
            return ckpt_cols

        # Training: YAML-authoritative
        return yaml_cols if yaml_cols else ckpt_cols

    def _merge_configs_from_checkpoint(self) -> None:
        """Merge configs from checkpoint with YAML configs based on task type."""
        from bmfm_targets.models import download_ckpt_from_huggingface

        def _safe_get(config, key):
            if config is None:
                return None
            value = getattr(config, key, None)
            if value is not None:
                return value
            if hasattr(config, "get"):
                return config.get(key)
            return None

        def _summarize_config(config):
            if config is None:
                return "None"
            summary = [f"type={type(config).__name__}"]
            if hasattr(config, "keys"):
                try:
                    summary.append(f"keys={list(config.keys())}")
                except Exception as exc:
                    summary.append(f"keys=<error: {exc}>")
            else:
                attrs = [
                    attr
                    for attr in dir(config)
                    if not attr.startswith("_")
                    and not callable(getattr(config, attr, None))
                ]
                if attrs:
                    summary.append(f"attrs={attrs}")
            return ", ".join(summary)

        def _summarize_fields(fields):
            if fields is None:
                return "None"
            if isinstance(fields, list):
                names = [getattr(field, "field_name", repr(field)) for field in fields]
                return f"list(len={len(fields)}, names={names})"
            return f"{type(fields).__name__}: {fields!r}"

        checkpoint = self._get_checkpoint()
        if not checkpoint:
            return
        logger.info(f"Loading configs from checkpoint {checkpoint}")
        if not os.path.isfile(checkpoint):
            checkpoint = download_ckpt_from_huggingface(checkpoint)

        try:
            ckpt_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {checkpoint}: {e}")

        if "hyper_parameters" not in ckpt_dict:
            raise KeyError(f"Checkpoint {checkpoint} missing 'hyper_parameters' key")

        ckpt_hyper = ckpt_dict["hyper_parameters"]

        is_training = isinstance(self.task, TrainingTaskConfig)
        is_predict = isinstance(self.task, PredictTaskConfig)

        logger.info(
            "checkpoint merge: loading hyperparameters from %s; hyper_parameters=%s",
            checkpoint,
            _summarize_config(ckpt_hyper),
        )

        # Access checkpoint hyperparameters (can be dict or object)
        ckpt_model_config = _safe_get(ckpt_hyper, "model_config")
        ckpt_trainer_config = _safe_get(ckpt_hyper, "trainer_config")

        logger.info(
            "checkpoint merge: model config=%s; trainer config=%s",
            _summarize_config(ckpt_model_config),
            _summarize_config(ckpt_trainer_config),
        )

        # Try model_config first (preferred), fall back to top-level hyper_parameters (legacy)
        ckpt_fields = _safe_get(ckpt_model_config, "fields")
        if ckpt_fields is None:
            ckpt_fields = _safe_get(ckpt_hyper, "fields")
        logger.info(
            "checkpoint merge: extracted fields from checkpoint=%s; yaml fields=%s",
            _summarize_fields(ckpt_fields),
            _summarize_fields(self.fields),
        )
        # Merge fields (always merge, needed for model instantiation)
        if ckpt_fields is not None:
            self.fields = self._merge_fields(ckpt_fields, self.fields, is_training)
        else:
            logger.warning(
                "checkpoint merge: no fields found in checkpoint model config or top-level hyperparameters"
            )

        ckpt_label_columns = _safe_get(ckpt_model_config, "label_columns")
        if ckpt_label_columns is None:
            ckpt_label_columns = _safe_get(ckpt_hyper, "label_columns")

        # Merge label_columns (skip in predict mode for cross-dataset prediction)
        if not is_predict and ckpt_label_columns:
            self.label_columns = self._merge_label_columns(
                ckpt_label_columns, self.label_columns, is_training
            )
        elif is_predict and ckpt_model_config:
            # In predict mode, check if checkpoint has label decoder weights
            # If not, explicitly clear label_columns to prevent model instantiation with label heads
            has_label_weights = any(
                "label_predictions" in k for k in ckpt_dict["state_dict"].keys()
            )
            if not has_label_weights and ckpt_label_columns:
                logger.warning(
                    f"fields: Checkpoint has {len(ckpt_label_columns)} label_columns in config "
                    "but no label decoder weights. Clearing label_columns for predict mode."
                )
                self.label_columns = []

        # Populate model config from checkpoint when none was provided in YAML (e.g. finetune)
        if self.model is None and ckpt_model_config is not None:
            self.model = ckpt_model_config
            self.model.checkpoint = checkpoint

        # Merge trainer config
        if ckpt_trainer_config and self.trainer:
            self.trainer = self.trainer.merge_from_checkpoint(ckpt_trainer_config)

    @staticmethod
    def _instantiate_and_setup_data_module(
        data_module_partial, tokenizer, fields, label_columns, task
    ) -> pl.LightningDataModule:
        dm = data_module_partial(
            tokenizer=tokenizer, fields=fields, label_columns=label_columns
        )
        dataset_kwargs = getattr(dm, "dataset_kwargs", None)
        if dataset_kwargs and not isinstance(dataset_kwargs, dict):
            dm.dataset_kwargs = OmegaConf.to_container(dataset_kwargs)
        dm.prepare_data()
        dm.transform_datasets = False
        dm.setup(task.setup_stage)
        return dm

    @staticmethod
    def save_tokenizer(tokenizer, save_dir):
        legacy_format = not tokenizer.is_fast
        tokenizer.save_pretrained(str(save_dir), legacy_format=legacy_format)

    @staticmethod
    def update_fields(fields: list[FieldInfo], tokenizer):
        for field_info in fields:
            field_info.update_vocab_size(tokenizer)
            if field_info.pretrained_embedding:
                field_info.update_pretrained_embedding_indices(tokenizer)

    @staticmethod
    def _instantiate_loss_tasks(
        losses: list[Any],
        fields: list[FieldInfo],
        label_columns: list[LabelColumnInfo],
    ) -> list[Any]:
        """
        Convert loss configs to LossTask objects at the config layer.

        By this point, Hydra configs should already be instantiated by
        from_omegaconf_config_schema(). This method only handles:
        - Already-instantiated LossTask objects (pass through)
        - Old-style dicts (convert via loss_dict_to_task)

        Does NOT call bind() - that happens in training module.

        Args:
        ----
            losses: List of loss configs (LossTask objects or old-style dicts)
            fields: List of FieldInfo from model config
            label_columns: List of LabelColumnInfo from model config

        Returns:
        -------
            list[LossTask]: List of instantiated (but not bound) loss tasks

        Raises:
        ------
            TypeError: If Hydra config with _target_ is passed (should be instantiated upstream)

        """
        from bmfm_targets.training.losses import LossTask, loss_dict_to_task

        result = []
        for loss_config in losses:
            if isinstance(loss_config, LossTask):
                # Already instantiated - pass through
                result.append(loss_config)
            elif isinstance(loss_config, dict):
                if "_target_" in loss_config:
                    raise TypeError(
                        f"Hydra config with _target_ should be instantiated in "
                        f"from_omegaconf_config_schema(), not here. Got: {loss_config}"
                    )
                # Old-style dict - convert via compat layer
                result.append(loss_dict_to_task(loss_config, fields, label_columns))
            else:
                raise TypeError(f"Expected LossTask or dict, got {type(loss_config)}")

        return result

    @staticmethod
    def update_label_columns(label_columns: list[LabelColumnInfo], label_dict):
        for label_column in label_columns:
            if (
                label_dict is not None
                and label_column.label_column_name in label_dict
                and label_column.n_unique_values is None
            ):
                label_column.update_n_unique_values(label_dict)

    def _load_tokenizer_from_cfg(self):
        # to avoid circular import
        from bmfm_targets.models import download_tokenizer_from_huggingface
        from bmfm_targets.tokenization import load_tokenizer

        checkpoint = self._get_checkpoint()
        if checkpoint:
            if os.path.isfile(checkpoint):
                identifier: str = os.path.dirname(checkpoint)
            else:
                identifier = str(download_tokenizer_from_huggingface(checkpoint))
            for f in self.fields:
                if f.vocab_update_strategy == "dynamic":
                    logger.warning(
                        f"Field {f.field_name} is set to dynamic vocab update strategy. "
                        "Switching to static vocab update strategy as you are loading from a checkpoint."
                    )
                f.vocab_update_strategy = "static"
        else:
            identifier = self.tokenizer.identifier

        return load_tokenizer(identifier, self.tokenizer.prepend_tokens)

    def _get_checkpoint(self):
        task_ckpt = getattr(self.task, "checkpoint", None)
        model_ckpt = None
        if self.model is not None:
            if isinstance(self.model, functools.partial):
                model_ckpt = self.model.keywords.get("checkpoint")
            else:
                model_ckpt = getattr(self.model, "checkpoint", None)
        if task_ckpt is not None:
            if model_ckpt is not None and model_ckpt != task_ckpt:
                logger.warning(
                    "Found different checkpoints in task and model config, using task checkpoint"
                )
            return task_ckpt
        return model_ckpt

    def add_checkpointing_callbacks(self):
        """
        Saves last and best model checkpoints. Last mode checkpoint is saved to a fixed filename,
        `last.ckpt` each time validation is run, to allows re-running training from the last model created.
        Best model based on validation loss is saved to a file holding the epoch step and validation loss
        details. Previous best model is deleted each time a "new" best model is saved.
        """
        filename = "epoch={epoch}-step={step}-val_loss={validation/loss:.2f}"

        self.task.callbacks.append(
            ModelCheckpoint(
                dirpath=Path(self.task.default_root_dir),
                save_last=(not self.task.checkpoints_every_n_train_steps),
                save_top_k=0,
                filename=filename,
                auto_insert_metric_name=False,
            )
        )
        if self.task.checkpoints_every_n_train_steps:
            self.task.callbacks.append(
                ModelCheckpoint(
                    dirpath=Path(self.task.default_root_dir),
                    save_last=True,
                    save_top_k=0,
                    filename=filename,
                    auto_insert_metric_name=False,
                    every_n_train_steps=self.task.checkpoints_every_n_train_steps,
                )
            )
        self.task.callbacks.append(
            ModelCheckpoint(
                dirpath=Path(self.task.default_root_dir),
                save_top_k=1,
                monitor="validation/loss",
                mode="min",
                filename=filename,
                auto_insert_metric_name=False,
            )
        )
