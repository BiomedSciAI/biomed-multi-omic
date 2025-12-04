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
    TestTaskConfig,
    TokenizerConfig,
    TrainerConfig,
    TrainingTaskConfig,
)
from bmfm_targets.config.model_config import SCModelConfigBase
from bmfm_targets.config.training_config import BaseTaskConfig

logger = logging.getLogger(__name__)


def default_fields() -> list[FieldInfo]:
    return [
        FieldInfo(field_name="genes", is_masked=False, vocab_update_strategy="static"),
        FieldInfo(
            field_name="expressions",
            is_masked=True,
            vocab_update_strategy="static",
            decode_modes={"token_scores": {}},
        ),
    ]


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
    # track_clearml follows kwargs for clearml.Task.init
    track_clearml: dict | None = None
    # a simple dict with key `seed_value`
    seed: dict | None = None

    @classmethod
    def from_omegaconf_config_schema(cls, dict_config: DictConfig):
        # self.data_module = instantiate(self.data_module)
        new_kwargs = {
            k: cls._instantiate_configs_only(v)
            for k, v in dict_config.items()
            if v is not None and k in cls.__annotations__
        }
        return cls(**new_kwargs)

    @classmethod
    def _instantiate_configs_only(cls, val):
        if OmegaConf.is_config(val):
            return instantiate(val, _convert_="object")
        if OmegaConf.is_dict(val):
            return instantiate(val, _convert_="all")
        if OmegaConf.is_list(val):
            return [instantiate(x, _convert_="object") for x in val]
        if isinstance(val, functools.partial):
            keywords = instantiate(val.keywords, _convert_="all")
            return functools.partial(val.func, **keywords)
        return val

    def complete_config(self):
        self._load_missing_configs_from_ckpt()
        tokenizer = self._load_tokenizer_from_cfg()
        if isinstance(self.data_module, functools.partial):
            self.data_module = self._instantiate_and_setup_data_module(
                self.data_module, tokenizer, self.fields, self.label_columns, self.task
            )
        self.save_tokenizer(tokenizer, self.task.default_root_dir)
        self.update_fields(self.fields, tokenizer)
        if self.label_columns:
            self.update_label_columns(self.label_columns, self.data_module.label_dict)
        if isinstance(self.model, functools.partial):
            self.model = self._instantiate_model_config(
                self.model, self.data_module, self.fields, self.label_columns
            )
        if (
            isinstance(self.task, TrainingTaskConfig)
            and self.task.enable_checkpointing is not False
        ):
            self.add_checkpointing_callbacks()

    def _load_missing_configs_from_ckpt(self):
        from bmfm_targets.models import download_ckpt_from_huggingface

        checkpoint = self._get_checkpoint()
        if not checkpoint:
            return
        if not os.path.isfile(checkpoint):
            checkpoint = download_ckpt_from_huggingface(checkpoint)
        ckpt_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if self.fields is None:
            self.fields = ckpt_dict["hyper_parameters"]["model_config"].fields
        if isinstance(self.task, TestTaskConfig) and self.label_columns is None:
            self.label_columns = getattr(
                ckpt_dict["hyper_parameters"]["model_config"], "label_columns", None
            )

    @staticmethod
    def _instantiate_model_config(partial_model, data_module, fields, label_columns):
        return partial_model(
            fields=fields,
            label_columns=label_columns,
            pad_token_id=data_module.tokenizer.pad_token_id,
        )

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
                identifier = os.path.dirname(checkpoint)
            else:
                identifier = download_tokenizer_from_huggingface(checkpoint)
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
                model_ckpt = self.model.checkpoint
        if task_ckpt is not None:
            if model_ckpt is not None:
                logger.warning(
                    "Found checkpoint in model config and task config, using task checkpoint"
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
