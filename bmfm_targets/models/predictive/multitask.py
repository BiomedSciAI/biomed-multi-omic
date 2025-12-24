import collections
from logging import getLogger

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.models import auto

import bmfm_targets.config
from bmfm_targets.models import register_configs_and_models
from bmfm_targets.models.model_utils import (
    SequenceClassifierOutputWithEmbeddings,
)
from bmfm_targets.models.predictive.layers import GradientReversal
from bmfm_targets.training.metrics import LabelLossTask

logger = getLogger()

register_configs_and_models()


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config: bmfm_targets.config.SCModelConfigBase, output_size: int):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, output_size)

        self.config = config

    def forward(self, pooled_output, **kwargs):
        x = self.dense(pooled_output)
        x = ACT2FN[self.config.hidden_act](x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class MultiTaskClassifier(nn.Module):
    def __init__(
        self, base_model: PreTrainedModel, loss_tasks: list[LabelLossTask], config=None
    ):
        """
        A classifier with multiple classification heads.

        Args:
        ----
            base_model (PreTrainedModel): the base model to classify the outputs of. Model
                should support pooler_output though it will fall back on first token pooling
            tasks (list[dict]): a list of tasks to define losses for. Each task dict must include:
                  - `label_column_name` which matches one of the `label_columns` in the dataset.
                It can optionally include:
                  - `task_group` which can be any string and, if present, will be used to create
                    a new layer for all `label_column_names` in the same `task_group`.
                  - `classifier_depth` which can be 1 for a simple nn.Linear classifier, or 2 for
                    a 2-layer classifier head. If absent, will use 1 for models with poolers and 2
                    for models without poolers.

        """
        super().__init__()
        self.base_model_prefix = "base_model"
        self.base_model = base_model
        self.config: bmfm_targets.config.SCModelConfigBase = (
            base_model.config if config is None else config
        )
        self.shared_latents = nn.ModuleDict()
        self.dropout = nn.Dropout(self.config.classifier_dropout)
        self.loss_tasks = loss_tasks
        self.classifiers = nn.ModuleDict()
        self.shared_latent_map = collections.defaultdict(list)

        for loss_task in loss_tasks:
            label_column_name = loss_task.label_column.label_column_name
            task_group = loss_task.label_column.task_group
            if task_group is not None and task_group not in self.shared_latents:
                self.shared_latents[task_group] = nn.Linear(
                    self.config.hidden_size, self.config.hidden_size
                )
            self.classifiers[label_column_name] = self._make_classifier(
                loss_task.label_column,
                loss_task.output_size,
            )

            self.shared_latent_map[task_group].append(label_column_name)

        self.post_init()

    def post_init(self):
        self.apply(self.base_model._init_weights)

    # TODO: @michael Should we keep this output_size as part of task or label_column's output size is enough?
    def _make_classifier(
        self, label_column: bmfm_targets.config.LabelColumnInfo, output_size
    ):
        depth = label_column.classifier_depth
        if depth == 1:
            cls = nn.Linear(self.config.hidden_size, output_size)
        elif depth == 2:
            cls = ClassificationHead(self.config, output_size)
        else:
            raise NotImplementedError(f"`classifier_depth`={depth} not supported.")
        if label_column.gradient_reversal_coefficient is not None:
            return nn.Sequential(
                GradientReversal(label_column.gradient_reversal_coefficient), cls
            )
        else:
            return cls

    @property
    def device(self) -> torch.device:
        return self.base_model.device

    @classmethod
    def from_ckpt(
        cls,
        ckpt_path: str,
        loss_tasks: list[LabelLossTask] | None = None,
        model_config: bmfm_targets.config.SCModelConfigBase | None = None,
        device=None,
    ):
        """
        Load MultiTask model from chekpoint.

        Will load base model from checkpoint for non-MultiTaskClassifier checkpoints.
        For MultiTaskClassifier checkpoints, the entire model will be loaded.

        The `tasks` and `label_output_size_dict` arguments can be used to change the tasks on the loaded
        checkpoint, or to add tasks to a base model.

        Currently, if a MultiTaskClassifier is trained, saved and then loaded with a different set of tasks
        the previous checkpoint's classifier weights are not preserved.

        model_config should be read directly from the ckpt to avoid potential architecture
        mismatch, but the legacy ckpts do not have it.
        It is supported here as an extra parameter but warned when used with a warning.

        Args:
        ----
            ckpt_path (str): path to checkpoint as generated by Pytorch Lightning
            label_output_size_dict (dict[str,int] | None, optional): Number of outputs for each label.
                Not required if loading from a MultiTaskClassifier checkpoint. Defaults to None.
            tasks (list[dict] | None, optional): List of tasks/losses.
                Not required if loading from a MultiTaskClassifier checkpoint. Defaults to None.
            model_config (SCModelConfigBase | None, optional): config object.
                Not required if loading from a post April 2024 checkpoint, because
                the model_config is stored inside the ckpt itself. Defaults to None.

        Returns:
        -------
            MultiTaskClassifier: model with weights loaded from ckpt

        """
        logger.info("Loading model from checkpoint " + str(ckpt_path))
        device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )
        ckpt_dict = torch.load(ckpt_path, map_location=device, weights_only=False)

        # manage loading model from lightning checkpoint
        if model_config is None and "model_config" in ckpt_dict["hyper_parameters"]:
            model_config = ckpt_dict["hyper_parameters"]["model_config"]
        # do not load the initial checkpoint weights
        model_config.checkpoint = None
        seq_cls_model = auto.AutoModelForSequenceClassification.from_config(
            model_config
        )

        state_dict = ckpt_dict["state_dict"]
        for key in list(state_dict.keys()):
            mt_key_name = key.removeprefix("model.")
            if model_config.model_type == mt_key_name.split(".")[0]:
                mt_key_name = ".".join(["base_model"] + mt_key_name.split(".")[1:])

            state_dict[mt_key_name] = state_dict.pop(key)

        if hasattr(seq_cls_model, "get_base_model"):
            base_model = seq_cls_model.get_base_model()
        else:
            base_model = getattr(seq_cls_model, model_config.model_type)

        # TODO: I dont understand this logic. Please take a look at this. If losses is none then load losses from checkpoint? why?
        if loss_tasks is None:
            loss_tasks = ckpt_dict["hyper_parameters"]["trainer_config"].losses

        mt = cls(base_model, loss_tasks, model_config)

        key_report = mt.load_state_dict(state_dict, strict=False)
        logger.info(f"Loading complete. {len(state_dict)} layers in ckpt.")
        logger.info(f"Unexpected keys: {key_report.unexpected_keys}")
        logger.info(f"Missing keys: {key_report.missing_keys}")
        return mt

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> SequenceClassifierOutputWithEmbeddings:
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if "pooler_output" in outputs:
            pooled_output = outputs["pooler_output"]
        else:
            pooled_output = outputs["last_hidden_state"][:, 0, :]
        pooled_output = self.dropout(pooled_output)

        logits = {}
        for task_group, task_list in self.shared_latent_map.items():
            if task_group is not None:
                task_pooled_output = self.shared_latents[task_group](pooled_output)
                task_pooled_output = self.dropout(task_pooled_output)
                task_pooled_output = ACT2FN[self.config.hidden_act](task_pooled_output)
            else:
                task_pooled_output = pooled_output

            for task_label in task_list:
                logits[task_label] = self.classifiers[task_label](task_pooled_output)

        return SequenceClassifierOutputWithEmbeddings(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            embeddings=pooled_output,
        )
