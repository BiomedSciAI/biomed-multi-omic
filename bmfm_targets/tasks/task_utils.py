import inspect
import logging
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TypeVar

import anndata
import clearml
import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import transformers
from clearml import Task, TaskTypes
from clearml.backend_api.session.defs import MissingConfigError
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from bmfm_targets import config
from bmfm_targets.models import download_ckpt_from_huggingface
from bmfm_targets.training.data_module import DataModule
from bmfm_targets.training.metrics import (
    log_confusion_matrix_to_clearml,
)
from bmfm_targets.training.metrics.metric_functions import calculate_95_ci
from bmfm_targets.training.modules import (
    SequenceClassificationTrainingModule,
    get_training_module_class_for_data_module,
)
from bmfm_targets.training.modules.base import BaseTrainingModule

TaskInstance = TypeVar("TaskInstance", bound="Task")

logger = logging.getLogger(__name__)


def main_config(cfg):
    OmegaConf.register_new_resolver("num_cores_auto", num_available_cores, replace=True)

    cfg_dict_conf = hydra.utils.instantiate(cfg, _recursive_=False)
    cfg_obj = config.SCBertMainConfig.from_omegaconf_config_schema(cfg_dict_conf)
    cfg_obj.complete_config()

    if cfg_obj.seed:
        seed_value = seed(cfg_obj.seed["seed_value"])
        logger.info(f"seed: {seed_value}")

    if cfg_obj.task.accelerator == "gpu" and cfg_obj.task.tf32_mode:
        torch.set_float32_matmul_precision(cfg_obj.task.tf32_mode)
    return cfg_obj


def main_run(
    task_config: config.training_config.BaseTaskConfig,
    model_config: config.SCModelConfigBase,
    data_module: DataModule,
    trainer_config: config.TrainerConfig,
    clearml_logger: clearml.Logger | None,
) -> pl.Trainer:
    pl_trainer = make_trainer_for_task(task_config)
    if isinstance(task_config, config.PredictTaskConfig):
        predict_run(pl_trainer, task_config, model_config, data_module, trainer_config)
    elif isinstance(task_config, config.TestTaskConfig):
        test_run(
            pl_trainer,
            task_config,
            model_config,
            data_module,
            trainer_config,
            clearml_logger,
        )
    elif isinstance(task_config, config.TrainingTaskConfig):
        train_run(pl_trainer, task_config, model_config, data_module, trainer_config)
    elif isinstance(task_config, config.InterpretTaskConfig):
        interpret_run(pl_trainer, task_config, data_module, model_config)
    return pl_trainer


def make_trainer_for_task(task_config):
    valid_kwargs = inspect.signature(pl.Trainer).parameters.keys()
    trainer_kwargs = {k: v for k, v in asdict(task_config).items() if k in valid_kwargs}
    pl_trainer = pl.Trainer(**trainer_kwargs)
    return pl_trainer


def train_run(pl_trainer, task_config, model_config, data_module, trainer_config):
    if model_config.checkpoint:
        if not os.path.isfile(model_config.checkpoint):
            model_config.checkpoint = download_ckpt_from_huggingface(
                model_config.checkpoint
            )
    pl_module_factory = get_training_module_class_for_data_module(data_module)
    checkpoint_path = None
    if task_config.resume_training_from_ckpt:
        checkpoint_path = model_config.checkpoint
        model_config.checkpoint = None
    extra_kwargs = prepare_extra_training_module_kwargs(data_module)
    if data_module.label_dict is not None:
        extra_kwargs["label_dict"] = data_module.label_dict
    pl_module = pl_module_factory(model_config, trainer_config, **extra_kwargs)
    train(
        pl_trainer,
        pl_data_module=data_module,
        pl_module=pl_module,
        task_config=task_config,
        checkpoint_path=checkpoint_path,
    )


def test_run(
    pl_trainer: pl.Trainer,
    task_config: config.TestTaskConfig,
    model_config: config.model_config.SCModelConfigBase | None,
    data_module: DataModule,
    trainer_config: config.TrainerConfig | None,
    clearml_logger: clearml.Logger | None,
):
    if task_config.checkpoint is None:
        logger.error("Test task requires TestTaskConfig.checkpoint to be set")
        sys.exit(1)

    if not os.path.isfile(task_config.checkpoint):
        task_config.checkpoint = download_ckpt_from_huggingface(task_config.checkpoint)

    pl_module = instantiate_module_from_checkpoint(
        task_config, data_module, model_config, trainer_config
    )

    if task_config.num_bootstrap_runs >= 1:
        data_module.bootstrap_test_dataloader = True
        runs_list = [
            test(
                pl_trainer,
                pl_module,
                data_module,
                task_config.checkpoint,
            )
            for _ in range(task_config.num_bootstrap_runs)
        ]
        n = task_config.num_bootstrap_runs
    else:
        runs_list = [
            test(
                pl_trainer,
                pl_module,
                data_module,
                task_config.checkpoint,
            )
        ]
        n = len(data_module.test_dataset)
    results = {key: [d[key] for d in runs_list] for key in runs_list[0].keys()}
    metrics_to_log = prepare_test_metrics_for_logging(
        results, n, ci_method=task_config.ci_method
    )
    if clearml_logger:
        logger.info("Confusion_matrix is reported for last bootstrap only")
        log_test_metrics(clearml_logger, metrics_to_log, data_module.label_dict)
    return runs_list


def predict_run(pl_trainer, task_config, model_config, data_module, trainer_config):
    if task_config.checkpoint is None:
        logger.error("Predict task requires a checkpoint in task.checkpoint")
        sys.exit(1)

    if not os.path.isfile(task_config.checkpoint):
        task_config.checkpoint = download_ckpt_from_huggingface(task_config.checkpoint)

    pl_module = instantiate_module_from_checkpoint(
        task_config, data_module, model_config, trainer_config
    )

    results = predict(pl_trainer, pl_module, data_module)
    if task_config.output_predictions:
        if pl_module.MODELING_STRATEGY in ("multitask", "sequence_classification"):
            save_prediction_results(
                task_config.default_root_dir,
                pl_module.label_dict,
                results,
            )
            save_logits_results(
                task_config.default_root_dir,
                pl_module.label_dict,
                results,
            )
        else:
            logger.warning(
                f"Predictions not supported for modeling strategy: {pl_module.MODELING_STRATEGY}"
            )

    if task_config.output_embeddings:
        save_embeddings_results(task_config.default_root_dir, results)


def instantiate_module_from_checkpoint(
    task_config, data_module, model_config, trainer_config
):
    from bmfm_targets.tokenization import load_tokenizer

    if task_config.checkpoint is None:
        raise ValueError("Cannot run test or predict run without a checkpoint!")
    data_module.tokenizer = load_tokenizer(os.path.dirname(task_config.checkpoint))

    pl_factory = get_training_module_class_for_data_module(data_module)
    extra_kwargs = prepare_extra_training_module_kwargs(data_module)
    if trainer_config is not None:
        extra_kwargs["trainer_config"] = trainer_config
    extra_kwargs["config_is_loaded_from_ckpt"] = True
    if model_config is None:
        logger.info(
            "Model config is none then loading model from checkpoint "
            + str(task_config.checkpoint)
        )
        pl_module = pl_factory.load_from_checkpoint(
            task_config.checkpoint, **extra_kwargs, weights_only=False
        )
    else:
        model_config.checkpoint = task_config.checkpoint
        for field in model_config.fields:
            field.update_vocab_size(data_module.tokenizer)
        if data_module.label_dict is not None:
            extra_kwargs["label_dict"] = data_module.label_dict
        pl_module = pl_factory(model_config=model_config, **extra_kwargs)

    return pl_module


def prepare_extra_training_module_kwargs(
    data_module: DataModule, de_genes_key: str = "rank_genes_groups_cov_all"
):
    """
    Prepare kwargs for module instantiation.

    Different extra kwargs are available for different training modules.
    The behavior is determined by the data module, which provides different kwargs.

    Args:
    ----
        data_module (DataModule): the data module
        de_genes_key (str, optional): The set of DE genes to use for comparing perturbations.
          For matching scGPT perturbation metrics use "rank_genes_groups_cov_all"
          other supported values are "top_non_dropout_de_20" and "top_non_zero_de_20"
          Defaults to "rank_genes_groups_cov_all".

    Returns:
    -------
        dict: additional kwargs

    """
    module_kwargs = {"tokenizer": data_module.tokenizer}
    ds = data_module.get_dataset_instance()

    if hasattr(ds, "group_means"):
        module_kwargs["group_means"] = ds.group_means
        logger.info("Training dataset contains group means ...")
    else:
        logger.info("Training dataset does not contains group means ...")

    if (
        hasattr(ds, "processed_data")
        and isinstance(ds.processed_data, anndata.AnnData)
        and de_genes_key in ds.processed_data.uns
    ):
        top20_de = {k: v[:20] for k, v in ds.processed_data.uns[de_genes_key].items()}
        module_kwargs["top20_de"] = top20_de

    return module_kwargs


def seed(seed_value: int) -> int:
    pl.seed_everything(seed_value, workers=True)
    transformers.set_seed(seed_value)
    return seed_value


def train(
    pl_trainer: pl.Trainer,
    pl_data_module: DataModule,
    pl_module: pl.LightningModule,
    task_config: config.TrainingTaskConfig,
    checkpoint_path: str | None = None,
):
    if task_config.freeze_layers:
        logger.info("Freezing layers")
        match_str = (
            r"\.encoder\."
            if isinstance(task_config.freeze_layers, bool)
            and task_config.freeze_layers is True
            else task_config.freeze_layers
        )
        for name, param in pl_module.model.named_parameters():
            if re.search(match_str, name):
                param.requires_grad = False
            else:
                logger.info(f"Unfrozen layer: {name}")

    pl_trainer.callbacks = (
        getattr(pl_trainer, "callbacks", None) or []
    ) + pl_data_module.get_trainer_callbacks()

    pl_trainer.fit(
        model=pl_module,
        datamodule=pl_data_module,
        ckpt_path=checkpoint_path,
    )


def test(
    pl_trainer: pl.Trainer,
    pl_module: BaseTrainingModule,
    test_dataloader: DataLoader,
    checkpoint_path: str | None = None,
):
    kwargs = {
        "model": pl_module,
        "dataloaders": test_dataloader,
        "ckpt_path": checkpoint_path,
    }
    # handle pytorch-lightning >=2.6.0 which requires weights_only and earlier which
    # cannot have it as an arg
    if "weights_only" in pl_trainer.test.__annotations__:
        kwargs["weights_only"] = False
    pl_trainer.test(**kwargs)
    metrics = pl_module.test_metrics.compute()
    # flatten nested dict of label:metric_name:metric_value
    metrics = {
        f"{label_name}_{metric_name}": metric_value
        for label_name, metrics_dict in metrics.items()
        for metric_name, metric_value in metrics_dict.items()
    }

    return metrics


def predict(
    pl_trainer: pl.Trainer,
    pl_module: SequenceClassificationTrainingModule,
    pl_data_module: pl.LightningDataModule,
):
    batch_preds = pl_trainer.predict(model=pl_module, datamodule=pl_data_module)

    # join batch dicts to a single dict
    def _join_batches(k):
        return np.concatenate([d[k] for d in batch_preds], axis=0)

    predictions = {k: _join_batches(k) for k in batch_preds[0].keys()}

    return predictions


def save_embeddings_results(root_dir, results):
    supported_row_names = ["cell_name", "seq_id"]
    index_vals = next(
        (results.get(k) for k in supported_row_names if results.get(k) is not None),
        None,
    )
    embeddings_df = pd.DataFrame(results["embeddings"], index=index_vals)
    embeddings_df.to_csv(Path(root_dir) / "embeddings.csv", header=False)
    logger.info(f"Embeddings saved to {root_dir}/embeddings.csv")


def save_prediction_results(root_dir, label_dict, results):
    str_predictions = {}
    for predictions_name, raw_predictions in results.items():
        if "predictions" not in predictions_name:
            continue
        label_column_name = predictions_name.split("_predictions")[0]
        id_to_name = {v: k for k, v in label_dict[label_column_name].items()}
        if len(id_to_name) > 1:
            str_predictions[label_column_name] = [
                id_to_name[c] for c in raw_predictions
            ]
        else:  # regression case
            str_predictions[label_column_name] = raw_predictions

    pd.DataFrame(str_predictions, index=results["cell_name"]).to_csv(
        f"{root_dir}/predictions.csv"
    )


def save_logits_results(root_dir, label_dict, results):
    str_logits = {}
    str_probabilities = {}

    from scipy.special import softmax

    for logits_name, id_logits in results.items():
        if "logits" not in logits_name:
            continue
        id_softmax = softmax(id_logits, axis=1)
        label_column_name = logits_name.split("_logits")[0]
        if len(label_dict[label_column_name]) == 1:
            continue
        for key, value in label_dict[label_column_name].items():
            str_logits[label_column_name + "_" + key] = id_logits[:, value]
            str_probabilities[label_column_name + "_" + key] = id_softmax[:, value]

    if str_logits:
        pd.DataFrame(str_logits, index=results["cell_name"]).to_csv(
            f"{root_dir}/logits.csv"
        )
        pd.DataFrame(str_probabilities, index=results["cell_name"]).to_csv(
            f"{root_dir}/probabilities.csv"
        )


def prepare_test_metrics_for_logging(
    results: dict[str, list[torch.Tensor]], n: int, ci_method: str
):
    metrics_to_log = []  # this is shown in the Scalars tab
    df_metrics_rows = []  # this is shown in the Plots tab

    for metric_name, metric_value_list in results.items():
        if "confusion_matrix" in metric_name:
            label = metric_name.split("_confusion_matrix")[0]
            metrics_to_log.append(("confusion_matrix", label, metric_value_list[-1]))
            continue
        if len(metric_value_list) > 1:
            metrics_to_log.append(("histogram", metric_name, metric_value_list))
            if "confusion_matrix" in metric_name:
                continue
        else:
            metrics_to_log.append(("single_value", metric_name, metric_value_list[0]))
        metric_values = [val.item() for val in metric_value_list]
        logger.info(f"Calculating {ci_method} for CI for {metric_name}")
        mean_value, lower_ci, upper_ci = calculate_95_ci(
            metric_values, n, ci_method=ci_method
        )
        df_metrics_rows.append(
            {
                "Metric": metric_name,
                "Mean": np.round(mean_value, 2),
                "CI": f"[{np.round(lower_ci, 3)}, {np.round(upper_ci, 3)}]",
            }
        )
    ci_name = ci_method.replace("_", " ")
    df_metrics = pd.DataFrame(
        df_metrics_rows,
        columns=["Metric", "Mean", "CI"],
    )
    df_metrics = df_metrics.rename(
        columns={"CI": f"{ci_name} CI [Lower bound, Upper bound]"}
    )

    logger.info(df_metrics)
    metrics_to_log.append(("table", "metrics_summary", df_metrics))

    return metrics_to_log


def log_test_metrics(
    clearml_logger: clearml.Logger,
    metrics_to_log: list,
    label_dict: dict[str, dict[str, int]],
):
    for log_type, metric_name, data in metrics_to_log:
        if log_type == "confusion_matrix":
            if metric_name == "label_expressions_nonzero":
                logger.warning(
                    "label_expressions_nonzero logging is not supported in test tasks, see issue 982 https://github.ibm.com/BiomedSciAI-Innersource/bmfm-targets/issues/982 "
                )
                continue
            elif label_dict is None:
                logger.warning(
                    "Logging confusion matrix with no label dict not supported from this context."
                )
                continue
            log_confusion_matrix_from_metrics(
                metric_name, data, label_dict, prefix="test"
            )
        elif log_type == "histogram":
            clearml_logger.report_histogram(
                title=metric_name,
                series=metric_name,
                values=data,
            )
        elif log_type == "single_value":
            clearml_logger.report_single_value(metric_name, data)
        elif log_type == "table":
            clearml_logger.report_table(
                title=metric_name, series=metric_name, table_plot=data
            )


def log_confusion_matrix_from_metrics(label, cm, label_dict, prefix=""):
    id_sorted_labels = sorted(label_dict[label].items(), key=lambda x: x[1])
    display_labels = [item[0] for item in id_sorted_labels]
    title = f"{label} prediction"
    log_confusion_matrix_to_clearml(cm, prefix, display_labels, title, iteration=None)


def interpret_run(
    pl_trainer: pl.Trainer,
    task_config: config.InterpretTaskConfig,
    data_module: DataModule,
    model_config: config.model_config.SCModelConfigBase | None = None,
):
    from bmfm_targets.evaluation import interpret

    if task_config.checkpoint is not None and not os.path.isfile(
        task_config.checkpoint
    ):
        task_config.checkpoint = download_ckpt_from_huggingface(task_config.checkpoint)

    if model_config is None:
        module = interpret.SequenceClassificationAttributionModule.load_from_checkpoint(
            task_config.checkpoint,
            tokenizer=data_module.tokenizer,
            modeling_strategy=data_module.collation_strategy,
            attribute_kwargs=task_config.attribute_kwargs,
            attribute_filter=task_config.attribute_filter,
            weights_only=False,
        )
    else:
        module = interpret.SequenceClassificationAttributionModule(
            model_config,
            data_module.tokenizer,
            data_module.label_dict,
            modeling_strategy=data_module.collation_strategy,
            attribute_kwargs=task_config.attribute_kwargs,
            attribute_filter=task_config.attribute_filter,
        )

    data_module.batch_size = 1
    assert pl_trainer.max_epochs == 1
    # inference_mode must be false for this algorithm to work
    assert not pl_trainer.predict_loop.inference_mode
    attributions = pl_trainer.predict(module, datamodule=data_module)
    ofname = Path(task_config.default_root_dir) / "attributions.json"
    interpret.save_sample_attributions(attributions, ofname)


def num_available_cores(verbose: bool = True) -> int | None:
    if hasattr(os, "sched_getaffinity"):
        try:
            ans: int | None = len(os.sched_getaffinity(0))
            if verbose:
                print(
                    f"num_available_cores:: spotted affinity which restricts available cores. Returning {ans} cores"
                )
            return ans
        except Exception:
            pass

    ans = os.cpu_count()
    if verbose:
        print(f"num_available_cores:: Returning {ans} cores")
    return ans


def start_clearml_logger(
    project_name: str | None,
    task_name: str | None,
    task_type: str | TaskTypes = TaskTypes.training,
    tags: Sequence[str] | None = None,
    reuse_last_task_id: bool | str = True,
    continue_last_task: bool | str | int = False,
    output_uri: str | bool | None = None,
    auto_connect_arg_parser: bool | Mapping[str, bool] = True,
    auto_connect_frameworks: bool | Mapping[str, bool] = True,
    auto_resource_monitoring: bool = True,
    auto_connect_streams: bool | Mapping[str, bool] = True,
    deferred_init: bool = False,
) -> None | clearml.Logger:
    """
    Just a fuse function to quickly start the clearml logger. It sets up patches to pytorch lightning logging hooks so it doesn't need to be passed to any lightning logger.
    This function also checks if the NODE_RANK and LOCAL_RANK env variables have been set. In which case clearml will only be initialized on global rank 0.
    For information on all the arguments please see: https://clear.ml/docs/latest/docs/references/sdk/task/ or https://github.com/allegroai/clearml/blob/master/clearml/task.py.

    General Clearml instructions:
    Unless using offline mode, to use clearml, you must first make an account on their website https://app.clear.ml/login?redirect=%2Fsettings%2Fworkspace-configuration.
    Then, you must create a ~/clearml.conf file and specify server address as shown here https://clear.ml/docs/latest/docs/configs/clearml_conf/.
    Otherwise, offline mode instructions can be found here: https://clear.ml/docs/latest/docs/guides/set_offline/

    """
    if should_start_clearml_logger():
        try:
            task = Task.init(
                project_name=project_name,
                task_name=task_name,
                task_type=task_type,
                tags=tags,
                reuse_last_task_id=reuse_last_task_id,
                continue_last_task=continue_last_task,
                output_uri=output_uri,
                auto_connect_arg_parser=auto_connect_arg_parser,
                auto_connect_frameworks=auto_connect_frameworks,
                auto_resource_monitoring=auto_resource_monitoring,
                auto_connect_streams=auto_connect_streams,
                deferred_init=deferred_init,
            )
        except MissingConfigError:
            logger.warning("No clearml config found. Continuing without clearml.")
            return None

        return task.get_logger()


def should_start_clearml_logger():
    # check if we are in a distributed setting (if we are, must check that we are also on global rank 0)
    distributed = ("NODE_RANK" in os.environ) and ("LOCAL_RANK" in os.environ)
    if distributed:
        node_rank = int(os.environ["NODE_RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        if (node_rank == 0) and (local_rank == 0):
            return True
    else:
        # if not in a distributed setting, we can just start logger
        return True
    return False


def convert_task_types(type: str | None):
    if type is None:
        return TaskTypes.training


def find_ckpt(directory, ckpt_type):
    """
    Returns a full path of training checkpoint according tot he user specified checkpoint type: last or best.
    This is relevant when executing testing after training within the same yaml.
    """
    matching_ckpt = [
        ckpt for ckpt in os.listdir(directory) if ckpt_type.lower() in ckpt.lower()
    ]
    matching_ckpt_path = os.path.join(directory, matching_ckpt)
    return matching_ckpt_path


from omegaconf import DictConfig


def update_task_from_trainer(task: DictConfig, trainer: pl.Trainer):
    """
    If running testing after training directly from the same yaml, then the checkpoint need to be saved in a
    clean directory and be passed to the test tasks.
    """
    if config.TestTaskConfig.__name__ in task._target_:
        if task.checkpoint == "last":
            lasts = [
                x.last_model_path
                for x in trainer.checkpoint_callbacks
                if x.last_model_path
            ]
            assert len(lasts) == 1
            task.checkpoint = lasts[0]
        elif task.checkpoint == "best":
            bests = [
                x.best_model_path
                for x in trainer.checkpoint_callbacks
                if x.best_model_path
            ]
            assert len(bests) == 1
            task.checkpoint = bests[0]

    return task
