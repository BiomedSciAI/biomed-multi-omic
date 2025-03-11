from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytorch_lightning.callbacks


def default_callbacks():
    return [
        # pytorch_lightning.callbacks.RichProgressBar(console_kwargs={"stderr": True}),
        pytorch_lightning.callbacks.LearningRateMonitor("step"),
    ]


@dataclass
class BaseTaskConfig:
    # all tasks feed args to pl.Trainer so all tasks support these in principle
    accelerator: str = "auto"
    strategy: str = "auto"
    devices: str | int = "auto"
    num_nodes: int = 1
    precision: str = "16-mixed"
    fast_dev_run: int | bool = False
    max_epochs: int | None = None
    min_epochs: int | None = None
    max_steps: int = -1
    min_steps: int | None = None
    max_time: str | None = None
    limit_train_batches: int | float | None = None
    limit_val_batches: int | float | None = None
    limit_test_batches: int | float | None = None
    limit_predict_batches: int | float | None = None
    overfit_batches: int | float = 0
    val_check_interval: int | float | None = None
    check_val_every_n_epoch: int | None = 1
    num_sanity_val_steps: int | None = None
    log_every_n_steps: int | None = None
    enable_checkpointing: bool | None = None
    enable_progress_bar: bool | None = None
    enable_model_summary: bool | None = None
    accumulate_grad_batches: int = 1
    gradient_clip_val: int | float | None = None
    gradient_clip_algorithm: str | None = None
    deterministic: bool | str | None = None
    benchmark: bool | None = None
    inference_mode: bool = True
    use_distributed_sampler: bool = True
    profiler: str | None = None
    detect_anomaly: bool = False
    barebones: bool = False
    sync_batchnorm: bool = False
    reload_dataloaders_every_n_epochs: int = 0
    default_root_dir: str | Path | None = "/dccstor/bmfm-targets/scbert-pretrained"
    # # permitted nonNone values of tf32_mode:  "highest","high","medium"
    tf32_mode: str | None = None


@dataclass
class TrainingTaskConfig(BaseTaskConfig):
    max_epochs: int = 20
    max_steps: int = -1  # -1 means no limit
    precision: str = "16-mixed"
    val_check_interval: float | None = 0.05
    accelerator: str = "auto"
    callbacks: list[Any] = field(default_factory=default_callbacks)
    accumulate_grad_batches: int = 1
    freeze_encoder: bool = False
    gradient_clip_val: float | int | None = None
    num_sanity_val_steps: int = 0
    # whether to attempt to continue training using pl.Trainer.fit(..., ckpt_path=...)
    # see https://github.com/Lightning-AI/pytorch-lightning/issues/13246
    resume_training_from_ckpt: bool = False
    checkpoints_every_n_train_steps: int | None = None
    setup_stage = "fit"


@dataclass
class PredictTaskConfig(BaseTaskConfig):
    callbacks: list[Any] = field(default_factory=default_callbacks)
    checkpoint: str | None = None
    output_predictions: bool = True
    output_embeddings: bool = False
    setup_stage = "predict"


@dataclass
class TestTaskConfig(BaseTaskConfig):
    callbacks: list[Any] = field(default_factory=default_callbacks)
    # if num_botstrap_runs = 0 no resampling will occur
    num_bootstrap_runs: int = 0
    ci_method: str = "binomial"
    checkpoint: str | None = None
    setup_stage = "test"


@dataclass
class InterpretTaskConfig(BaseTaskConfig):
    max_epochs: int = 1
    inference_mode: bool = False
    checkpoint: str | None = None
    # kwargs to pass to LayerIntegratedGradients.attribute
    # see https://captum.ai/api/layer.html#layer-integrated-gradients for valid options
    attribute_kwargs: dict | None = None
    attribute_filter: dict | None = None
    setup_stage = "predict"


@dataclass
class TrainerConfig:
    """
    Configuration for training.

    Args:
    ----
        batch_size: The batch size to use for training.
        betas: The beta values for the AdamW optimizer.
        epsilon: The epsilon value for the AdamW optimizer.
        learning_rate: The learning rate to use for the AdamW optimizer.
        lr_decay_steps: The number of steps after which to decay the learning rate.
            If None, no learning rate decay is used.
        warmup_steps: The number of steps to use for warmup. If None, no warmup is used.
        weight_decay: The weight decay value to use for the AdamW optimizer.
            If None, no weight decay is used.
        losses: list of dicts with keys "name" and optionally "weight". Supported names
            are "cross_entropy", "focal" and "token_value". Weights are used for a simple
            weighted sum and will be normalized. If no weights are supplied, losses
            are equally weighted. Default is equally weighted both losses. Note that
            token_value cannot be calculated for gene masking, and this component
            will be skipped.
            If the loss name is "cross_entropy", an additional argument is supported:
            label_smoothing (float) defaults to 0.01, see
            https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
            If the loss name is "focal", an additional argument is supported:
            focal_gamma (float) defaults to 2.0
        pooling_method: which pooling method to use to generate embeddings (if requested)
            the options are "pooling_layer", "first_token" or "mean_pooling".
            Note that "pooling_layer" will only give meaningful output with sequence
            classification.
        metrics: list of dicts with keys "name" and other optional parameters for each metric
        batch_prediction_behavior (str|int|None): whether to track batch_predictions, track and dump or
            do not track at all.
              "dump" - dump every batch to disk (uses lots of hd space and lots of memory)
              "track" - tracks the batches for metric calculations if defined (uses lots of memory)
              integer - tracks the n most recent batches for track or dump (to control the memory usage)
              null/None - does not track at all. No extra memory usage, nothing on disk.
            Rich metrics require batch_predictions but it may be too much memory for a given run.
            Default is not to track at all.
    """

    batch_size: int = 64
    betas: tuple[float, float] = (0.9, 0.99)
    epsilon: float = 1e-8
    learning_rate: float = 1e-4
    losses: list[dict] = field(default_factory=list)
    lr_decay_steps: int | None = None
    warmup_steps: int = 0
    weight_decay: float | None = None
    metrics: list[dict] | None = None
    pooling_method: str = "pooling_layer"
    batch_prediction_behavior: str | int | None = None
