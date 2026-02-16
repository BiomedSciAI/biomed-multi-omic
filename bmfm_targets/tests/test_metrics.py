from functools import partial

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bmfm_targets.config import FieldInfo, LabelColumnInfo, SCBertConfig, TrainerConfig
from bmfm_targets.tests.helpers import TorchListDataset
from bmfm_targets.tokenization import get_gene2vec_tokenizer
from bmfm_targets.training.losses import (
    CrossEntropyObjective,
    LabelSource,
    LossTask,
    MSEObjective,
    adapt_hce_prediction_and_labels_to_metrics_entries,
)
from bmfm_targets.training.metrics import (
    get_metric_object,
    metric_functions,
)
from bmfm_targets.training.modules.base import BaseTrainingModule


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("sequence_len", [64, 128])
def test_token_value_loss(batch_size, sequence_len):
    tokenizer = get_gene2vec_tokenizer()
    token_values = tokenizer.get_token_values("expressions")
    loss_fct = metric_functions.TokenValueLoss(token_values)
    vocab_size = tokenizer.field_vocab_size("expressions")
    logits_size = (batch_size, sequence_len, vocab_size)
    labels_size = (batch_size, sequence_len)

    all_tokens = tokenizer.get_field_tokenizer("expressions").get_vocab()
    non_special_ids = [
        v for t, v in all_tokens.items() if t not in tokenizer.all_special_tokens
    ]

    test_labels = torch.randint(
        0,
        len(non_special_ids) - 1,
        labels_size,
        generator=torch.random.manual_seed(42),
    ).apply_(lambda x: non_special_ids[x])
    one_hot_labels = F.one_hot(test_labels, num_classes=vocab_size).float()

    # Add controlled noise to the one-hot vectors (adjust the noise_factor)
    large_number = 10
    losses = []
    for noise_factor in [0.01, 1, 10]:
        test_logits = large_number * one_hot_labels + noise_factor * torch.randn_like(
            one_hot_labels,
        )

        assert test_logits.shape == logits_size
        loss = loss_fct(test_logits, test_labels)
        assert loss is not None
        losses.append(loss)
    loss_diff = np.diff(losses)
    # as the noise around the ground truth increases, the loss increases
    assert (loss_diff > 0).all()


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("sequence_len", [64, 128])
def test_mse_loss(batch_size, sequence_len):
    tokenizer = get_gene2vec_tokenizer()
    logit_special_token_mask = tokenizer.get_special_token_binary_mask("expressions")

    loss_fct = partial(F.mse_loss, reduction="none")
    vocab_size = tokenizer.field_vocab_size("expressions")
    reg_logits_size = (batch_size, sequence_len)
    labels_size = (batch_size, sequence_len)

    test_labels = torch.randint(
        len(tokenizer.all_special_tokens),
        vocab_size - 1,
        labels_size,
        generator=torch.random.manual_seed(42),
    )

    # Add controlled noise to the one-hot vectors (adjust the noise_factor)
    large_number = 10
    losses = []
    for noise_factor in [0.01, 1, 10]:
        test_reg_logits = (
            torch.randn_like(test_labels.float()) * noise_factor + test_labels
        )

        assert test_reg_logits.shape == reg_logits_size
        loss = loss_fct(test_reg_logits, test_labels).mean()
        assert loss is not None
        losses.append(loss)
    loss_diff = np.diff(losses)
    # as the noise around the ground truth increases, the loss increases
    assert (loss_diff > 0).all()


class MetricsTestingTrainingModule(BaseTrainingModule):
    MODELING_STRATEGY = "multitask"

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        outputs = batch["outputs"]
        batch_size = len(labels)
        step_metrics = self.split_metrics("train")(outputs, labels)
        self.log_metrics(step_metrics, "train", batch_size)
        return None

    def validation_step(self, batch, batch_idx):
        labels = batch["labels"]
        outputs = batch["outputs"]
        self.split_metrics("validation")(outputs, labels)


def test_metrics_aggregate_correctly():
    model_config = SCBertConfig(
        fields=[FieldInfo("dummy", 123)],
        label_columns=[
            LabelColumnInfo(label_column_name="categorical", n_unique_values=3),
            LabelColumnInfo(label_column_name="regression", n_unique_values=1),
        ],
        hidden_size=32,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
    )

    # Create LossTask objects using new API
    categorical_task = LossTask(
        source=LabelSource(label_name="categorical"),
        objective=CrossEntropyObjective(),
        weight=1.0,
        metrics=[{"name": "accuracy"}, {"name": "f1"}],
    )
    categorical_task.bind(
        fields=model_config.fields,
        label_columns=model_config.label_columns,
        tokenizer=None,
    )

    regression_task = LossTask(
        source=LabelSource(label_name="regression"),
        objective=MSEObjective(),
        weight=1.0,
        metrics=[{"name": "mse"}],
    )
    regression_task.bind(
        fields=model_config.fields,
        label_columns=model_config.label_columns,
        tokenizer=None,
    )

    trainer_config = TrainerConfig(losses=[categorical_task, regression_task])
    module = MetricsTestingTrainingModule(model_config, trainer_config)

    batch_size = 5
    n_batches = 10

    def generate_data_point(idx):
        gt = {"categorical": idx % 3, "regression": 0.1 + idx % 2}
        pred = {"categorical": 1, "regression": 0.1}
        return {"labels": gt, "outputs": pred}

    data_list = [generate_data_point(i) for i in range(batch_size * n_batches)]
    ds = TorchListDataset(data_list)
    dl = DataLoader(ds, batch_size=batch_size)
    val_dl = DataLoader(ds, batch_size=batch_size)
    trainer = pl.Trainer(max_epochs=1, accelerator="cpu", val_check_interval=1.0)
    trainer.fit(module, train_dataloaders=dl, val_dataloaders=val_dl)

    outputs_c = torch.Tensor([item["outputs"]["categorical"] for item in data_list])
    labels_c = torch.Tensor([item["labels"]["categorical"] for item in data_list])
    f1 = get_metric_object({"name": "f1"}, num_classes=3)
    acc = get_metric_object({"name": "accuracy"}, num_classes=3)

    full_epoch_f1 = f1(outputs_c, labels_c)
    full_epoch_acc = acc(outputs_c, labels_c)

    outputs_r = torch.Tensor([item["outputs"]["regression"] for item in data_list])
    labels_r = torch.Tensor([item["labels"]["regression"] for item in data_list])

    mse = get_metric_object({"name": "mse"}, num_classes=1)
    full_epoch_mse = mse(outputs_r, labels_r)

    # assert trainer.logged_metrics["train/categorical_f1_epoch"] == full_epoch_f1
    assert trainer.logged_metrics["validation/categorical_f1"] == full_epoch_f1

    # assert trainer.logged_metrics["train/categorical_accuracy_epoch"] == full_epoch_acc
    assert trainer.logged_metrics["validation/categorical_accuracy"] == full_epoch_acc

    # assert trainer.logged_metrics["train/regression_mse_epoch"] == full_epoch_mse
    assert trainer.logged_metrics["validation/regression_mse"] == full_epoch_mse


def test_mse_loss_does_not_return_nan():
    logits = torch.tensor([0.1, 0.2, 0.3, 0.4])
    labels = torch.tensor([-100, -100, -100, -100])
    loss = metric_functions.mse_loss(logits, labels)
    assert not torch.isnan(loss)


def test_mae_loss_does_not_return_nan():
    logits = torch.tensor([0.1, 0.2, 0.3, 0.4])
    labels = torch.tensor([-100, -100, -100, -100])
    loss = metric_functions.mae_loss(logits, labels)
    assert not torch.isnan(loss)


def test_classification_loss_with_ignore_zero():
    ignore_zero = True
    problem_type = "regression"
    loss_name = "mse"
    output_size = 1
    logits = torch.tensor([0.0418, 0.0552, 0.0690, 0.0362, 0.0287])
    labels = torch.tensor([2.4572, 2.4550, 2.5594, 0.0000, 2.7735])
    loss_ignoreZero = metric_functions.classification_loss(
        logits, labels, loss_name, output_size, problem_type, ignore_zero
    )

    ignore_zero = False
    loss = metric_functions.classification_loss(
        logits, labels, loss_name, output_size, problem_type, ignore_zero
    )
    assert not torch.isnan(loss_ignoreZero)
    assert not loss_ignoreZero == torch.tensor(6.3323)
    assert not torch.isnan(loss)
    assert not loss == torch.tensor(5.0661)


def test_hce_classification_loss():
    loss_name = "hce"
    output_size = 1
    flag = 1.0
    logits = torch.tensor([1000.0, 1000.0, -1000.0, -1000.0, -1000.0])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, flag])
    hce_loss = metric_functions.classification_loss(
        logits, labels, loss_name, output_size
    )
    assert not torch.isnan(hce_loss)
    assert pytest.approx(hce_loss) == 0.0

    logits = torch.tensor([1000.0, 0.0, -1000.0, -1000.0, -1000.0])
    hce_loss = metric_functions.classification_loss(
        logits, labels, loss_name, output_size
    )
    assert not torch.isnan(hce_loss)
    assert pytest.approx(hce_loss) == 0.0

    logits = torch.tensor([1000.0, 0.0, 1000.0, -1000.0, -1000.0])
    hce_loss = metric_functions.classification_loss(
        logits, labels, loss_name, output_size
    )
    assert not torch.isnan(hce_loss)
    assert torch.all(hce_loss > 0.0)

    flag = 0.0
    labels = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, flag])
    hce_loss = metric_functions.classification_loss(
        logits, labels, loss_name, output_size
    )
    assert not torch.isnan(hce_loss)
    assert pytest.approx(hce_loss) == 0.0


def test_single_label_classification_loss():
    problem_type = "single_label_classification"
    loss_name = "cross_entropy"
    output_size = 1
    logits = torch.tensor([0.0418, 0.0552])
    labels = torch.tensor([0, 0])

    loss = metric_functions.classification_loss(
        logits, labels, loss_name, output_size, problem_type
    )
    assert not torch.isnan(loss)


def test_adapt_hce_prediction_and_labels_to_metrics_entries():
    logits = torch.tensor(
        [
            [10.0, 20.0, 5.0],
            [4.0, 1.0, 8.0],
            [9.0, 1.0, 8.0],
        ]
    )
    labels = torch.tensor(
        [
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    target_pred = torch.tensor([1, 2, 0])
    pred, new_labels = adapt_hce_prediction_and_labels_to_metrics_entries(
        logits, labels
    )
    assert new_labels[0] in (0, 2)
    assert new_labels[1] == 2
    assert new_labels[2] == -100
