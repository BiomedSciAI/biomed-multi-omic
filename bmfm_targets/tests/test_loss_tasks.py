import pytest
import torch
from transformers.modeling_outputs import TokenClassifierOutput

from bmfm_targets.config import FieldInfo, LabelColumnInfo
from bmfm_targets.training.metrics import loss_handling


@pytest.fixture(scope="module")
def seq_label_field_losses():
    return [
        loss_handling.FieldLossTask(
            loss_name="mse",
            weight=1,
            field=FieldInfo(
                field_name="label_expressions",
                vocab_size=55,
                pretrained_embedding=None,
                vocab_update_strategy="static",
                is_masked=False,
                is_input=False,
                decode_modes=["regression", "is_zero"],
                tokenization_strategy="continuous_value_encoder",
                num_special_tokens=5,
                continuous_value_encoder_kwargs={
                    "kind": "mlp_with_special_token_embedding",
                    "zero_as_special_token": True,
                },
            ),
            ignore_zero=True,
            link_function=None,
        ),
        loss_handling.FieldLossTask(
            loss_name="is_zero_bce",
            weight=1,
            field=FieldInfo(
                field_name="label_expressions",
                vocab_size=55,
                pretrained_embedding=None,
                vocab_update_strategy="static",
                is_masked=False,
                is_input=False,
                decode_modes=["regression", "is_zero"],
                tokenization_strategy="continuous_value_encoder",
                num_special_tokens=5,
                continuous_value_encoder_kwargs={
                    "kind": "mlp_with_special_token_embedding",
                    "zero_as_special_token": True,
                },
            ),
            ignore_zero=True,
            link_function=None,
        ),
    ]


@pytest.fixture(scope="module")
def sample_seq_label_outputs():
    batch_size = 3
    seq_len = 128
    logits_dict = {
        "label_expressions_regression": torch.randn((batch_size, seq_len, 1)) + 5,
        "label_expressions_is_zero": torch.randn((batch_size, seq_len, 1)) * 0.1 + 0.5,
    }

    labels = torch.randn((batch_size, seq_len)) + 5
    labels[:, 0] = -100  # CLS
    labels[:, -1] = -100  # PAD

    labels_dict = {"label_expressions": labels}
    return TokenClassifierOutput(logits=logits_dict), labels_dict


def test_link_function_label_loss():
    batch_size, seq_len = 5, 10
    logits = torch.randn((batch_size, seq_len))
    labels = torch.randn((batch_size, seq_len))
    lc = LabelColumnInfo("test", is_regression_label=True)
    lt_link = loss_handling.LabelLossTask(
        label_column=lc, loss_name="mse", link_function="exp"
    )
    loss_link = lt_link.calculate_loss(logits, labels)
    assert loss_link > 0

    lt_no_link = loss_handling.LabelLossTask(label_column=lc, loss_name="mse")
    loss_no_link = lt_no_link.calculate_loss(logits, labels)
    assert loss_no_link > 0

    # with no training exponentiated logits will be shifted by a lot
    assert loss_link > loss_no_link


def test_can_calculate_seq_lab_loss(seq_label_field_losses, sample_seq_label_outputs):
    outputs, labels = sample_seq_label_outputs
    loss = loss_handling.calculate_losses(
        seq_label_field_losses, outputs.logits, labels
    )
    assert loss["loss"] > 0


def test_no_loss_gives_zero(seq_label_field_losses, sample_seq_label_outputs):
    outputs, labels = sample_seq_label_outputs
    dummy_labels = labels.copy()
    dummy_labels["label_expressions"] = -100 + 0 * dummy_labels["label_expressions"]
    loss = loss_handling.calculate_losses(
        seq_label_field_losses, outputs.logits, dummy_labels
    )
    assert loss["loss"] == 0


def test_can_calculate_seq_lab_predictions(
    seq_label_field_losses, sample_seq_label_outputs
):
    outputs, labels = sample_seq_label_outputs
    predictions = loss_handling.calculate_predictions(
        seq_label_field_losses, outputs.logits
    )
    batch_size, seq_len = predictions["label_expressions"].shape
    should_be_zero = outputs["logits"]["label_expressions_is_zero"] > 0.5
    assert (
        predictions["label_expressions"][should_be_zero.view(batch_size, seq_len)] == 0
    ).all()

    mse_only = [l for l in seq_label_field_losses if l.loss_name == "mse"]
    predictions = loss_handling.calculate_predictions(mse_only, outputs.logits)
    assert not (predictions["label_expressions"] == 0).any()


def test_nan_loss_gives_valid_zero():
    logits = {"test_token_scores": torch.randn(10, 10)}
    labels = {"test": torch.tensor([-100 for i in range(10)])}
    lt = loss_handling.FieldLossTask(
        field=FieldInfo(field_name="test", vocab_size=10),
        loss_name="cross_entropy",
    )

    loss = loss_handling.calculate_losses([lt], logits, labels)
    assert loss["loss"] == 0
