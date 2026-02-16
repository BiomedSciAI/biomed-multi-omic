import pytest
import torch
from transformers.modeling_outputs import TokenClassifierOutput

import bmfm_targets.training.losses.task
from bmfm_targets.config import FieldInfo, LabelColumnInfo
from bmfm_targets.training import losses
from bmfm_targets.training.losses import (
    CrossEntropyObjective,
    FieldSource,
    IsZeroBCEObjective,
    LabelSource,
    LossTask,
    MSEObjective,
    WCEDFieldSource,
)


@pytest.fixture(scope="module")
def seq_label_field_losses():
    """Create loss tasks using new composition-based architecture."""
    field = FieldInfo(
        field_name="label_expressions",
        vocab_size=55,
        pretrained_embedding=None,
        vocab_update_strategy="static",
        is_masked=False,
        is_input=False,
        decode_modes=["regression", "is_zero"],
        tokenization_strategy="continuous_value_encoder",
        num_special_tokens=5,
        encoder_kwargs={
            "kind": "mlp_with_special_token_embedding",
            "zero_as_special_token": True,
        },
    )

    # Create MSE loss task
    mse_task = LossTask(
        source=FieldSource(
            field_name="label_expressions", decoder_key="label_expressions_regression"
        ),
        objective=MSEObjective(ignore_zero=True),
        weight=1.0,
    )
    # Manually set field for testing (normally done by bind())
    mse_task.source.field = field

    # Create IsZeroBCE loss task
    is_zero_task = LossTask(
        source=FieldSource(
            field_name="label_expressions", decoder_key="label_expressions_is_zero"
        ),
        objective=IsZeroBCEObjective(),
        weight=1.0,
    )
    # Manually set field for testing (normally done by bind())
    is_zero_task.source.field = field

    return [mse_task, is_zero_task]


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
    """Test label loss with link function using new architecture."""
    batch_size, seq_len = 5, 10

    lc = LabelColumnInfo("test", is_regression_label=True)

    # Create task with link function
    lt_link = LossTask(
        source=LabelSource(label_name="test"),
        objective=MSEObjective(link_function="exp"),
        weight=1.0,
    )
    # Manually set label_column for testing (normally done by bind())
    lt_link.source.label_column = lc

    logits = {"test": torch.randn((batch_size, seq_len))}
    labels = {"test": torch.randn((batch_size, seq_len))}

    # Create mock objects
    mock_outputs = type("Outputs", (), {"logits": logits})()
    mock_batch = {"labels": labels}

    loss_link = lt_link.calculate_loss(mock_outputs, mock_batch)
    assert loss_link > 0

    # Create task without link function
    lt_no_link = LossTask(
        source=LabelSource(label_name="test"),
        objective=MSEObjective(),
        weight=1.0,
    )
    # Manually set label_column for testing (normally done by bind())
    lt_no_link.source.label_column = lc

    loss_no_link = lt_no_link.calculate_loss(mock_outputs, mock_batch)
    assert loss_no_link > 0

    # with no training exponentiated logits will be shifted by a lot
    assert loss_link > loss_no_link


def test_can_calculate_seq_lab_loss(seq_label_field_losses, sample_seq_label_outputs):
    outputs, labels = sample_seq_label_outputs
    loss = losses.calculate_losses(seq_label_field_losses, outputs.logits, labels)
    assert loss["loss"] > 0


def test_no_loss_gives_zero(seq_label_field_losses, sample_seq_label_outputs):
    outputs, labels = sample_seq_label_outputs
    dummy_labels = labels.copy()
    dummy_labels["label_expressions"] = -100 + 0 * dummy_labels["label_expressions"]
    loss = losses.calculate_losses(seq_label_field_losses, outputs.logits, dummy_labels)
    assert loss["loss"] == 0


def test_can_calculate_seq_lab_predictions(
    seq_label_field_losses, sample_seq_label_outputs
):
    outputs, labels = sample_seq_label_outputs
    predictions = losses.calculate_predictions(seq_label_field_losses, outputs.logits)
    # Predictions have shape [batch, seq, 1], squeeze to [batch, seq]
    pred_tensor = predictions["label_expressions"].squeeze(-1)
    batch_size, seq_len = pred_tensor.shape
    should_be_zero = outputs.logits["label_expressions_is_zero"].squeeze(-1) > 0.5
    assert (pred_tensor[should_be_zero] == 0).all()

    # Test with MSE only
    mse_only = [
        l for l in seq_label_field_losses if isinstance(l.objective, MSEObjective)
    ]
    predictions = losses.calculate_predictions(mse_only, outputs.logits)
    pred_tensor = predictions["label_expressions"].squeeze(-1)
    assert not (pred_tensor == 0).any()


def test_nan_loss_gives_valid_zero():
    """Test that all-ignored labels give zero loss."""
    logits = {"test_token_scores": torch.randn(10, 10)}
    labels = {"test": torch.tensor([-100 for i in range(10)])}

    field = FieldInfo(field_name="test", vocab_size=10)
    lt = LossTask(
        source=FieldSource(field_name="test", decoder_key="test_token_scores"),
        objective=CrossEntropyObjective(),
        weight=1.0,
    )
    # Manually set field for testing (normally done by bind())
    lt.source.field = field
    # Bind objective to output size
    lt.objective.bind(field.vocab_size)

    loss = losses.calculate_losses([lt], logits, labels)
    assert loss["loss"] == 0


def test_mse_loss_on_token_scores_output():
    logits = {"test_token_scores": torch.ones(10, 10) * 0.5}
    labels = {"test": torch.ones(10, 10) * (-100)}
    for i in range(10):
        labels["test"][i, i] = 1
    from bmfm_targets.training.metrics import mse_loss

    input_logits = logits["test_token_scores"]
    input_labels = labels["test"]

    loss = mse_loss(input_logits, input_labels)
    expected_loss = torch.tensor((0.5) ** 2, dtype=loss.dtype)
    torch.testing.assert_close(loss, expected_loss)

    for i in range(10):
        input_labels[i, 0] = 2

    loss = mse_loss(input_logits, input_labels)
    assert loss > torch.tensor((0.5) ** 2)


def test_mae_loss_on_token_scores_output():
    logits = {"test_token_scores": torch.ones(10, 10) * 0.5}
    labels = {"test": torch.ones(10, 10) * (-100)}
    for i in range(10):
        labels["test"][i, i] = 1
    from bmfm_targets.training.metrics import mae_loss

    input_logits = logits["test_token_scores"]
    input_labels = labels["test"]

    loss = mae_loss(input_logits, input_labels)
    expected_loss = torch.tensor((0.5), dtype=loss.dtype)
    torch.testing.assert_close(loss, expected_loss)

    for i in range(10):
        input_labels[i, 0] = 2

    loss = mae_loss(input_logits, input_labels)
    assert loss > torch.tensor(0.5)


def test_wced_loss_populates_fields_correctly():
    """Test WCED loss configuration using new composition architecture."""
    field = FieldInfo(
        "expressions",
        decode_modes={
            "wced": {"vocab_field": "genes", "logit_outputs": ["mse", "is_zero_bce"]}
        },
    )

    # Test non_input_genes with is_zero_bce (index 1)
    lt = LossTask(
        source=WCEDFieldSource(
            field_name="expressions",
            wced_target="non_input_genes",
        ),
        objective=IsZeroBCEObjective(),
        weight=1.0,
    )
    # Manually set field (normally done by resolve_schema)
    lt.source.field = field
    # Manually set WCED attributes (normally done by resolve_schema)
    lt.source.label_set = "non_input"
    # Manually set decoder_output_index based on is_zero_bce position
    lt.source.decoder_output_index = (
        bmfm_targets.training.losses.task.lookup_wced_output_index("is_zero_bce", field)
    )

    assert lt.source.decoder_key == "expressions_wced"
    assert lt.source.wced_target == "non_input_genes"
    assert lt.source.label_set == "non_input"
    assert lt.source.decoder_output_index == 1

    # Test input_genes with mse (index 0)
    lt = LossTask(
        source=WCEDFieldSource(
            field_name="expressions",
            wced_target="input_genes",
        ),
        objective=MSEObjective(),
        weight=1.0,
    )
    lt.source.field = field
    lt.source.label_set = "input"
    lt.source.decoder_output_index = (
        bmfm_targets.training.losses.task.lookup_wced_output_index("mse", field)
    )

    assert lt.source.decoder_key == "expressions_wced"
    assert lt.source.wced_target == "input_genes"
    assert lt.source.label_set == "input"
    assert lt.source.decoder_output_index == 0

    # Test all_genes with mse (index 0)
    lt = LossTask(
        source=WCEDFieldSource(
            field_name="expressions",
            wced_target="all_genes",
        ),
        objective=MSEObjective(),
        weight=1.0,
    )
    lt.source.field = field
    lt.source.label_set = "all"
    lt.source.decoder_output_index = (
        bmfm_targets.training.losses.task.lookup_wced_output_index("mse", field)
    )

    assert lt.source.decoder_key == "expressions_wced"
    assert lt.source.wced_target == "all_genes"
    assert lt.source.label_set == "all"
    assert lt.source.decoder_output_index == 0


def test_lookup_wced_output_index():
    field = FieldInfo(
        "expressions",
        decode_modes={
            "wced": {"vocab_field": "genes", "logit_outputs": ["mse", "is_zero_bce"]}
        },
    )
    assert bmfm_targets.training.losses.task.lookup_wced_output_index("mse", field) == 0
    assert (
        bmfm_targets.training.losses.task.lookup_wced_output_index("is_zero_bce", field)
        == 1
    )
    # Loss not in logit_outputs should return None (not raise)
    result = bmfm_targets.training.losses.task.lookup_wced_output_index(
        "cross_entropy", field
    )
    assert result is None

    field = FieldInfo("expressions", decode_modes={"regression": {}})
    # This should return None, not raise ValueError
    result = bmfm_targets.training.losses.task.lookup_wced_output_index("mse", field)
    assert result is None
    field = FieldInfo(
        "expressions",
        decode_modes={"wced": {"vocab_field": "genes", "logit_outputs": ["mse"]}},
    )
    assert (
        bmfm_targets.training.losses.task.lookup_wced_output_index("mse", field) is None
    )


def test_wced_mse_and_is_zero_focal():
    """Test WCED with MSE and IsZeroFocal using new composition architecture."""
    from bmfm_targets.training.losses.objectives import IsZeroFocalObjective

    field = FieldInfo(
        "expressions",
        decode_modes={
            "wced": {"vocab_field": "genes", "logit_outputs": ["mse", "is_zero_focal"]}
        },
    )
    batch_size, seq_len, vocab_size = 10, 16, 12
    output_dim = len(field.decode_modes["wced"]["logit_outputs"])
    logits = {
        "expressions_wced": torch.rand(batch_size, seq_len, vocab_size, output_dim)
    }
    labels = {"expressions": {"input": torch.ones(batch_size, vocab_size) * (-100)}}
    for i in range(batch_size):
        non_ignore_indices = torch.randperm(vocab_size)[: vocab_size // 2]
        labels["expressions"]["input"][i, non_ignore_indices] = torch.rand(
            vocab_size // 2
        )

    # Test MSE with input_genes (index 0)
    lt = LossTask(
        source=WCEDFieldSource(
            field_name="expressions",
            wced_target="input_genes",
        ),
        objective=MSEObjective(),
        weight=1.0,
    )
    lt.source.field = field
    lt.source.label_set = "input"
    lt.source.decoder_output_index = (
        bmfm_targets.training.losses.task.lookup_wced_output_index("mse", field)
    )

    assert lt.source.decoder_key == "expressions_wced"
    assert lt.source.wced_target == "input_genes"
    assert lt.source.label_set == "input"
    assert lt.source.decoder_output_index == 0

    # Create mock objects for calculate_loss
    mock_outputs = type("Outputs", (), {"logits": logits})()
    mock_batch = {"labels": labels}

    loss = lt.calculate_loss(mock_outputs, mock_batch)
    assert not torch.isnan(loss)
    assert loss > 0

    # Test IsZeroFocal with input_genes (index 1)
    lt = LossTask(
        source=WCEDFieldSource(
            field_name="expressions",
            wced_target="input_genes",
        ),
        objective=IsZeroFocalObjective(focal_gamma=3),
        weight=1.0,
    )
    lt.source.field = field
    lt.source.label_set = "input"
    lt.source.decoder_output_index = (
        bmfm_targets.training.losses.task.lookup_wced_output_index(
            "is_zero_focal", field
        )
    )

    assert lt.source.decoder_key == "expressions_wced"
    assert lt.source.wced_target == "input_genes"
    assert lt.source.label_set == "input"
    assert lt.objective.focal_gamma == 3
    assert lt.source.decoder_output_index == 1

    loss = lt.calculate_loss(mock_outputs, mock_batch)
    assert not torch.isnan(loss)
    assert loss > 0


# Made with Bob
