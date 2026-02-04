import tempfile
from pathlib import Path

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from bmfm_targets import config
from bmfm_targets.models.predictive import (
    scbert,
    scmodernbert,
    scnystromformer,
)
from bmfm_targets.models.predictive.llama import (
    LlamaForMaskedLMConfig,
    LlamaForMultiTaskConfig,
)
from bmfm_targets.models.predictive.scnystromformer.modeling_scnystromformer import (
    SCNystromformerSelfAttention,
)
from bmfm_targets.tests import helpers
from bmfm_targets.tokenization import MultiFieldCollator
from bmfm_targets.training.losses import (
    CrossEntropyObjective,
    FieldSource,
    FocalObjective,
    LossTask,
    MSEObjective,
    TokenValueObjective,
    WCEDFieldSource,
)
from bmfm_targets.training.losses.objectives import IsZeroBCEObjective
from bmfm_targets.training.masking import Masker
from bmfm_targets.training.metrics.metric_functions import (
    ce_loss,
    focal_loss,
    mse_loss,
    token_value_loss,
)
from bmfm_targets.training.modules import MLMTrainingModule


def test_nystromformer_forward():
    """Test Nystromformer forward pass with cross-entropy and focal loss objectives."""
    vocab_size = 100
    batch_size = 3
    sequence_len = 8
    fields = [
        config.FieldInfo("genes", vocab_size),
        config.FieldInfo(
            "expressions", vocab_size, is_masked=True, decode_modes={"token_scores": {}}
        ),
    ]

    model_config = config.SCNystromformerConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_landmarks=2,
        fields=fields,
        max_position_embeddings=sequence_len,
    )
    model = scnystromformer.SCNystromformerForMaskedLM(model_config)

    input_ids = torch.randint(0, 99, (batch_size, 2, sequence_len))
    attention_mask = torch.ones((batch_size, sequence_len))
    labels = torch.randint(0, 99, (batch_size, 1, sequence_len))

    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

    # Test cross-entropy loss
    ce_objective = CrossEntropyObjective(label_smoothing=0.01)
    ce_objective.bind(vocab_size)
    total_ce_loss = ce_objective.compute(
        outputs.logits["expressions_token_scores"], labels[:, 0]
    )
    assert not total_ce_loss.isinf()
    assert total_ce_loss > 0
    assert tuple(outputs.logits["expressions_token_scores"].shape) == (
        batch_size,
        sequence_len,
        vocab_size,
    )

    # Test focal loss
    focal_objective = FocalObjective()
    focal_objective.bind(vocab_size)
    total_focal_loss = focal_objective.compute(
        outputs.logits["expressions_token_scores"], labels[:, 0]
    )
    assert not total_focal_loss.isinf()
    assert total_focal_loss > 0


def test_scbert_forward():
    """Test SCBert forward pass with multiple loss objectives."""
    vocab_size = 100
    batch_size = 3
    sequence_len = 10
    fields = [
        config.FieldInfo("genes", vocab_size),
        config.FieldInfo(
            "expressions", vocab_size, is_masked=True, decode_modes={"token_scores": {}}
        ),
    ]

    model_config = config.SCBertConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        fields=fields,
    )
    model = scbert.SCBertForMaskedLM(model_config)

    input_ids = torch.randint(0, 99, (batch_size, 2, sequence_len))
    attention_mask = torch.ones((batch_size, sequence_len))
    labels = torch.randint(0, 99, (batch_size, 1, sequence_len))

    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

    # Test cross-entropy loss
    ce_objective = CrossEntropyObjective(label_smoothing=0.01)
    ce_objective.bind(vocab_size)
    total_ce_loss = ce_objective.compute(
        outputs.logits["expressions_token_scores"], labels[:, 0]
    )
    assert not total_ce_loss.isinf()
    assert total_ce_loss > 0

    # Test focal loss
    focal_objective = FocalObjective()
    focal_objective.bind(vocab_size)
    total_focal_loss = focal_objective.compute(
        outputs.logits["expressions_token_scores"], labels[:, 0]
    )
    assert not total_focal_loss.isinf()
    assert total_focal_loss > 0

    # Test token value loss
    token_value_objective = TokenValueObjective()
    # For testing, we can skip the tokenizer requirement by directly setting token_values
    token_value_objective.output_size = vocab_size
    token_value_objective.set_token_values(list(np.arange(vocab_size, dtype=float)))
    total_token_value_loss = token_value_objective.compute(
        outputs.logits["expressions_token_scores"], labels[:, 0]
    )
    assert not total_token_value_loss.isinf()
    assert total_token_value_loss > 0

    # Test combined loss and backward pass
    loss = total_ce_loss + total_token_value_loss
    assert loss is not None
    loss.backward()

    assert tuple(outputs.logits["expressions_token_scores"].shape) == (
        batch_size,
        sequence_len,
        vocab_size,
    )


def test_scbert_multitask_forward():
    """Test SCBert multitask forward pass with cross-entropy loss on MLM and labels."""
    vocab_size = 100
    batch_size = 3
    sequence_len = 10
    output_size = 10
    label_columns = [
        config.LabelColumnInfo(
            label_column_name="cell_type", n_unique_values=output_size
        )
    ]
    fields = [
        config.FieldInfo("genes", vocab_size),
        config.FieldInfo(
            "expressions", vocab_size, is_masked=True, decode_modes={"token_scores": {}}
        ),
    ]

    model_config = config.SCBertConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        fields=fields,
        label_columns=label_columns,
    )
    model = scbert.SCBertForMultiTaskModeling(model_config)

    input_ids = torch.randint(0, 99, (batch_size, 2, sequence_len))
    attention_mask = torch.ones((batch_size, sequence_len))
    mlm_labels = torch.randint(0, 99, (batch_size, 1, sequence_len))
    cell_type_labels = torch.randint(0, output_size, (batch_size,))

    outputs = model(input_ids, attention_mask=attention_mask)

    # Test MLM cross-entropy loss
    mlm_ce_objective = CrossEntropyObjective(label_smoothing=0.01)
    mlm_ce_objective.bind(vocab_size)
    mlm_loss = mlm_ce_objective.compute(
        outputs.logits["expressions_token_scores"], mlm_labels[:, 0]
    )

    # Test label cross-entropy loss
    label_ce_objective = CrossEntropyObjective(label_smoothing=0.01)
    label_ce_objective.bind(output_size)
    label_loss = label_ce_objective.compute(
        outputs.logits["cell_type"], cell_type_labels
    )

    total_ce_loss = mlm_loss + label_loss
    assert not total_ce_loss.isinf()
    assert total_ce_loss > 0
    total_ce_loss.backward()


def test_focal_loss_scbert_multitask_forward():
    """Test SCBert multitask forward pass with focal loss on MLM and labels."""
    vocab_size = 100
    batch_size = 3
    sequence_len = 10
    output_size = 10
    label_columns = [
        config.LabelColumnInfo(
            label_column_name="cell_type", n_unique_values=output_size
        )
    ]
    fields = [
        config.FieldInfo("genes", vocab_size),
        config.FieldInfo(
            "expressions", vocab_size, is_masked=True, decode_modes={"token_scores": {}}
        ),
    ]

    model_config = config.SCBertConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        fields=fields,
        label_columns=label_columns,
    )
    model = scbert.SCBertForMultiTaskModeling(model_config)

    input_ids = torch.randint(0, 99, (batch_size, 2, sequence_len))
    attention_mask = torch.ones((batch_size, sequence_len))
    mlm_labels = torch.randint(0, 99, (batch_size, 1, sequence_len))
    cell_type_labels = torch.randint(0, output_size, (batch_size,))

    outputs = model(input_ids, attention_mask=attention_mask)

    # Test MLM focal loss
    mlm_focal_objective = FocalObjective()
    mlm_focal_objective.bind(vocab_size)
    mlm_focal_loss = mlm_focal_objective.compute(
        outputs.logits["expressions_token_scores"], mlm_labels[:, 0]
    )

    # Test label focal loss
    label_focal_objective = FocalObjective()
    label_focal_objective.bind(output_size)
    label_focal_loss = label_focal_objective.compute(
        outputs.logits["cell_type"], cell_type_labels
    )

    total_focal_loss = mlm_focal_loss + label_focal_loss
    assert not total_focal_loss.isinf()
    assert total_focal_loss > 0
    total_focal_loss.backward()


def test_scbert_forward_dummy_data():
    batch_size = 3
    sequence_len = 100
    dataset = helpers.generate_dataset(
        10 * batch_size, min_seq_len=sequence_len, max_seq_len=sequence_len, seed=42
    )
    tokenizer = helpers.load_test_tokenizer()
    fields = [
        config.FieldInfo("genes", is_masked=True, decode_modes={"token_scores": {}}),
        config.FieldInfo(
            "expressions", is_masked=True, decode_modes={"token_scores": {}}
        ),
    ]
    for f in fields:
        f.update_vocab_size(tokenizer)
    collator = MultiFieldCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=2,
        fields=fields,
        masker=Masker(
            change_ratio=0.2, mask_ratio=1.0, switch_ratio=0.0, tokenizer=tokenizer
        ),
    )
    model_config = config.SCBertConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        fields=fields,
    )
    model = scbert.SCBertForMaskedLM(model_config)

    dataloader = DataLoader(dataset=dataset, collate_fn=collator, batch_size=batch_size)

    for batch in dataloader:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        labels = batch["labels"]
        total_ce_loss = 0
        total_token_value_loss = 0
        for idx, field in enumerate(filter(lambda x: x.is_masked, fields)):
            total_ce_loss += ce_loss(
                logits=outputs.logits[field.field_name + "_token_scores"].reshape(
                    -1, field.vocab_size
                ),
                labels=labels[field.field_name].reshape(-1),
                label_smoothing=0.01,
            )
            token_values = tokenizer.get_token_values(field.field_name)
            if token_values is not None:
                total_token_value_loss += token_value_loss(
                    labels=labels[field.field_name],
                    logits=outputs.logits[field.field_name + "_token_scores"],
                    token_values=token_values,
                )
        assert not total_ce_loss.isinf()
        assert total_ce_loss.requires_grad
        assert total_ce_loss > 0

        assert total_token_value_loss.requires_grad
        assert not total_token_value_loss.isinf()
        assert total_token_value_loss > 0

        loss = total_ce_loss + total_token_value_loss
        assert loss is not None

        loss.backward()

    for batch in dataloader:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        labels = batch["labels"]
        total_focal_loss = 0
        total_token_value_loss = 0
        for idx, field in enumerate(filter(lambda x: x.is_masked, fields)):
            total_focal_loss += focal_loss(
                logits=outputs.logits[field.field_name + "_token_scores"].reshape(
                    -1, field.vocab_size
                ),
                labels=labels[field.field_name].reshape(-1),
            )
            token_values = tokenizer.get_token_values(field.field_name)
            if token_values is not None:
                total_token_value_loss += token_value_loss(
                    labels=labels[field.field_name],
                    logits=outputs.logits[field.field_name + "_token_scores"],
                    token_values=token_values,
                )
        assert not total_focal_loss.isinf()
        assert total_focal_loss.requires_grad
        assert total_focal_loss > 0

        loss = total_focal_loss + total_token_value_loss
        assert loss is not None

        loss.backward()


@pytest.mark.parametrize(
    ("inverse_method", "inverse_n_iter"),
    [("newton", 10), ("chebyshev", 10), ("original", 10), (None, None)],
)
def test_nystromformer_self_attention_pinv_attr(inverse_method, inverse_n_iter):
    cfg = {
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_landmarks": 32,
        "conv_kernel_size": None,
        "max_position_embeddings": 64,
        "attention_probs_dropout_prob": 0.2,
    }
    if inverse_method:
        cfg["inverse_method"] = inverse_method
        cfg["inverse_n_iter"] = inverse_n_iter
    else:
        inverse_method = "original"
        inverse_n_iter = 6
    cfg = config.SCNystromformerConfig(**cfg)
    model = SCNystromformerSelfAttention(cfg)
    assert model.inverse_method == inverse_method
    assert model.inverse_n_iter == inverse_n_iter


@pytest.mark.parametrize(
    ("n_iter", "inverse_method", "N"),
    [
        (10, "original", 4),
        (13, "newton", 4),
        (10, "chebyshev", 4),
        (16, "original", 16),
        (19, "newton", 16),
        (16, "chebyshev", 16),
    ],
)
def test_nystromformer_self_attention_pinv(n_iter, inverse_method, N):
    rng = torch.Generator()
    rng = rng.manual_seed(42)
    a = torch.rand(1, 1, N, N, generator=rng)
    a = torch.softmax(a, dim=-1)
    eps = torch.finfo(torch.float32).eps
    l = torch.max(torch.linalg.eigvals(a).abs())
    tol = eps * l * (N**2)
    p_inv = SCNystromformerSelfAttention.iterative_inv(
        mat=a, n_iter=n_iter, inverse_method=inverse_method
    )
    actual = torch.matmul(a.view(N, N), p_inv.view(N, N))
    expected = torch.eye(a.shape[-2])
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)


def test_scbert_with_pl_module_mask_both():
    """
    Test masking both genes and expressions using new API.

    NOTE: Metrics come from module's DEFAULT_METRICS, not from LossTask.metrics.
    """
    fields = [
        config.FieldInfo("genes", is_masked=True, decode_modes={"token_scores": {}}),
        config.FieldInfo(
            "expressions", is_masked=True, decode_modes={"token_scores": {}}
        ),
    ]
    losses = [
        LossTask(
            source=FieldSource("genes"),
            objective=CrossEntropyObjective(),
            weight=1,
        ),
        LossTask(
            source=FieldSource("expressions"),
            objective=CrossEntropyObjective(),
            weight=1,
        ),
        LossTask(
            source=FieldSource("expressions"),
            objective=TokenValueObjective(),
            weight=1,
        ),
    ]
    trainer_config = config.TrainerConfig(losses=losses)
    tokenizer = helpers.load_test_tokenizer()
    for field in fields:
        field.update_vocab_size(tokenizer)
    model_config = config.SCBertConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        hidden_size=32,
        fields=fields,
    )
    trainer = generate_and_train(fields, trainer_config, model_config, tokenizer)

    # Check essential loss metrics are present
    assert "train/genes_cross_entropy_loss_step" in trainer.logged_metrics
    assert "train/expressions_cross_entropy_loss_step" in trainer.logged_metrics
    assert "train/expressions_token_value_loss_step" in trainer.logged_metrics
    assert "train/loss_step" in trainer.logged_metrics


def test_scbert_with_pl_module_mask_both_regression():
    """Test masking genes (classification) and expressions (regression) with explicit metrics using new API."""
    fields = [
        config.FieldInfo("genes", is_masked=True, decode_modes={"token_scores": {}}),
        config.FieldInfo(
            "expressions", is_masked=True, decode_modes={"regression": {}}
        ),
    ]
    losses = [
        LossTask(
            source=FieldSource("genes"),
            objective=CrossEntropyObjective(),
            weight=1,
            metrics=[{"name": "accuracy"}],  # Explicit: only accuracy
        ),
        LossTask(
            source=FieldSource("expressions"),
            objective=MSEObjective(),
            weight=1,
            metrics=[],  # Explicit: no metrics for regression
        ),
    ]
    trainer_config = config.TrainerConfig(losses=losses)
    tokenizer = helpers.load_test_tokenizer()
    for field in fields:
        field.update_vocab_size(tokenizer)
    model_config = config.SCBertConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        hidden_size=32,
        fields=fields,
    )
    trainer = generate_and_train(fields, trainer_config, model_config, tokenizer)

    # Use helper to build expected metrics dynamically
    expected_metrics = helpers.build_expected_metric_keys(losses)
    assert set(trainer.logged_metrics.keys()) == expected_metrics


def test_scbert_with_pl_module_mask_expressions():
    """Test masking only expressions using default MLM losses (backward compat test)."""
    fields = [
        config.FieldInfo("genes", is_masked=False),
        config.FieldInfo(
            "expressions", is_masked=True, decode_modes={"token_scores": {}}
        ),
    ]
    # This test uses the old helper function to test backward compatibility
    trainer_config = config.TrainerConfig(
        losses=helpers.default_mlm_losses_from_fields(fields)
    )
    tokenizer = helpers.load_test_tokenizer()
    for field in fields:
        field.update_vocab_size(tokenizer)
    model_config = config.SCBertConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        hidden_size=32,
        fields=fields,
    )
    trainer = generate_and_train(fields, trainer_config, model_config, tokenizer)

    # Use helper to build expected metrics dynamically from dict format
    expected_metrics = helpers.build_expected_metric_keys(
        helpers.default_mlm_losses_from_fields(fields)
    )
    assert set(trainer.logged_metrics.keys()) == expected_metrics


def test_all_models_with_pl_module_position_embedding():
    fields = [
        config.FieldInfo("genes", is_masked=False),
        config.FieldInfo(
            "expressions", is_masked=True, decode_modes={"token_scores": {}}
        ),
    ]
    trainer_config = config.TrainerConfig(
        losses=helpers.default_mlm_losses_from_fields(fields)
    )
    tokenizer = helpers.load_test_tokenizer()
    for field in fields:
        field.update_vocab_size(tokenizer)
    for config_factory in [
        config.SCBertConfig,
        config.SCNystromformerConfig,
        config.SCPerformerConfig,
    ]:
        for position_embedding_type in ["absolute", "sinusoidal"]:
            model_config = config_factory(
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=64,
                hidden_size=32,
                fields=fields,
                position_embedding_type=position_embedding_type,
            )
            generate_and_train(fields, trainer_config, model_config, tokenizer)


def test_scbert_with_pl_module_weighted_losses():
    fields = [
        config.FieldInfo("genes", is_masked=False),
        config.FieldInfo(
            "expressions", is_masked=True, decode_modes={"token_scores": {}}
        ),
    ]
    losses = [
        LossTask(
            source=FieldSource("expressions"),
            objective=CrossEntropyObjective(),
            weight=1,
        ),
        LossTask(
            source=FieldSource("expressions"),
            objective=TokenValueObjective(),
            weight=5,
        ),
    ]

    trainer_config = config.TrainerConfig(losses=losses)
    tokenizer = helpers.load_test_tokenizer()
    for field in fields:
        field.update_vocab_size(tokenizer)
    model_config = config.SCBertConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        hidden_size=32,
        fields=fields,
    )
    trainer = generate_and_train(fields, trainer_config, model_config, tokenizer)

    # Check that loss metrics are present
    assert "train/expressions_cross_entropy_loss_step" in trainer.logged_metrics
    assert "train/expressions_token_value_loss_step" in trainer.logged_metrics
    assert "train/loss_step" in trainer.logged_metrics

    # Test weighted loss calculation
    ce_loss_value = trainer.logged_metrics["train/expressions_cross_entropy_loss_step"]
    tv_loss_value = trainer.logged_metrics["train/expressions_token_value_loss_step"]
    weighted_sum = (ce_loss_value * 1 + tv_loss_value * 5) / 6
    total_loss = trainer.logged_metrics["train/loss_step"]
    np.testing.assert_almost_equal(total_loss, weighted_sum)


def test_scbert_with_pl_module_weighted_losses_including_zero_weights():
    fields = [
        config.FieldInfo("genes", is_masked=False),
        config.FieldInfo(
            "expressions", is_masked=True, decode_modes={"token_scores": {}}
        ),
    ]
    losses = [
        LossTask(
            source=FieldSource("expressions"),
            objective=CrossEntropyObjective(),
            weight=0,  # Zero weight - loss computed but not used
        ),
        LossTask(
            source=FieldSource("expressions"),
            objective=TokenValueObjective(),
            weight=5,
        ),
    ]

    trainer_config = config.TrainerConfig(losses=losses)
    tokenizer = helpers.load_test_tokenizer()
    for field in fields:
        field.update_vocab_size(tokenizer)
    model_config = config.SCBertConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        hidden_size=32,
        fields=fields,
    )
    trainer = generate_and_train(fields, trainer_config, model_config, tokenizer)

    # Check essential loss metrics are present
    assert "train/expressions_cross_entropy_loss_step" in trainer.logged_metrics
    assert "train/expressions_token_value_loss_step" in trainer.logged_metrics
    assert "train/loss_step" in trainer.logged_metrics

    # Test weighted loss calculation with zero weight
    ce_loss_value = trainer.logged_metrics["train/expressions_cross_entropy_loss_step"]
    tv_loss_value = trainer.logged_metrics["train/expressions_token_value_loss_step"]
    weighted_sum = (ce_loss_value * 0 + tv_loss_value * 5) / 5
    total_loss = trainer.logged_metrics["train/loss_step"]
    np.testing.assert_almost_equal(total_loss, weighted_sum)


def test_scbert_with_pl_module_token_value_loss_only():
    """Test token value loss only with explicit accuracy metric using new API."""
    fields = [
        config.FieldInfo("genes", is_masked=False),
        config.FieldInfo(
            "expressions", is_masked=True, decode_modes={"token_scores": {}}
        ),
    ]
    losses = [
        LossTask(
            source=FieldSource("expressions"),
            objective=TokenValueObjective(),
            metrics=[{"name": "accuracy"}],  # Explicit: only accuracy
        )
    ]

    trainer_config = config.TrainerConfig(losses=losses)
    tokenizer = helpers.load_test_tokenizer()
    for field in fields:
        field.update_vocab_size(tokenizer)
    model_config = config.SCBertConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        hidden_size=32,
        fields=fields,
    )
    trainer = generate_and_train(fields, trainer_config, model_config, tokenizer)

    # Use helper to build expected metrics dynamically
    expected_metrics = helpers.build_expected_metric_keys(losses)
    assert set(trainer.logged_metrics.keys()) == expected_metrics

    # Test that total loss equals token_value loss (only loss)
    total_loss = trainer.logged_metrics["train/loss_step"]
    total_token_value_loss = trainer.logged_metrics[
        "train/expressions_token_value_loss_step"
    ]
    np.testing.assert_almost_equal(total_loss, total_token_value_loss)


def test_scbert_with_pl_module_mse_loss_only():
    """Test MSE loss only with no metrics using new API."""
    fields = [
        config.FieldInfo("genes", is_masked=False),
        config.FieldInfo(
            "expressions", is_masked=True, decode_modes={"regression": {}}
        ),
    ]
    losses = [
        LossTask(
            source=FieldSource("expressions"),
            objective=MSEObjective(),
            metrics=[],  # Explicit: no metrics
        )
    ]

    trainer_config = config.TrainerConfig(losses=losses)
    tokenizer = helpers.load_test_tokenizer()
    for field in fields:
        field.update_vocab_size(tokenizer)
    model_config = config.SCBertConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        hidden_size=32,
        fields=fields,
    )
    trainer = generate_and_train(fields, trainer_config, model_config, tokenizer)

    # Use helper to build expected metrics dynamically
    expected_metrics = helpers.build_expected_metric_keys(losses)
    assert set(trainer.logged_metrics.keys()) == expected_metrics

    # Test that total loss equals MSE loss (only loss)
    total_loss = trainer.logged_metrics["train/loss_step"]
    total_mse_loss = trainer.logged_metrics["train/expressions_mse_loss_step"]
    np.testing.assert_almost_equal(total_loss, total_mse_loss)


@pytest.mark.xfail()
def test_scbert_with_pl_module_token_value_loss_only_gene_masking_only():
    """Test token value loss on genes only (xfail - known issue)."""
    fields = [
        config.FieldInfo("genes", is_masked=True, decode_modes={"token_scores": {}}),
        config.FieldInfo("expressions", is_masked=False),
    ]
    losses = [
        LossTask(
            source=FieldSource("genes"),
            objective=TokenValueObjective(),
            metrics=[],  # Explicit: no metrics
        ),
    ]

    trainer_config = config.TrainerConfig(losses=losses)
    tokenizer = helpers.load_test_tokenizer()
    for field in fields:
        field.update_vocab_size(tokenizer)
    model_config = config.SCBertConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        hidden_size=32,
        fields=fields,
    )
    trainer = generate_and_train(fields, trainer_config, model_config, tokenizer)

    # Use helper to build expected metrics dynamically
    expected_metrics = helpers.build_expected_metric_keys(losses)
    assert set(trainer.logged_metrics.keys()) == expected_metrics

    total_loss = trainer.logged_metrics["train/loss_step"]
    total_token_value_loss = trainer.logged_metrics["train/genes_token_value_loss_step"]
    np.testing.assert_almost_equal(total_loss, total_token_value_loss)


def test_scbert_with_all_valid_losses():
    """Test all valid loss types with explicit metrics using new API."""
    fields = [
        config.FieldInfo("genes", is_masked=True, decode_modes={"token_scores": {}}),
        config.FieldInfo(
            "expressions",
            is_masked=True,
            decode_modes={"regression": {}, "token_scores": {}},
        ),
    ]
    losses = [
        LossTask(
            source=FieldSource("expressions"),
            objective=TokenValueObjective(),
            metrics=[],  # Explicit: no metrics for token_value
        ),
        LossTask(
            source=FieldSource("expressions"),
            objective=MSEObjective(),
            metrics=[],  # Explicit: no metrics for MSE
        ),
        LossTask(
            source=FieldSource("expressions"),
            objective=CrossEntropyObjective(),
            metrics=[
                {"name": "accuracy"}
            ],  # Explicit: only accuracy, not f1 or perplexity
        ),
        LossTask(
            source=FieldSource("genes"),
            objective=CrossEntropyObjective(),
            metrics=[
                {"name": "accuracy"}
            ],  # Explicit: only accuracy, not f1 or perplexity
        ),
    ]

    trainer_config = config.TrainerConfig(losses=losses)
    tokenizer = helpers.load_test_tokenizer()
    for field in fields:
        field.update_vocab_size(tokenizer)
    model_config = config.SCBertConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        hidden_size=32,
        fields=fields,
    )
    trainer = generate_and_train(fields, trainer_config, model_config, tokenizer)

    # Use helper to build expected metrics dynamically
    expected_metrics = helpers.build_expected_metric_keys(losses)
    assert set(trainer.logged_metrics.keys()) == expected_metrics


def test_scbert_all_frozen_pre_computed_gene_embeddings():
    fields = set_fields_pretrained_embedding(
        str(Path(__file__).parent / "test_vocab/pre_computed_gene_embeddings_full.txt")
    )
    losses = [
        LossTask(
            source=FieldSource("genes"),
            objective=CrossEntropyObjective(),
            weight=1,
        ),
        LossTask(
            source=FieldSource("expressions"),
            objective=CrossEntropyObjective(),
            weight=1,
        ),
    ]
    trainer_config = config.TrainerConfig(losses=losses)
    tokenizer = helpers.load_test_tokenizer()
    for field in fields:
        field.update_vocab_size(tokenizer)
        if field.pretrained_embedding:
            field.update_pretrained_embedding_indices(tokenizer)

    model_config = config.SCBertConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        hidden_size=32,
        fields=fields,
    )
    trainer = generate_and_train(fields, trainer_config, model_config, tokenizer)
    frozen_ind_embedding = fields[0].pretrained_embedding.embedding_indices_to_freeze
    pre_computed_ind = fields[0].pretrained_embedding.pre_trained_indices_to_use
    pre_computed_embedding = (
        fields[0].pretrained_embedding.load_pretrained_embeddings().values
    )
    gene_embedding_post_training = (
        trainer.model.model.scbert.embeddings.genes_embeddings.weight.data
    )
    np.testing.assert_allclose(
        gene_embedding_post_training[frozen_ind_embedding],
        pre_computed_embedding[pre_computed_ind],
    )


@pytest.mark.parametrize(
    "file_name",
    [
        "test_vocab/pre_computed_gene_embeddings_missing.txt",
        "test_vocab/pre_computed_gene_embeddings_missing_unk.txt",
    ],
)
def test_scbert_partly_frozen_pre_computed_gene_embeddings_with_missing(file_name):
    fields = set_fields_pretrained_embedding(str(Path(__file__).parent / file_name))
    losses = [
        LossTask(
            source=FieldSource("genes"),
            objective=CrossEntropyObjective(),
            weight=1,
        ),
        LossTask(
            source=FieldSource("expressions"),
            objective=CrossEntropyObjective(),
            weight=1,
        ),
    ]
    trainer_config = config.TrainerConfig(losses=losses)
    tokenizer = helpers.load_test_tokenizer()
    for field in fields:
        field.update_vocab_size(tokenizer)
        if field.pretrained_embedding:
            field.update_pretrained_embedding_indices(tokenizer)

    model_config = config.SCBertConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        hidden_size=32,
        fields=fields,
    )
    trainer, init_gene_weights = generate_and_train(
        fields,
        trainer_config,
        model_config,
        tokenizer,
        return_init_gene_weights=True,
    )

    frozen_ind_embedding = fields[0].pretrained_embedding.embedding_indices_to_freeze
    pre_computed_ind = fields[0].pretrained_embedding.pre_trained_indices_to_use
    pre_computed_embeddings = (
        fields[0]
        .pretrained_embedding.load_pretrained_embeddings()
        .values[pre_computed_ind]
    )
    post_training_embedding = (
        trainer.model.model.scbert.embeddings.genes_embeddings.weight.data
    )

    post_training_embeddings_frozen = post_training_embedding[frozen_ind_embedding]
    exclude_indices_tensor = torch.tensor(frozen_ind_embedding)
    mask = torch.ones(post_training_embedding.size(0), dtype=torch.bool)
    mask[exclude_indices_tensor] = False
    post_training_embeddings_unfrozen = post_training_embedding[mask]
    pre_training_embeddings_unfrozen = init_gene_weights[mask]

    np.testing.assert_allclose(post_training_embeddings_frozen, pre_computed_embeddings)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            post_training_embeddings_unfrozen, pre_training_embeddings_unfrozen
        )


def test_get_indices_of_pretrained_token_embeddings():
    test_dict = {
        Path(__file__).parent / "test_vocab/pre_computed_gene_embeddings_full.txt": 12,
        Path(__file__).parent
        / "test_vocab/pre_computed_gene_embeddings_missing.txt": 6,
    }
    for filename, expected_len in test_dict.items():
        fields = set_fields_pretrained_embedding(filename)
        tokenizer = helpers.load_test_tokenizer()
        for field in fields:
            field.update_vocab_size(tokenizer)
            if field.pretrained_embedding:
                field.update_pretrained_embedding_indices(tokenizer)
                assert (
                    len(field.pretrained_embedding.embedding_indices_to_freeze)
                    == expected_len
                )


def test_check_unk():
    fields = set_fields_pretrained_embedding(
        Path(__file__).parent
        / "test_vocab/pre_computed_gene_embeddings_missing_unk.txt"
    )
    tokenizer = helpers.load_test_tokenizer()
    for field in fields:
        field.update_vocab_size(tokenizer)
        if field.pretrained_embedding:
            field.update_pretrained_embedding_indices(tokenizer)
            assert len(field.pretrained_embedding.pre_trained_indices_to_use) == 9


def set_fields_pretrained_embedding(filename):
    fields = [
        config.FieldInfo(
            "genes",
            is_masked=True,
            decode_modes={"token_scores": {}},
            pretrained_embedding=config.PreTrainedEmbeddingConfig(filename=filename),
        ),
        config.FieldInfo(
            "expressions", is_masked=True, decode_modes={"token_scores": {}}
        ),
    ]
    return fields


def generate_and_train(
    fields,
    trainer_config,
    model_config,
    tokenizer,
    batch_size=3,
    generated_data_sequence_len=512,
    return_init_gene_weights=False,
    token_fields=["genes", "expressions"],
    scalar_valued_fields=None,
    masker=None,
    collator_sequence_len=None,
):
    if masker is None:
        masker = Masker(
            change_ratio=0.2, mask_ratio=1.0, switch_ratio=0.0, tokenizer=tokenizer
        )
    if collator_sequence_len is None:
        collator_sequence_len = generated_data_sequence_len
    collator = MultiFieldCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=2,
        fields=fields,
        masker=masker,
        max_length=collator_sequence_len,
    )
    if isinstance(model_config, config.SCNystromformerConfig):
        model_config.max_position_embeddings = collator_sequence_len
    pl_module = MLMTrainingModule(model_config, trainer_config, tokenizer)
    if return_init_gene_weights:
        base_model = getattr(pl_module.model, pl_module.model.base_model_prefix)
        gene_embeddings = base_model.embeddings.genes_embeddings
        gene_embedding_init = gene_embeddings.weight.data.clone()

    dataset = helpers.generate_dataset(
        10 * batch_size,
        min_seq_len=generated_data_sequence_len,
        max_seq_len=generated_data_sequence_len,
        seed=42,
        token_fields=token_fields,
        scalar_valued_fields=scalar_valued_fields,
        tokenizer=tokenizer,
    )
    val_dataset = helpers.generate_dataset(
        2 * batch_size,
        min_seq_len=generated_data_sequence_len,
        max_seq_len=generated_data_sequence_len,
        seed=64,
        token_fields=token_fields,
        scalar_valued_fields=scalar_valued_fields,
        tokenizer=tokenizer,
    )
    dataloader = DataLoader(dataset=dataset, collate_fn=collator, batch_size=batch_size)

    val_dataloader = DataLoader(
        dataset=val_dataset, collate_fn=collator, batch_size=batch_size
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = pl.Trainer(
            max_epochs=2,
            log_every_n_steps=5,
            check_val_every_n_epoch=1,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            default_root_dir=tmpdir,
        )

        trainer.fit(
            model=pl_module,
            train_dataloaders=dataloader,
            val_dataloaders=val_dataloader,
        )
        for metric_name, metric_value in trainer.logged_metrics.items():
            assert not metric_value.isinf(), metric_name + " is inf"
            if "accuracy" not in metric_name:
                assert metric_value > 0, metric_name + " is zero"

        if return_init_gene_weights:
            return trainer, gene_embedding_init
        return trainer


def test_scbert_forward_with_continuous_value_encoder():
    vocab_size = 100
    num_special_tokens = 5
    batch_size = 3
    sequence_len = 10
    fields = [
        config.FieldInfo("genes", vocab_size),
        config.FieldInfo(
            "expressions",
            vocab_size=None,
            num_special_tokens=num_special_tokens,
            is_masked=True,
            tokenization_strategy="continuous_value_encoder",
            decode_modes=["regression"],
        ),
    ]

    model_config = config.SCBertConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        fields=fields,
    )
    model = scbert.SCBertForMaskedLM(model_config)

    input_ids = torch.randint(0, 99, (batch_size, 2, sequence_len))
    attention_mask = torch.ones((batch_size, sequence_len))
    labels = torch.randint(0, 99, (batch_size, 1, sequence_len))

    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

    total_mse_loss = mse_loss(
        logits=outputs.logits["expressions_regression"].reshape(-1),
        labels=labels[:, 0].reshape(-1),
    )
    assert not total_mse_loss.isinf()
    assert total_mse_loss > 0

    total_mse_loss.backward()
    assert tuple(outputs.logits["expressions_regression"].shape) == (
        batch_size,
        sequence_len,
        1,
    )


def test_scbert_forward_with_scale_adapt():
    vocab_size = 100
    num_special_tokens = 5
    batch_size = 3
    sequence_len = 10

    encoder_kwargs = {
        "kind": "scale_adapt",
        "n_sin_basis": 11,
        "shift": 0.0,
        "basis_scale": 0.1,
        "sigmoid_centers": [0.0],
        "sigmoid_orientations": [1.0],
    }

    fields = [
        config.FieldInfo("genes", vocab_size),
        config.FieldInfo(
            "expressions",
            vocab_size=None,
            num_special_tokens=num_special_tokens,
            is_masked=True,
            tokenization_strategy="continuous_value_encoder",
            encoder_kwargs=encoder_kwargs,
            decode_modes=["regression"],
        ),
    ]

    model_config = config.SCBertConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        fields=fields,
    )
    model = scbert.SCBertForMaskedLM(model_config)

    input_ids = torch.randint(0, 99, (batch_size, 2, sequence_len))
    attention_mask = torch.ones((batch_size, sequence_len))
    labels = torch.randint(0, 99, (batch_size, 1, sequence_len))

    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

    total_mse_loss = mse_loss(
        logits=outputs.logits["expressions_regression"].reshape(-1),
        labels=labels[:, 0].reshape(-1),
    )
    assert not total_mse_loss.isinf()
    assert total_mse_loss > 0

    total_mse_loss.backward()
    assert tuple(outputs.logits["expressions_regression"].shape) == (
        batch_size,
        sequence_len,
        1,
    )


def test_scbert_train_with_no_tokenization(gene2vec_fields_regression_no_tokenization):
    fields = gene2vec_fields_regression_no_tokenization
    losses = [
        LossTask(
            source=FieldSource("genes"),
            objective=CrossEntropyObjective(),
            metrics=[{"name": "accuracy"}],
            weight=1,
        ),
        LossTask(
            source=FieldSource("expressions"),
            objective=MSEObjective(),
            weight=1,
        ),
    ]
    from bmfm_targets.tokenization import load_tokenizer

    trainer_config = config.TrainerConfig(losses=losses)
    tokenizer = load_tokenizer("gene2vec")

    model_config = config.SCBertConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        hidden_size=32,
        fields=fields,
    )
    trainer = generate_and_train(
        fields,
        trainer_config,
        model_config,
        tokenizer,
        token_fields=["genes"],
        scalar_valued_fields=["expressions"],
    )


def test_focal_w_default_parameters_equals_ce_loss():
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    labels = torch.tensor([0, 1, 0])
    ce_loss_val = ce_loss(logits, labels, label_smoothing=0.0)
    focal_loss_val = focal_loss(logits, labels, focal_gamma=0.0)
    assert ce_loss_val.item() == pytest.approx(focal_loss_val.item(), abs=1e-6)


def test_scmodernbert_forward():
    vocab_size = 100
    batch_size = 2
    sequence_len = 10
    fields = [
        config.FieldInfo("genes", vocab_size),
        config.FieldInfo(
            "expressions", vocab_size, is_masked=True, decode_modes={"token_scores": {}}
        ),
    ]

    model_config = config.SCModernBertConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        fields=fields,
    )
    model = scmodernbert.SCModernBertForMaskedLM(model_config)

    input_ids = torch.randint(0, 99, (batch_size, 2, sequence_len))
    attention_mask = torch.ones((batch_size, sequence_len))
    labels = torch.randint(0, 99, (batch_size, 1, sequence_len))

    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

    total_ce_loss = ce_loss(
        logits=outputs.logits["expressions_token_scores"].reshape(-1, vocab_size),
        labels=labels[:, 0].reshape(-1),
        label_smoothing=0.01,
    )
    assert not total_ce_loss.isinf()
    assert total_ce_loss > 0

    total_focal_loss = focal_loss(
        logits=outputs.logits["expressions_token_scores"].reshape(-1, vocab_size),
        labels=labels[:, 0].reshape(-1),
    )
    assert not total_focal_loss.isinf()
    assert total_focal_loss > 0

    total_token_value_loss = token_value_loss(
        logits=outputs.logits["expressions_token_scores"],
        labels=labels[:, 0],
        token_values=np.arange(vocab_size),
    )

    assert not total_token_value_loss.isinf()
    assert total_token_value_loss > 0

    loss = total_ce_loss + total_token_value_loss
    assert loss is not None

    loss.backward()
    assert tuple(outputs.logits["expressions_token_scores"].shape) == (
        batch_size,
        sequence_len,
        vocab_size,
    )


def test_scmodernbert_forward_with_continuous_value_encoder():
    vocab_size = 100
    num_special_tokens = 5
    batch_size = 3
    sequence_len = 10
    fields = [
        config.FieldInfo("genes", vocab_size),
        config.FieldInfo(
            "expressions",
            vocab_size=None,
            num_special_tokens=num_special_tokens,
            is_masked=True,
            tokenization_strategy="continuous_value_encoder",
            decode_modes=["regression"],
        ),
    ]

    model_config = config.SCModernBertConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        fields=fields,
    )
    model = scmodernbert.SCModernBertForMaskedLM(model_config)

    input_ids = torch.randint(0, 99, (batch_size, 2, sequence_len))
    attention_mask = torch.ones((batch_size, sequence_len))
    labels = torch.randint(0, 99, (batch_size, 1, sequence_len))

    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

    total_mse_loss = mse_loss(
        logits=outputs.logits["expressions_regression"].reshape(-1),
        labels=labels[:, 0].reshape(-1),
    )
    assert not total_mse_loss.isinf()
    assert total_mse_loss > 0

    total_mse_loss.backward()
    assert tuple(outputs.logits["expressions_regression"].shape) == (
        batch_size,
        sequence_len,
        1,
    )


def test_focal_loss_scmodernbert_multitask_forward():
    vocab_size = 100
    batch_size = 3
    sequence_len = 10
    output_size = 10
    label_columns = [
        config.LabelColumnInfo(
            label_column_name="cell_type", n_unique_values=output_size
        )
    ]
    fields = [
        config.FieldInfo("genes", vocab_size),
        config.FieldInfo(
            "expressions", vocab_size, is_masked=True, decode_modes={"token_scores": {}}
        ),
    ]

    model_config = config.SCModernBertConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        fields=fields,
        label_columns=label_columns,
    )
    model = scmodernbert.SCModernBertForMultiTaskModeling(model_config)

    input_ids = torch.randint(0, 99, (batch_size, 2, sequence_len))
    attention_mask = torch.ones((batch_size, sequence_len))
    mlm_labels = torch.randint(0, 99, (batch_size, 1, sequence_len))

    cell_type_labels = torch.randint(0, output_size, (batch_size,))

    outputs = model(input_ids, attention_mask=attention_mask)
    mlm_focal_loss = focal_loss(
        logits=outputs.logits["expressions_token_scores"].reshape(-1, vocab_size),
        labels=mlm_labels[:, 0].reshape(-1),
    )
    label_focal_loss = focal_loss(
        logits=outputs.logits["cell_type"],
        labels=cell_type_labels,
    )
    total_focal_loss = mlm_focal_loss + label_focal_loss
    assert not total_focal_loss.isinf()
    assert total_focal_loss > 0

    total_focal_loss.backward()


def test_scbert_nomask(mock_clearml_logger):
    fields = [
        config.FieldInfo("genes"),
        config.FieldInfo(
            "expressions",
            decode_modes={
                "wced": {
                    "vocab_field": "genes",
                    "logit_outputs": ["mse", "is_zero_bce"],
                }
            },
        ),
    ]
    losses = []
    for target in "all_genes", "non_input_genes", "input_genes":
        losses.extend(
            [
                LossTask(WCEDFieldSource("expressions", target), MSEObjective()),
                LossTask(WCEDFieldSource("expressions", target), IsZeroBCEObjective()),
            ]
        )

    trainer_config = config.TrainerConfig(
        losses=losses, batch_prediction_behavior="track"
    )
    from bmfm_targets.tokenization import load_tokenizer

    tokenizer = load_tokenizer("protein_coding")
    for field in fields:
        field.update_vocab_size(tokenizer)
    model_config = config.SCBertConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        hidden_size=32,
        fields=fields,
    )
    from bmfm_targets.training.masking.strategy import WCEDMasker

    trainer = generate_and_train(
        fields,
        trainer_config,
        model_config,
        tokenizer,
        masker=WCEDMasker(tokenizer=tokenizer),
        generated_data_sequence_len=512,
        collator_sequence_len=256,
    )
    assert (torch.tensor([*trainer.logged_metrics.values()]) > 0).all()
    assert {*trainer.logged_metrics.keys()} == {
        "train/expressions_input_genes_is_zero_bce_loss_epoch",
        "train/expressions_input_genes_is_zero_bce_loss_step",
        "train/expressions_input_genes_mse_loss_epoch",
        "train/expressions_input_genes_mse_loss_step",
        "train/expressions_non_input_genes_is_zero_bce_loss_epoch",
        "train/expressions_non_input_genes_is_zero_bce_loss_step",
        "train/expressions_non_input_genes_mse_loss_epoch",
        "train/expressions_non_input_genes_mse_loss_step",
        "train/expressions_all_genes_is_zero_bce_loss_epoch",
        "train/expressions_all_genes_is_zero_bce_loss_step",
        "train/expressions_all_genes_mse_loss_epoch",
        "train/expressions_all_genes_mse_loss_step",
        "train/loss_epoch",
        "train/loss_step",
        "validation/expressions_input_genes_is_zero_bce_loss",
        "validation/expressions_input_genes_mse_loss",
        "validation/expressions_non_input_genes_is_zero_bce_loss",
        "validation/expressions_non_input_genes_mse_loss",
        "validation/expressions_all_genes_is_zero_bce_loss",
        "validation/expressions_all_genes_mse_loss",
        "validation/loss",
    }


def test_llama_multitask_forward():
    from bmfm_targets.models.predictive.llama import LlamaForMultiTaskConfig

    vocab_size = 100
    batch_size = 3
    sequence_len = 10
    output_size = 10
    label_columns = [
        config.LabelColumnInfo(
            label_column_name="cell_type", n_unique_values=output_size
        )
    ]
    fields = [
        config.FieldInfo("genes", vocab_size),
        config.FieldInfo(
            "expressions", vocab_size, is_masked=True, decode_modes={"token_scores": {}}
        ),
    ]

    model_config = LlamaForMultiTaskConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        fields=fields,
        label_columns=label_columns,
    )
    model = model_config.build_model(config.ModelingStrategy.MULTITASK)

    input_ids = torch.randint(0, 99, (batch_size, 2, sequence_len))
    attention_mask = torch.ones((batch_size, sequence_len))
    mlm_labels = torch.randint(0, 99, (batch_size, 1, sequence_len))

    cell_type_labels = torch.randint(0, output_size, (batch_size,))

    outputs = model(input_ids, attention_mask=attention_mask)
    mlm_loss = ce_loss(
        logits=outputs.logits["expressions_token_scores"].reshape(-1, vocab_size),
        labels=mlm_labels[:, 0].reshape(-1),
        label_smoothing=0.01,
    )
    label_loss = ce_loss(
        logits=outputs.logits["cell_type"],
        labels=cell_type_labels,
        label_smoothing=0.01,
    )
    total_ce_loss = mlm_loss + label_loss
    assert not total_ce_loss.isinf()
    assert total_ce_loss > 0

    total_ce_loss.backward()


def test_llama_forward_with_continuous_value_encoder():
    vocab_size = 100
    num_special_tokens = 5
    batch_size = 3
    sequence_len = 10
    output_size = 10
    fields = [
        config.FieldInfo("genes", vocab_size),
        config.FieldInfo(
            "expressions",
            vocab_size=None,
            num_special_tokens=num_special_tokens,
            is_masked=True,
            tokenization_strategy="continuous_value_encoder",
            decode_modes=["regression"],
        ),
    ]
    label_columns = [
        config.LabelColumnInfo(
            label_column_name="cell_type", n_unique_values=output_size
        )
    ]

    model_config = LlamaForMultiTaskConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        fields=fields,
        label_columns=label_columns,
    )
    model = model_config.build_model(config.ModelingStrategy.MULTITASK)

    input_ids = torch.randint(0, 99, (batch_size, 2, sequence_len))
    attention_mask = torch.ones((batch_size, sequence_len))
    labels = torch.randint(0, 99, (batch_size, 1, sequence_len))

    outputs = model(input_ids, attention_mask=attention_mask)

    total_mse_loss = mse_loss(
        logits=outputs.logits["expressions_regression"].reshape(-1),
        labels=labels[:, 0].reshape(-1),
    )
    assert not total_mse_loss.isinf()
    assert total_mse_loss > 0

    total_mse_loss.backward()
    assert tuple(outputs.logits["expressions_regression"].shape) == (
        batch_size,
        sequence_len,
        1,
    )


def test_llama_mlm_forward():
    vocab_size = 100
    batch_size = 3
    sequence_len = 10
    fields = [
        config.FieldInfo("genes", vocab_size),
        config.FieldInfo(
            "expressions", vocab_size, is_masked=True, decode_modes={"token_scores": {}}
        ),
    ]

    model_config = LlamaForMaskedLMConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        fields=fields,
    )
    model = model_config.build_model(config.ModelingStrategy.MLM)

    input_ids = torch.randint(0, 99, (batch_size, 2, sequence_len))
    attention_mask = torch.ones((batch_size, sequence_len))
    labels = torch.randint(0, 99, (batch_size, 1, sequence_len))

    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    total_ce_loss = ce_loss(
        logits=outputs.logits["expressions_token_scores"].reshape(-1, vocab_size),
        labels=labels[:, 0].reshape(-1),
        label_smoothing=0.01,
    )
    assert not total_ce_loss.isinf()
    assert total_ce_loss > 0

    total_focal_loss = focal_loss(
        logits=outputs.logits["expressions_token_scores"].reshape(-1, vocab_size),
        labels=labels[:, 0].reshape(-1),
    )
    assert not total_focal_loss.isinf()
    assert total_focal_loss > 0

    total_token_value_loss = token_value_loss(
        logits=outputs.logits["expressions_token_scores"],
        labels=labels[:, 0],
        token_values=np.arange(vocab_size),
    )

    assert not total_token_value_loss.isinf()
    assert total_token_value_loss > 0

    loss = total_ce_loss + total_token_value_loss
    assert loss is not None

    loss.backward()
    assert tuple(outputs.logits["expressions_token_scores"].shape) == (
        batch_size,
        sequence_len,
        vocab_size,
    )


def test_llama_nomask(mock_clearml_logger):
    fields = [
        config.FieldInfo("genes"),
        config.FieldInfo(
            "expressions",
            decode_modes={
                "wced": {
                    "vocab_field": "genes",
                    "logit_outputs": ["mse", "is_zero_bce"],
                }
            },
        ),
    ]
    losses = []
    for target in "all_genes", "non_input_genes", "input_genes":
        losses.extend(
            [
                LossTask(WCEDFieldSource("expressions", target), MSEObjective()),
                LossTask(WCEDFieldSource("expressions", target), IsZeroBCEObjective()),
            ]
        )
    trainer_config = config.TrainerConfig(
        losses=losses, batch_prediction_behavior="track"
    )
    from bmfm_targets.tokenization import load_tokenizer

    tokenizer = load_tokenizer("protein_coding")
    for field in fields:
        field.update_vocab_size(tokenizer)
    model_config = LlamaForMaskedLMConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_size=32,
        fields=fields,
    )
    from bmfm_targets.training.masking.strategy import WCEDMasker

    trainer = generate_and_train(
        fields,
        trainer_config,
        model_config,
        tokenizer,
        masker=WCEDMasker(tokenizer=tokenizer),
        generated_data_sequence_len=512,
        collator_sequence_len=256,
    )
    assert (torch.tensor([*trainer.logged_metrics.values()]) > 0).all()
    assert {*trainer.logged_metrics.keys()} == {
        "train/expressions_input_genes_is_zero_bce_loss_epoch",
        "train/expressions_input_genes_is_zero_bce_loss_step",
        "train/expressions_input_genes_mse_loss_epoch",
        "train/expressions_input_genes_mse_loss_step",
        "train/expressions_non_input_genes_is_zero_bce_loss_epoch",
        "train/expressions_non_input_genes_is_zero_bce_loss_step",
        "train/expressions_non_input_genes_mse_loss_epoch",
        "train/expressions_non_input_genes_mse_loss_step",
        "train/expressions_all_genes_is_zero_bce_loss_epoch",
        "train/expressions_all_genes_is_zero_bce_loss_step",
        "train/expressions_all_genes_mse_loss_epoch",
        "train/expressions_all_genes_mse_loss_step",
        "train/loss_epoch",
        "train/loss_step",
        "validation/expressions_input_genes_is_zero_bce_loss",
        "validation/expressions_input_genes_mse_loss",
        "validation/expressions_non_input_genes_is_zero_bce_loss",
        "validation/expressions_non_input_genes_mse_loss",
        "validation/expressions_all_genes_is_zero_bce_loss",
        "validation/expressions_all_genes_mse_loss",
        "validation/loss",
    }


def test_llama_train_with_no_tokenization(gene2vec_fields_regression_no_tokenization):
    fields = gene2vec_fields_regression_no_tokenization
    losses = [
        LossTask(
            source=FieldSource("genes"),
            objective=CrossEntropyObjective(),
            metrics=[{"name": "accuracy"}],
            weight=1,
        ),
        LossTask(
            source=FieldSource("expressions"),
            objective=MSEObjective(),
            weight=1,
        ),
    ]
    from bmfm_targets.tokenization import load_tokenizer

    trainer_config = config.TrainerConfig(losses=losses)
    tokenizer = load_tokenizer("gene2vec")

    model_config = LlamaForMaskedLMConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        hidden_size=32,
        fields=fields,
    )
    trainer = generate_and_train(
        fields,
        trainer_config,
        model_config,
        tokenizer,
        token_fields=["genes"],
        scalar_valued_fields=["expressions"],
    )


def test_llama_with_all_valid_losses():
    """Test Llama with all valid loss types using explicit metrics and new API."""
    fields = [
        config.FieldInfo("genes", is_masked=True, decode_modes={"token_scores": {}}),
        config.FieldInfo(
            "expressions",
            is_masked=True,
            decode_modes={"regression": {}, "token_scores": {}},
        ),
    ]
    losses = [
        LossTask(
            source=FieldSource("expressions"),
            objective=TokenValueObjective(),
            metrics=[],  # Explicit: no metrics
        ),
        LossTask(
            source=FieldSource("expressions"),
            objective=MSEObjective(),
            metrics=[],  # Explicit: no metrics
        ),
        LossTask(
            source=FieldSource("expressions"),
            objective=CrossEntropyObjective(),
            metrics=[{"name": "accuracy"}],  # Explicit: only accuracy
        ),
        LossTask(
            source=FieldSource("genes"),
            objective=CrossEntropyObjective(),
            metrics=[{"name": "accuracy"}],  # Explicit: only accuracy
        ),
    ]

    trainer_config = config.TrainerConfig(losses=losses)
    tokenizer = helpers.load_test_tokenizer()
    for field in fields:
        field.update_vocab_size(tokenizer)
    model_config = LlamaForMaskedLMConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_size=32,
        fields=fields,
    )
    trainer = generate_and_train(fields, trainer_config, model_config, tokenizer)

    # Use helper to build expected metrics dynamically
    expected_metrics = helpers.build_expected_metric_keys(losses)
    assert set(trainer.logged_metrics.keys()) == expected_metrics


def test_lllama_forward_with_scale_adapt():
    vocab_size = 100
    num_special_tokens = 5
    batch_size = 3
    sequence_len = 10

    encoder_kwargs = {
        "kind": "scale_adapt",
        "n_sin_basis": 11,
        "shift": 0.0,
        "basis_scale": 0.1,
        "sigmoid_centers": [0.0],
        "sigmoid_orientations": [1.0],
    }

    fields = [
        config.FieldInfo("genes", vocab_size),
        config.FieldInfo(
            "expressions",
            vocab_size=None,
            num_special_tokens=num_special_tokens,
            is_masked=True,
            tokenization_strategy="continuous_value_encoder",
            encoder_kwargs=encoder_kwargs,
            decode_modes=["regression"],
        ),
    ]

    model_config = LlamaForMaskedLMConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        fields=fields,
    )
    model = model_config.build_model(strategy="mlm")

    input_ids = torch.randint(0, 99, (batch_size, 2, sequence_len))
    attention_mask = torch.ones((batch_size, sequence_len))
    labels = torch.randint(0, 99, (batch_size, 1, sequence_len))

    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

    total_mse_loss = mse_loss(
        logits=outputs.logits["expressions_regression"].reshape(-1),
        labels=labels[:, 0].reshape(-1),
    )
    assert not total_mse_loss.isinf()
    assert total_mse_loss > 0

    total_mse_loss.backward()
    assert tuple(outputs.logits["expressions_regression"].shape) == (
        batch_size,
        sequence_len,
        1,
    )
