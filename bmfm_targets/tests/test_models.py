import tempfile
from pathlib import Path

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from bmfm_targets import config
from bmfm_targets.models.predictive import scbert, scnystromformer
from bmfm_targets.models.predictive.scnystromformer.modeling_scnystromformer import (
    SCNystromformerSelfAttention,
)
from bmfm_targets.tests import helpers
from bmfm_targets.tokenization import MultiFieldCollator
from bmfm_targets.training.masking import Masker
from bmfm_targets.training.metrics import (
    ce_loss,
    focal_loss,
    mse_loss,
    token_value_loss,
)
from bmfm_targets.training.modules import MLMTrainingModule


def test_nystromformer_forward():
    vocab_size = 100
    batch_size = 3
    sequence_len = 8
    fields = [
        config.FieldInfo("genes", vocab_size),
        config.FieldInfo("expressions", vocab_size, is_masked=True),
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
    total_ce_loss = ce_loss(
        logits=outputs.logits["expressions_token_scores"].reshape(-1, vocab_size),
        labels=labels[:, 0].reshape(-1),
        label_smoothing=0.01,
    )
    assert not total_ce_loss.isinf()
    assert total_ce_loss > 0
    assert tuple(outputs.logits["expressions_token_scores"].shape) == (
        batch_size,
        sequence_len,
        vocab_size,
    )

    total_focal_loss = focal_loss(
        logits=outputs.logits["expressions_token_scores"].reshape(-1, vocab_size),
        labels=labels[:, 0].reshape(-1),
    )
    assert not total_focal_loss.isinf()
    assert total_focal_loss > 0


def test_scbert_forward():
    vocab_size = 100
    batch_size = 3
    sequence_len = 10
    fields = [
        config.FieldInfo("genes", vocab_size),
        config.FieldInfo("expressions", vocab_size, is_masked=True),
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


def test_scbert_multitask_forward():
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
        config.FieldInfo("expressions", vocab_size, is_masked=True),
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


def test_focal_loss_scbert_multitask_forward():
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
        config.FieldInfo("expressions", vocab_size, is_masked=True),
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


def test_scbert_forward_dummy_data():
    batch_size = 3
    sequence_len = 100
    dataset = helpers.generate_dataset(
        10 * batch_size, min_seq_len=sequence_len, max_seq_len=sequence_len, seed=42
    )
    tokenizer = helpers.load_test_tokenizer()
    fields = [
        config.FieldInfo("genes", is_masked=True),
        config.FieldInfo("expressions", is_masked=True),
    ]
    for f in fields:
        f.update_vocab_size(tokenizer)
    collator = MultiFieldCollator(
        tokenizer=tokenizer,
        mlm=True,
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
    fields = [
        config.FieldInfo("genes", is_masked=True),
        config.FieldInfo("expressions", is_masked=True),
    ]
    losses = [
        {"field_name": "genes", "name": "cross_entropy", "weight": 1},
        {"field_name": "expressions", "name": "cross_entropy", "weight": 1},
        {"field_name": "expressions", "name": "token_value", "weight": 1},
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

    assert {*trainer.logged_metrics.keys()} == {
        "train/expressions_accuracy_epoch",
        "train/expressions_accuracy_step",
        "train/expressions_cross_entropy_loss_epoch",
        "train/expressions_cross_entropy_loss_step",
        "train/expressions_perplexity_epoch",
        "train/expressions_perplexity_step",
        "train/expressions_token_value_loss_epoch",
        "train/expressions_token_value_loss_step",
        "train/genes_accuracy_epoch",
        "train/genes_accuracy_step",
        "train/genes_cross_entropy_loss_epoch",
        "train/genes_cross_entropy_loss_step",
        "train/genes_perplexity_epoch",
        "train/genes_perplexity_step",
        "train/loss_epoch",
        "train/loss_step",
        "validation/expressions_accuracy",
        "validation/expressions_cross_entropy_loss",
        "validation/expressions_perplexity",
        "validation/expressions_token_value_loss",
        "validation/genes_accuracy",
        "validation/genes_cross_entropy_loss",
        "validation/genes_perplexity",
        "validation/loss",
    }


def test_scbert_with_pl_module_mask_both_regression():
    fields = [
        config.FieldInfo("genes", is_masked=True),
        config.FieldInfo("expressions", is_masked=True, decode_modes=["regression"]),
    ]
    losses = [
        {"field_name": "genes", "name": "cross_entropy", "weight": 1},
        {"field_name": "expressions", "name": "token_mse", "weight": 1},
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

    assert {*trainer.logged_metrics.keys()} == {
        "train/expressions_token_mse_loss_epoch",
        "train/expressions_token_mse_loss_step",
        "train/genes_accuracy_epoch",
        "train/genes_accuracy_step",
        "train/genes_cross_entropy_loss_epoch",
        "train/genes_cross_entropy_loss_step",
        "train/genes_perplexity_epoch",
        "train/genes_perplexity_step",
        "train/loss_epoch",
        "train/loss_step",
        "validation/expressions_token_mse_loss",
        "validation/genes_accuracy",
        "validation/genes_cross_entropy_loss",
        "validation/genes_perplexity",
        "validation/loss",
    }


def test_scbert_with_pl_module_mask_expressions():
    fields = [
        config.FieldInfo("genes", is_masked=False),
        config.FieldInfo("expressions", is_masked=True),
    ]
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

    assert {*trainer.logged_metrics.keys()} == {
        "train/expressions_accuracy_epoch",
        "train/expressions_accuracy_step",
        "train/expressions_cross_entropy_loss_epoch",
        "train/expressions_cross_entropy_loss_step",
        "train/expressions_perplexity_epoch",
        "train/expressions_perplexity_step",
        "train/loss_epoch",
        "train/loss_step",
        "validation/expressions_accuracy",
        "validation/expressions_cross_entropy_loss",
        "validation/expressions_perplexity",
        "validation/loss",
    }


def test_all_models_with_pl_module_position_embedding():
    fields = [
        config.FieldInfo("genes", is_masked=False),
        config.FieldInfo("expressions", is_masked=True),
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
        config.FieldInfo("expressions", is_masked=True),
    ]
    losses = [
        {"field_name": "expressions", "name": "cross_entropy", "weight": 1},
        {"field_name": "expressions", "name": "token_value", "weight": 5},
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

    assert set(trainer.logged_metrics.keys()) == {
        "train/expressions_accuracy_epoch",
        "train/expressions_accuracy_step",
        "train/expressions_cross_entropy_loss_epoch",
        "train/expressions_cross_entropy_loss_step",
        "train/expressions_perplexity_epoch",
        "train/expressions_perplexity_step",
        "train/expressions_token_value_loss_epoch",
        "train/expressions_token_value_loss_step",
        "train/loss_epoch",
        "train/loss_step",
        "validation/expressions_accuracy",
        "validation/expressions_cross_entropy_loss",
        "validation/expressions_perplexity",
        "validation/expressions_token_value_loss",
        "validation/loss",
    }

    weighted_sum = 0
    for w in losses:
        weighted_sum += trainer.logged_metrics[
            f"train/{w['field_name']}_{w['name']}_loss_step"
        ]
    weighted_sum /= sum(w["weight"] for w in losses)

    total_loss = trainer.logged_metrics["train/loss_step"]
    np.testing.assert_almost_equal(total_loss, weighted_sum)


def test_scbert_with_pl_module_token_value_loss_only():
    fields = [
        config.FieldInfo("genes", is_masked=False),
        config.FieldInfo("expressions", is_masked=True),
    ]
    losses = [{"field_name": "expressions", "name": "token_value"}]

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

    assert {*trainer.logged_metrics.keys()} == {
        "train/expressions_accuracy_epoch",
        "train/expressions_accuracy_step",
        "train/expressions_token_value_loss_epoch",
        "train/expressions_token_value_loss_step",
        "train/loss_epoch",
        "train/loss_step",
        "validation/expressions_accuracy",
        "validation/expressions_token_value_loss",
        "validation/loss",
    }

    total_loss = trainer.logged_metrics["train/loss_step"]
    total_token_value_loss = trainer.logged_metrics[
        "train/expressions_token_value_loss_step"
    ]

    np.testing.assert_almost_equal(total_loss, total_token_value_loss)


def test_scbert_with_pl_module_mse_loss_only():
    fields = [
        config.FieldInfo("genes", is_masked=False),
        config.FieldInfo("expressions", is_masked=True, decode_modes=["regression"]),
    ]
    losses = [{"field_name": "expressions", "name": "token_mse"}]

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

    assert {*trainer.logged_metrics.keys()} == {
        "train/expressions_token_mse_loss_epoch",
        "train/expressions_token_mse_loss_step",
        "train/loss_epoch",
        "train/loss_step",
        "validation/expressions_token_mse_loss",
        "validation/loss",
    }

    total_loss = trainer.logged_metrics["train/loss_step"]
    total_mse_loss = trainer.logged_metrics["train/expressions_token_mse_loss_step"]

    np.testing.assert_almost_equal(total_loss, total_mse_loss)


@pytest.mark.xfail()
def test_scbert_with_pl_module_token_value_loss_only_gene_masking_only():
    fields = [
        config.FieldInfo("genes", is_masked=True),
        config.FieldInfo("expressions", is_masked=False),
    ]
    losses = [
        {"field_name": "genes", "name": "token_value"},
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

    assert {*trainer.logged_metrics.keys()} == {
        "train/expressions_accuracy",
        "train/loss",
        "train/epoch_loss",
        "train/token_value_loss",
    }

    total_loss = trainer.logged_metrics["train/loss"]
    total_token_value_loss = trainer.logged_metrics["train/token_value_loss"]

    np.testing.assert_almost_equal(total_loss, total_token_value_loss)


def test_scbert_with_all_valid_losses():
    fields = [
        config.FieldInfo("genes", is_masked=True),
        config.FieldInfo(
            "expressions",
            is_masked=True,
            decode_modes=["regression", "token_scores"],
        ),
    ]
    losses = [
        {"field_name": "expressions", "name": "token_value"},
        {"field_name": "expressions", "name": "token_mse"},
        {"field_name": "expressions", "name": "cross_entropy"},
        {"field_name": "genes", "name": "cross_entropy"},
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

    assert {*trainer.logged_metrics.keys()} == {
        "train/expressions_accuracy_epoch",
        "train/expressions_accuracy_step",
        "train/expressions_cross_entropy_loss_epoch",
        "train/expressions_cross_entropy_loss_step",
        "train/expressions_perplexity_epoch",
        "train/expressions_perplexity_step",
        "train/expressions_token_mse_loss_epoch",
        "train/expressions_token_mse_loss_step",
        "train/expressions_token_value_loss_epoch",
        "train/expressions_token_value_loss_step",
        "train/genes_accuracy_epoch",
        "train/genes_accuracy_step",
        "train/genes_cross_entropy_loss_epoch",
        "train/genes_cross_entropy_loss_step",
        "train/genes_perplexity_epoch",
        "train/genes_perplexity_step",
        "train/loss_epoch",
        "train/loss_step",
        "validation/expressions_accuracy",
        "validation/expressions_cross_entropy_loss",
        "validation/expressions_perplexity",
        "validation/expressions_token_mse_loss",
        "validation/expressions_token_value_loss",
        "validation/genes_accuracy",
        "validation/genes_cross_entropy_loss",
        "validation/genes_perplexity",
        "validation/loss",
    }


def test_scbert_all_frozen_pre_computed_gene_embeddings():
    fields = set_fields_pretrained_embedding(
        str(Path(__file__).parent / "test_vocab/pre_computed_gene_embeddings_full.txt")
    )
    losses = [
        {"field_name": "genes", "name": "cross_entropy", "weight": 1},
        {"field_name": "expressions", "name": "cross_entropy", "weight": 1},
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
        {"field_name": "genes", "name": "cross_entropy", "weight": 1},
        {"field_name": "expressions", "name": "cross_entropy", "weight": 1},
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
            pretrained_embedding=config.PreTrainedEmbeddingConfig(filename=filename),
        ),
        config.FieldInfo("expressions", is_masked=True),
    ]
    return fields


def generate_and_train(
    fields,
    trainer_config,
    model_config,
    tokenizer,
    batch_size=3,
    sequence_len=512,
    return_init_gene_weights=False,
    token_fields=["genes", "expressions"],
    scalar_valued_fields=None,
):
    collator = MultiFieldCollator(
        tokenizer=tokenizer,
        mlm=True,
        pad_to_multiple_of=2,
        fields=fields,
        masker=Masker(
            change_ratio=0.2, mask_ratio=1.0, switch_ratio=0.0, tokenizer=tokenizer
        ),
        max_length=sequence_len,
    )
    if isinstance(model_config, config.SCNystromformerConfig):
        model_config.max_position_embeddings = sequence_len
    pl_module = MLMTrainingModule(model_config, trainer_config, tokenizer)
    if return_init_gene_weights:
        base_model = getattr(pl_module.model, pl_module.model.base_model_prefix)
        gene_embeddings = base_model.embeddings.genes_embeddings
        gene_embedding_init = gene_embeddings.weight.data.clone()

    dataset = helpers.generate_dataset(
        10 * batch_size,
        min_seq_len=sequence_len,
        max_seq_len=sequence_len,
        seed=42,
        token_fields=token_fields,
        scalar_valued_fields=scalar_valued_fields,
        tokenizer=tokenizer,
    )
    val_dataset = helpers.generate_dataset(
        2 * batch_size,
        min_seq_len=sequence_len,
        max_seq_len=sequence_len,
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
            max_steps=1,
            log_every_n_steps=1,
            val_check_interval=1,
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

    continuous_value_encoder_kwargs = {
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
            continuous_value_encoder_kwargs=continuous_value_encoder_kwargs,
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
        {"field_name": "genes", "name": "cross_entropy", "weight": 1},
        {"field_name": "expressions", "name": "mse", "weight": 1},
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
