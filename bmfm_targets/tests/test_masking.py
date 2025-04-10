import tempfile

import numpy as np
import torch

from bmfm_targets import config
from bmfm_targets.config.tokenization_config import FieldInfo
from bmfm_targets.datasets import zheng68k
from bmfm_targets.tasks.task_utils import make_trainer_for_task, train
from bmfm_targets.tests.helpers import get_test_task_config
from bmfm_targets.tokenization import load_tokenizer
from bmfm_targets.training.masking import (
    Masker,
    MaskingStrategy,
    prevent_attention_to_masked,
)
from bmfm_targets.training.modules import MLMTrainingModule

from .helpers import MockTestDataPaths


def test_mask_single_field():
    masker = Masker(
        change_ratio=0.3,
        mask_ratio=0.9,
        switch_ratio=0,
        tokenizer=load_tokenizer("gene2vec"),
    )
    field = FieldInfo("genes")
    gen = torch.manual_seed(42)
    input_ids = torch.randint(0, 100, size=(10000,), generator=gen)
    specials_mask = torch.zeros_like(input_ids)
    specials_mask[input_ids < len(masker.tokenizer.all_special_ids)] = 1
    field_encoding = {"input_ids": input_ids, "special_tokens_mask": specials_mask}

    random_tensor = torch.rand_like(
        input_ids,
        layout=torch.strided,
        dtype=torch.float,
        device=input_ids.device,
    )
    inputs, labels = masker.mask_single_field(field, field_encoding, random_tensor)

    mask_count = input_ids.shape[0] * masker.change_ratio * masker.mask_ratio
    tol = 100
    # expected number of tokens are masked
    masked = inputs == masker.tokenizer.mask_token_id
    active_labels = labels != -100
    assert abs((masked & active_labels).sum() - mask_count) < tol
    # expected number of tokens are not masked
    non_mask_ratio = masker.change_ratio * (1 - masker.mask_ratio - masker.switch_ratio)
    non_mask_count = input_ids.shape[0] * non_mask_ratio
    assert abs((~masked & active_labels).sum() - non_mask_count) < tol
    # no special tokens are masked
    assert (labels[specials_mask.bool()] == -100).all()
    # probs should be about the same on both sides
    assert (
        abs(masked[masked.shape[0] // 2 :].sum() - masked[: masked.shape[0] // 2].sum())
        < tol
    )


def test_mask_single_field_with_mask_probs():
    masker = Masker(
        change_ratio=0.3,
        mask_ratio=0.9,
        switch_ratio=0,
        tokenizer=load_tokenizer("gene2vec"),
    )
    field = FieldInfo("genes")
    gen = torch.manual_seed(42)
    input_ids = torch.randint(0, 100, size=(10000,), generator=gen)
    specials_mask = torch.zeros_like(input_ids)
    specials_mask[input_ids < len(masker.tokenizer.all_special_ids)] = 1
    field_encoding = {"input_ids": input_ids, "special_tokens_mask": specials_mask}

    mask_probs = torch.arange(0, input_ids.shape[0]).float()

    random_tensor = torch.rand_like(
        input_ids,
        layout=torch.strided,
        dtype=torch.float,
        device=input_ids.device,
    )
    inputs, labels = masker.mask_single_field(
        field, field_encoding, random_tensor, mask_probs
    )

    mask_count = input_ids.shape[0] * masker.change_ratio * masker.mask_ratio
    tol = 100
    # expected number of tokens are masked
    masked = inputs == masker.tokenizer.mask_token_id
    active_labels = labels != -100
    assert abs((masked & active_labels).sum() - mask_count) < tol
    # expected number of tokens are not masked
    non_mask_ratio = masker.change_ratio * (1 - masker.mask_ratio - masker.switch_ratio)
    non_mask_count = input_ids.shape[0] * non_mask_ratio
    assert abs((~masked & active_labels).sum() - non_mask_count) < tol
    # no special tokens are masked
    assert (labels[specials_mask] == -100).all()

    # probs should be higher to the right
    assert (
        masked[masked.shape[0] // 2 :].sum() - masked[: masked.shape[0] // 2].sum()
        > tol
    )


def test_pattern_matching_masking_strategy(pl_mock_data_mlm_no_binning):
    dm = pl_mock_data_mlm_no_binning
    ms = MaskingStrategy(tokenizer=dm.tokenizer, pattern_weights=[("^RP", 0.5)])
    mfis = dm.train_dataset[:5]
    batch = dm.collate_fn.tokenize_batch(mfis)
    masking_probs = ms.get_mask_probs(batch)

    input_ids = batch["genes"]["input_ids"]
    tokens = ["x"] * len(dm.tokenizer.get_field_vocab("genes"))
    for k, v in dm.tokenizer.get_field_vocab("genes").items():
        tokens[v] = k
    tokens = np.array(tokens)
    down_prob_tokens = tokens[input_ids[(masking_probs < 1).nonzero(as_tuple=True)]]
    full_prob_tokens = tokens[input_ids[(masking_probs == 1).nonzero(as_tuple=True)]]
    assert all(x.startswith("RP") for x in down_prob_tokens)
    assert not any(x.startswith("RP") for x in full_prob_tokens)

    assert (masking_probs > 0).all()


def test_pattern_matching_masking_strategy_inside_masker(pl_mock_data_mlm_no_binning):
    dm = pl_mock_data_mlm_no_binning
    dm.masker = Masker(
        change_ratio=0.6,
        mask_ratio=0.9,
        switch_ratio=0.0,
        tokenizer=dm.tokenizer,
        masking_strategy=MaskingStrategy(
            tokenizer=dm.tokenizer, pattern_weights=[("^RP", 0.0)]
        ),
    )
    for batch in dm.train_dataloader():
        expression_input_ids = batch["input_ids"][:, 1, :]
        gene_input_ids = batch["input_ids"][:, 0, :]
        break

    tokens = ["x"] * len(dm.tokenizer.get_field_vocab("genes"))
    for k, v in dm.tokenizer.get_field_vocab("genes").items():
        tokens[v] = k
    tokens = np.array(tokens)
    masked_gene_tokens = tokens[
        gene_input_ids[(expression_input_ids == -5).nonzero(as_tuple=True)]
        .int()
        .numpy()
    ]
    non_msked_gene_tokens = tokens[
        gene_input_ids[(expression_input_ids != --5).nonzero(as_tuple=True)]
        .int()
        .numpy()
    ]
    assert sum(x.startswith("RP") for x in masked_gene_tokens) == 0
    assert any(x.startswith("RP") for x in non_msked_gene_tokens)


def test_can_update_masking_probs(pl_mock_data_mlm_no_binning):
    dm = pl_mock_data_mlm_no_binning
    ms = MaskingStrategy(tokenizer=dm.tokenizer, should_update_from_errors=True)
    mfis = dm.train_dataset[:5]
    batch = dm.collate_fn.tokenize_batch(mfis)
    masking_probs = ms.get_mask_probs(batch)

    genes_to_upweight = mfis[0]["genes"][:5]
    genes_to_downweight = mfis[0]["genes"][5:10]

    updated_token_probs = {}
    for u, d in zip(genes_to_upweight, genes_to_downweight):
        updated_token_probs[u] = 2.0
        updated_token_probs[d] = 0.5

    ms.update_token_masking_probs(updated_token_probs)
    updated_masking_probs = ms.get_mask_probs(batch)
    assert not (updated_masking_probs == masking_probs).all()


def test_updatable_token_masking_prob_masking_strategy_inside_masker(
    pl_mock_data_mlm_no_binning,
):
    dm = zheng68k.Zheng68kDataModule(
        fields=pl_mock_data_mlm_no_binning.fields,
        tokenizer=pl_mock_data_mlm_no_binning.tokenizer,
        data_dir=pl_mock_data_mlm_no_binning.data_dir,
        processed_name=MockTestDataPaths.no_binning_name,
        transform_kwargs={"transforms": []},
        dataset_kwargs={"source_h5ad_file_name":"mock_test_data.h5ad"},
        masking_strategy=MaskingStrategy(
            tokenizer=pl_mock_data_mlm_no_binning.tokenizer,
            should_update_from_errors=True,
        ),
        transform_datasets=False,
        batch_size=4,
        limit_dataset_samples={"train": 32, "dev": 256},
        mlm=True,
        collation_strategy="language_modeling",
        rda_transform=2000,
        sequence_order="random",
        max_length=64,
        pad_to_multiple_of=2,
    )
    dm.prepare_data()
    dm.setup("fit")

    model_config = config.SCBertConfig(
        fields=dm.fields,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=32,
        hidden_size=16,
        pad_token_id=2,
    )
    trainer_config = config.TrainerConfig(
        batch_size=1,
        losses=[
            {"field_name": "genes", "name": "cross_entropy", "weight": 1},
            {
                "field_name": "expressions",
                "name": "mse",
                "ignore_zero": True,
                "weight": 1,
            },
            {"field_name": "expressions", "name": "is_zero_bce", "weight": 1},
        ],
        batch_prediction_behavior="track",
    )
    masking_strategy = dm.masking_strategy
    assert masking_strategy is not None
    p = masking_strategy.token_masking_probs
    assert all(p == 1)
    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = get_test_task_config(tmpdir)
        pl_trainer = make_trainer_for_task(task_config)
        pl_module = MLMTrainingModule(model_config, trainer_config, dm.tokenizer)
        train(
            pl_trainer, pl_data_module=dm, pl_module=pl_module, task_config=task_config
        )

    masking_strategy = dm.masking_strategy
    p = masking_strategy.token_masking_probs
    assert any(p < 1)


def test_masking_strategy_can_be_turned_off_in_val(pl_mock_data_mlm_no_binning):
    dm = zheng68k.Zheng68kDataModule(
        fields=pl_mock_data_mlm_no_binning.fields,
        tokenizer=pl_mock_data_mlm_no_binning.tokenizer,
        data_dir=pl_mock_data_mlm_no_binning.data_dir,
        processed_name=MockTestDataPaths.no_binning_name,
        transform_kwargs={"transforms": []},
        dataset_kwargs={"source_h5ad_file_name":"mock_test_data.h5ad"},
        masking_strategy=MaskingStrategy(
            tokenizer=pl_mock_data_mlm_no_binning.tokenizer,
            should_update_from_errors=True,
            use_for_validation=False,
        ),
        transform_datasets=False,
        batch_size=4,
        limit_dataset_samples={"train": 32, "dev": 256},
        mlm=True,
        collation_strategy="language_modeling",
        rda_transform=2000,
        sequence_order="random",
        max_length=64,
        pad_to_multiple_of=2,
    )
    dm.prepare_data()
    dm.setup("fit")
    val_dl = dm.val_dataloader()
    assert val_dl.collate_fn.masker.masking_strategy is None

    train_dl = dm.train_dataloader()
    assert train_dl.collate_fn.masker.masking_strategy is not None


def test_double_masking_prevented(pl_mock_data_mlm_no_binning):
    dm = zheng68k.Zheng68kDataModule(
        fields=pl_mock_data_mlm_no_binning.fields,
        tokenizer=pl_mock_data_mlm_no_binning.tokenizer,
        data_dir=pl_mock_data_mlm_no_binning.data_dir,
        processed_name=MockTestDataPaths.no_binning_name,
        transform_kwargs={"transforms": []},
        dataset_kwargs={"source_h5ad_file_name":"mock_test_data.h5ad"},
        transform_datasets=False,
        batch_size=16,
        limit_dataset_samples=64,
        mlm=True,
        collation_strategy="language_modeling",
        sequence_order="random",
        max_length=64,
        pad_to_multiple_of=2,
    )
    dm.prepare_data()
    dm.setup("fit")
    mask_id = dm.tokenizer.mask_token_id
    for batch in dm.train_dataloader():
        masked_genes = batch["input_ids"][:, 0] == mask_id
        masked_expressions = batch["input_ids"][:, 1] == -(mask_id + 1)
        assert masked_genes.sum() > 0
        assert masked_expressions.sum() > 0
        assert not (masked_genes & masked_expressions).any()


def test_basic_attention():
    batch = {"field": {"input_ids": torch.tensor([[1, 2, 3, 4]])}}
    masked_field = FieldInfo("field")
    attention_mask = torch.tensor([[1, 1, 1, 1]])
    mask_id = 3

    expected = torch.tensor(
        [
            [
                [True, True, False, True],
                [True, True, False, True],
                [True, True, True, True],
                [True, True, False, True],
            ]
        ]
    )
    result = prevent_attention_to_masked(batch, masked_field, attention_mask, mask_id)
    assert torch.equal(result, expected)


def test_masked_tokens_self_attend():
    batch = {"field": {"input_ids": torch.tensor([[1, 2, 3, 4, 3]])}}
    masked_field = FieldInfo("field")
    attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
    mask_id = 3

    result = prevent_attention_to_masked(batch, masked_field, attention_mask, mask_id)
    assert result[0, 2, 2] == True  # Masked token attends to itself
    assert result[0, 4, 4] == True  # Masked token attends to itself
    assert result[0, 2, 4] == False  # Masked tokens do not attend to each other


def test_padding_exclusion():
    batch = {"field": {"input_ids": torch.tensor([[1, 2, 0, 0]])}}
    masked_field = FieldInfo("field")
    attention_mask = torch.tensor([[1, 1, 0, 0]])
    mask_id = -1

    result = prevent_attention_to_masked(batch, masked_field, attention_mask, mask_id)
    assert result[0, 2, 2] == False  # Padding does not attend
    assert result[0, 0, 2] == False  # Non-padding does not attend to padding
    assert result[0, 1, 2] == False
