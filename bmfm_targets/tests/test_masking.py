import numpy as np
import torch

from bmfm_targets.config.tokenization_config import FieldInfo
from bmfm_targets.datasets import zheng68k
from bmfm_targets.tokenization import MultiFieldInstance, load_tokenizer
from bmfm_targets.training.masking import (
    Masker,
    MaskingStrategy,
    prevent_attention_to_masked,
)
from bmfm_targets.training.masking.strategy import LABEL_SET_FUNCTIONS, WCEDMasker

from .helpers import Zheng68kPaths, load_test_tokenizer


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


def test_pattern_matching_masking_strategy(pl_zheng_mlm_raw_counts):
    dm = pl_zheng_mlm_raw_counts
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


def test_pattern_matching_masking_strategy_inside_masker(pl_zheng_mlm_raw_counts):
    dm = pl_zheng_mlm_raw_counts
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


def test_double_masking_prevented(pl_zheng_mlm_raw_counts):
    dm = zheng68k.Zheng68kDataModule(
        fields=pl_zheng_mlm_raw_counts.fields,
        tokenizer=pl_zheng_mlm_raw_counts.tokenizer,
        data_dir=pl_zheng_mlm_raw_counts.data_dir,
        label_columns=pl_zheng_mlm_raw_counts.label_columns,
        processed_name=Zheng68kPaths.raw_counts_name,
        transform_kwargs={"transforms": []},
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
    batch = {
        "field": {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "special_tokens_mask": torch.tensor([[0, 0, 0, 0]]),
        }
    }
    masked_field = FieldInfo("field")
    attention_mask = torch.tensor([[1, 1, 1, 1]])
    mask_id = 3

    expected = torch.tensor(
        [
            [
                [1, 1, 0, 1],
                [1, 1, 0, 1],
                [1, 1, 1, 1],
                [1, 1, 0, 1],
            ]
        ]
    )
    result = prevent_attention_to_masked(batch, masked_field, attention_mask, mask_id)
    assert torch.equal(result, expected)


def test_masked_tokens_self_attend():
    batch = {
        "field": {
            "input_ids": torch.tensor([[1, 2, 3, 4, 3]]),
            "special_tokens_mask": torch.tensor([[0, 0, 0, 0, 0]]),
        }
    }
    masked_field = FieldInfo("field")
    attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
    mask_id = 3

    result = prevent_attention_to_masked(batch, masked_field, attention_mask, mask_id)
    assert result[0, 2, 2] == True  # Masked token attends to itself
    assert result[0, 4, 4] == True  # Masked token attends to itself
    assert result[0, 2, 4] == False  # Masked tokens do not attend to each other


def test_padding_exclusion():
    batch = {
        "field": {
            "input_ids": torch.tensor([[1, 2, 0, 0]]),
            "special_tokens_mask": torch.tensor([[0, 0, 1, 1]]),
        }
    }
    masked_field = FieldInfo("field")
    attention_mask = torch.tensor([[1, 1, 0, 0]])
    mask_id = 3

    result = prevent_attention_to_masked(batch, masked_field, attention_mask, mask_id)
    assert result[0, 2, 2] == False  # Padding does not attend
    assert result[0, 0, 2] == False  # Non-padding does not attend to padding
    assert result[0, 1, 2] == False


def test_special_token_exclusion_for_masked_tokens():
    batch = {
        "field": {
            "input_ids": torch.tensor([[2, 4, 7, 3, 3, 0, 0]]),
            "special_tokens_mask": torch.tensor([[1, 0, 0, 0, 0, 1, 1]]),
        }
    }
    masked_field = FieldInfo("field")
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0]])
    mask_id = 3

    result = prevent_attention_to_masked(batch, masked_field, attention_mask, mask_id)
    cls_attention_mask_result = result[0, 0, :]
    pad_tokens_mask_expected = torch.tensor(
        [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
    )
    pad_tokens_mask_result = result[0, 5:, :]
    cls_attention_mask_expected = torch.Tensor([1, 1, 1, 0, 0, 0, 0])
    assert torch.equal(cls_attention_mask_result, cls_attention_mask_expected)
    assert torch.equal(pad_tokens_mask_result, pad_tokens_mask_expected)


def test_wced_masking():
    tokenizer = load_test_tokenizer()
    masker = WCEDMasker(tokenizer=tokenizer, label_sets=["non_input", "input"])

    mfi_list = [
        MultiFieldInstance(
            data={
                "genes": ["token1", "token2", "token3", "token4", "token5"],
                "expressions": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        ),
        MultiFieldInstance(
            data={
                "genes": ["token6", "token7", "token8", "token9", "token10"],
                "expressions": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        ),
    ]
    fields = [
        FieldInfo("genes"),
        FieldInfo("expressions", tokenization_strategy="continuous_value_encoder"),
    ]
    tokenized = tokenizer(mfi_list, fields=fields, return_tensors="pt", max_length=4)
    batch = {"mfi": mfi_list}
    batch.update(tokenized)

    input_ids, labels, attention_mask = masker.mask_inputs(fields, batch)
    expressions_labels = labels["expressions"]

    assert "non_input" in expressions_labels.keys()
    assert "input" in expressions_labels.keys()

    torch.testing.assert_close(
        input_ids[:, 0, :].int(), tokenized["genes"]["input_ids"].int()
    )
    genes_vocab_size = len(tokenizer.get_field_vocab("genes"))
    assert expressions_labels["non_input"].shape == (len(mfi_list), genes_vocab_size)


def test_label_select_functions():
    from torch import tensor

    input_ids = tensor(
        [
            [[3.0, 5.0, 6.0, 1.0], [3.0, 1.0, 2.0, 1.0]],
            [[3.0, 10.0, 11.0, 1.0], [3.0, 1.0, 2.0, 1.0]],
        ]
    )
    lookup_ids = tensor([5, 6, 7, 8, 9])
    lookup_vals = tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    sample_idx = 0
    batch_size = input_ids.shape[0]
    vocab_len = 19

    label_tensor_non_input = -100 * torch.ones(batch_size, vocab_len, dtype=torch.float)
    LABEL_SET_FUNCTIONS["non_input"](
        label_tensor=label_tensor_non_input,
        sample_idx=sample_idx,
        lookup_ids=lookup_ids,
        lookup_vals=lookup_vals,
        input_ids=input_ids,
    )
    input_mask = torch.isin(lookup_ids, input_ids[sample_idx, 0])
    non_input_mask = ~input_mask
    assert label_tensor_non_input[sample_idx, lookup_ids[non_input_mask]].equal(
        lookup_vals[non_input_mask]
    )
    assert (label_tensor_non_input[sample_idx, lookup_ids[input_mask]] == -100).all()

    label_tensor_input = -100 * torch.ones(batch_size, vocab_len, dtype=torch.float)
    LABEL_SET_FUNCTIONS["input"](
        label_tensor=label_tensor_input,
        sample_idx=sample_idx,
        lookup_ids=lookup_ids,
        lookup_vals=lookup_vals,
        input_ids=input_ids,
    )
    input_mask = torch.isin(lookup_ids, input_ids[sample_idx, 0])
    non_input_mask = ~input_mask
    assert label_tensor_input[sample_idx, lookup_ids[input_mask]].equal(
        lookup_vals[input_mask]
    )
    assert (label_tensor_input[sample_idx, lookup_ids[non_input_mask]] == -100).all()

    label_tensor_all = -100 * torch.ones(batch_size, vocab_len, dtype=torch.float)
    LABEL_SET_FUNCTIONS["all"](
        label_tensor=label_tensor_all,
        sample_idx=sample_idx,
        lookup_ids=lookup_ids,
        lookup_vals=lookup_vals,
        input_ids=input_ids,
    )
    assert label_tensor_all[sample_idx, lookup_ids].equal(lookup_vals)
