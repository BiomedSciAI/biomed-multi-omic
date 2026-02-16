import pytest
import torch
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import PaddingStrategy

from bmfm_targets.config import FieldInfo, LabelColumnInfo
from bmfm_targets.tokenization import (
    MultiFieldCollator,
    MultiFieldInstance,
    get_snp2vec_tokenizer,
    load_tokenizer,
)
from bmfm_targets.training.masking import Masker

from .helpers import (
    generate_dataset,
    generate_hic_dataset,
    generate_sequence_labeling_perturbation_dataset,
    load_test_tokenizer,
    round_up_to_nearest_multiple_of_10,
)


def test_geneformer_collator(geneformer_gene2vec_fields):
    seq_len = 116
    batch_size = 16
    tokenizer = load_test_tokenizer()
    dataset = generate_dataset(batch_size * 10, seq_len, seq_len, seed=42)

    data_collator = MultiFieldCollator(
        tokenizer,
        pad_to_multiple_of=10,
        fields=geneformer_gene2vec_fields,
        sequence_order=None,
    )

    sorted_data_collator = MultiFieldCollator(
        tokenizer,
        pad_to_multiple_of=10,
        fields=geneformer_gene2vec_fields,
        sequence_order="sorted",
    )

    data_loader = DataLoader(
        dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=False
    )

    sorted_data_loader = DataLoader(
        dataset, collate_fn=sorted_data_collator, batch_size=batch_size, shuffle=False
    )

    for batch, sorted_batch in zip(data_loader, sorted_data_loader):
        input_ids = batch["input_ids"].squeeze()
        sorted_input_ids = sorted_batch["input_ids"].squeeze()
        assert input_ids.shape == (
            batch_size,
            round_up_to_nearest_multiple_of_10(seq_len + 2),
        )
        for instance_input_ids, instance_sorted_input_ids in zip(
            input_ids, sorted_input_ids
        ):
            assert set(instance_input_ids.tolist()) == set(
                instance_sorted_input_ids.tolist()
            )
            assert instance_input_ids.tolist() != instance_sorted_input_ids.tolist()


def test_collator(fields):
    seq_len = 116
    batch_size = 16
    n_fields = 2
    tokenizer = load_test_tokenizer()
    dataset = generate_dataset(batch_size * 10, seq_len, seq_len, seed=42)
    masker = Masker(
        change_ratio=0.15, mask_ratio=1.0, switch_ratio=0.0, tokenizer=tokenizer
    )

    data_collator = MultiFieldCollator(
        tokenizer,
        masker=masker,
        pad_to_multiple_of=10,
        fields=fields,
    )
    data_loader = DataLoader(
        dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=True
    )

    for batch in data_loader:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        assert (labels["expressions"] == -100).numpy().mean() > 0
        assert input_ids.shape == (
            batch_size,
            n_fields,
            round_up_to_nearest_multiple_of_10(seq_len + 2),
        )


def test_prevent_attention_to_masked(fields):
    seq_len = 10
    batch_size = 1
    n_fields = 2
    tokenizer = load_test_tokenizer()
    dataset = generate_dataset(batch_size * 10, seq_len, seq_len, seed=42)
    masker = Masker(
        change_ratio=0.50,
        mask_ratio=1.0,
        switch_ratio=0.0,
        tokenizer=tokenizer,
        prevent_attention_to_masked=True,
    )

    data_collator = MultiFieldCollator(
        tokenizer,
        masker=masker,
        pad_to_multiple_of=10,
        fields=fields,
    )
    data_loader = DataLoader(
        dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=True
    )

    for batch in data_loader:
        expression_input_ids = batch["input_ids"][:, 1]
        labels = batch["labels"]
        is_masked = expression_input_ids == 4
        is_pad = expression_input_ids == 2
        is_unmasked = ~(is_masked | is_pad)
        row_sums = torch.sum(batch["attention_mask"][is_masked], dim=1)
        first_row_sum = row_sums[0]
        assert not torch.any(batch["attention_mask"][is_pad])
        assert torch.any(batch["attention_mask"][is_unmasked])
        assert torch.all(row_sums == first_row_sum)
        assert first_row_sum == torch.sum(is_unmasked) - 1
        assert (labels["expressions"] == -100).numpy().mean() > 0


def test_collator_no_masking(fields):
    seq_len = 120
    batch_size = 16
    n_fields = 2
    tokenizer = load_test_tokenizer()
    dataset = generate_dataset(batch_size * 10, seq_len, seq_len, seed=42)
    data_collator = MultiFieldCollator(tokenizer, pad_to_multiple_of=10, fields=fields)
    data_loader = DataLoader(
        dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=True
    )

    for batch in data_loader:
        input_ids = batch["input_ids"]
        assert "labels" not in batch
        assert input_ids.shape == (
            batch_size,
            n_fields,
            round_up_to_nearest_multiple_of_10(seq_len + 2),
        )


def test_padding_on_samples_of_different_length(test_tokenizer_fields):
    batch_size = 16
    n_fields = 2
    min_seq_len = 4
    max_seq_len = 10
    dataset = generate_dataset(batch_size * 10, min_seq_len, max_seq_len, seed=42)
    tokenizer = load_test_tokenizer()
    masker = Masker(
        change_ratio=0.15, mask_ratio=1.0, switch_ratio=0, tokenizer=tokenizer
    )
    data_collator = MultiFieldCollator(
        tokenizer,
        masker=masker,
        padding=PaddingStrategy.LONGEST,
        pad_to_multiple_of=4,
        fields=test_tokenizer_fields,
    )
    data_loader = DataLoader(
        dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=True
    )

    for batch in data_loader:
        input_ids = batch["input_ids"]
        assert input_ids.shape == (batch_size, n_fields, 12)


def test_masking(fields, test_tokenizer_fields_multimask):
    seq_len = 120
    batch_size = 16
    tokenizer = load_test_tokenizer()
    dataset = generate_dataset(batch_size * 100, seq_len, seq_len, seed=42)

    pre_data_collator = MultiFieldCollator(
        tokenizer, pad_to_multiple_of=10, fields=fields
    )

    masking_masker = Masker(
        change_ratio=0.15, mask_ratio=1.0, switch_ratio=0.0, tokenizer=tokenizer
    )
    switching_masker = Masker(
        change_ratio=0.15, mask_ratio=0.0, switch_ratio=1.0, tokenizer=tokenizer
    )
    switching_and_masking_masker = Masker(
        change_ratio=0.15, mask_ratio=0.5, switch_ratio=0.5, tokenizer=tokenizer
    )

    comask_fields_masker = Masker(
        change_ratio=0.50,
        mask_ratio=1.0,
        switch_ratio=0.0,
        tokenizer=tokenizer,
        comask_across_fields=True,
    )

    collator = lambda masker: MultiFieldCollator(
        tokenizer, masker=masker, pad_to_multiple_of=10, fields=fields
    )

    comask_fields_collator = lambda masker: MultiFieldCollator(
        tokenizer,
        masker=masker,
        pad_to_multiple_of=10,
        fields=test_tokenizer_fields_multimask,
    )

    pre_data_loader = DataLoader(
        dataset, collate_fn=pre_data_collator, batch_size=batch_size, shuffle=False
    )

    mask_data_loader = DataLoader(
        dataset,
        collate_fn=collator(masking_masker),
        batch_size=batch_size,
        shuffle=False,
    )

    switch_data_loader = DataLoader(
        dataset,
        collate_fn=collator(switching_masker),
        batch_size=batch_size,
        shuffle=False,
    )

    mask_and_switch_data_loader = DataLoader(
        dataset,
        collate_fn=collator(switching_and_masking_masker),
        batch_size=batch_size,
        shuffle=False,
    )

    comask_fields_data_loader = DataLoader(
        dataset,
        collate_fn=comask_fields_collator(comask_fields_masker),
        batch_size=batch_size,
        shuffle=False,
    )

    pre_batch = next(iter(pre_data_loader))
    mask_batch = next(iter(mask_data_loader))
    switch_batch = next(iter(switch_data_loader))
    mask_and_switch_batch = next(iter(mask_and_switch_data_loader))
    comasked_fields_batch = next(iter(comask_fields_data_loader))

    pre_input_ids = pre_batch["input_ids"]
    mask_input_ids = mask_batch["input_ids"]
    switch_input_ids = switch_batch["input_ids"]
    mask_and_switch_ids = mask_and_switch_batch["input_ids"]
    comasked_fields_input_ids = comasked_fields_batch["input_ids"]

    # number of tokens ignoring padding
    num_tokens = (pre_input_ids != tokenizer.pad_token_id).numpy().sum()

    # check for masked tokens no switch
    changed_tokens = (pre_input_ids != mask_input_ids).numpy().sum()
    # assert 0.10 <= changed_tokens / num_tokens <= 0.20
    masked_tokens = (mask_input_ids == tokenizer.mask_token_id).numpy().sum()
    assert changed_tokens == masked_tokens

    # check for switched tokens no mask
    changed_tokens = (pre_input_ids != switch_input_ids).numpy().sum()
    masked_tokens = (switch_input_ids == tokenizer.mask_token_id).numpy().sum()
    switched_tokens = (pre_input_ids != switch_input_ids).numpy().sum() - masked_tokens
    # assert 0.10 <= changed_tokens / num_tokens <= 0.20
    assert masked_tokens == 0
    assert switched_tokens > 0

    # check for masked and switched tokens
    changed_tokens = (pre_input_ids != mask_and_switch_ids).numpy().sum()
    masked_tokens = (mask_and_switch_ids == tokenizer.mask_token_id).numpy().sum()
    switched_tokens = (
        pre_input_ids != mask_and_switch_ids
    ).numpy().sum() - masked_tokens
    assert changed_tokens > 0
    assert masked_tokens > 0
    assert switched_tokens > 0
    # assert 0.10 <= changed_tokens / num_tokens <= 0.20

    # check that both fields are masked in the same token indices
    gene_mask_matches = comasked_fields_input_ids[:, 0, :] == 4
    expresion_mask_matches = comasked_fields_input_ids[:, 1, :] == 4
    assert torch.equal(gene_mask_matches, expresion_mask_matches)


def test_sequence_labeling_collator(perturbation_fields_tokenized):
    seq_len = 120
    batch_size = 16
    n_fields = 3
    tokenizer = load_test_tokenizer()

    dataset = generate_sequence_labeling_perturbation_dataset(
        size=batch_size * 10,
        min_seq_len=seq_len,
        max_seq_len=seq_len,
        fields=perturbation_fields_tokenized,
        seed=42,
    )
    data_collator = MultiFieldCollator(
        tokenizer,
        pad_to_multiple_of=10,
        fields=perturbation_fields_tokenized,
        collation_strategy="sequence_labeling",
    )
    data_loader = DataLoader(
        dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=True
    )

    for batch in data_loader:
        input_ids = batch["input_ids"]
        assert (input_ids[:, 1, 1:-1] != 0).all()
        assert "label_expressions" in batch["labels"]
        assert input_ids.shape == (
            batch_size,
            n_fields,
            round_up_to_nearest_multiple_of_10(seq_len + 2),
        )
        assert batch["labels"]["label_expressions"].shape == (
            batch_size,
            round_up_to_nearest_multiple_of_10(seq_len + 2),
        )


def test_downsample_sequence_labeling_collator(
    label_expression_fields_gene2vec, pl_data_module_zheng68k_seq_cls
):
    batch_size = 4
    from bmfm_targets.tokenization import load_tokenizer

    tokenizer = load_tokenizer("gene2vec")

    dataset = pl_data_module_zheng68k_seq_cls.train_dataset

    data_collator = MultiFieldCollator(
        tokenizer,
        pad_to_multiple_of=2,
        fields=label_expression_fields_gene2vec,
        rda_transform="downsample",
        collation_strategy="sequence_labeling",
        max_length=16,
    )
    data_loader = DataLoader(
        dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=True
    )

    for batch in data_loader:
        input_ids = batch["input_ids"]
        assert "label_expressions" in batch["labels"]
        assert input_ids.shape[1] == 2
        assert batch["labels"]["label_expressions"].shape[1] == 16


def test_downsample_language_modeling_collator(
    all_genes_fields_with_rda_regression_masking, pl_data_module_panglao_rda
):
    batch_size = 4
    tokenizer = load_tokenizer("all_genes")
    dataset = pl_data_module_panglao_rda.train_dataset

    data_collator = MultiFieldCollator(
        tokenizer,
        pad_to_multiple_of=2,
        masker=Masker(0.3, 1, 0, tokenizer),
        fields=all_genes_fields_with_rda_regression_masking,
        rda_transform="downsample",
        max_length=16,
        padding="max_length",
    )

    data_loader = DataLoader(
        dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=True
    )

    for batch in data_loader:
        assert batch["input_ids"].shape[1] == 2
        assert "expressions" in batch["labels"]
        assert batch["input_ids"].shape[2] == 16


@pytest.mark.skip(reason="sps integration fails, requires additional review")
def test_downsample_language_modeling_collator_batchwise_pad_interleave_half_zeros(
    all_genes_fields_with_rda_regression_masking, pl_data_module_panglao_rda
):
    batch_size = 16
    tokenizer = load_tokenizer("all_genes")
    dataset = pl_data_module_panglao_rda.train_dataset

    data_collator = MultiFieldCollator(
        tokenizer,
        pad_to_multiple_of=2,
        fields=all_genes_fields_with_rda_regression_masking,
        collation_strategy="language_modeling",
        pad_zero_expression_strategy={
            "strategy": "batch_wise",
            "interleave_zero_ratio": 0.5,
        },
        max_length=512,
    )

    data_loader = DataLoader(
        dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=False
    )

    for batch in data_loader:
        assert (batch["input_ids"][:, 1, 2::2] == 0).all()
        break


def test_no_tokenization_preserves_input(gene2vec_fields_regression_no_tokenization):
    fields = gene2vec_fields_regression_no_tokenization

    expression_vals = [1.2344, 45.3438]
    mfi = MultiFieldInstance(
        data={"genes": ["ETS1", "SARM1"], "expressions": expression_vals},
        metadata={"cell_name": "fake"},
    )

    tokenizer = load_tokenizer("gene2vec")

    collator = MultiFieldCollator(
        tokenizer=tokenizer, fields=fields, pad_to_multiple_of=6
    )

    tokenized = collator([mfi])
    expression_input_ids = tokenized["input_ids"][0, 1]
    gene_input_ids = tokenized["input_ids"][0, 0]
    # slot 0 is CLS, slots 1 and 2 should match the floating point expression values
    expression_vals_tensor = torch.tensor(expression_vals, dtype=torch.float32)
    torch.testing.assert_close(
        expression_input_ids[1:3], expression_vals_tensor, rtol=1e-3, atol=1e-5
    )
    # meanwhile gene names should be tokenized correctly (not UNK)
    assert all(gene_input_ids[1:3] > len(tokenizer.all_special_tokens))


def test_no_tokenization_makes_special_tokens_negative(
    gene2vec_fields_regression_no_tokenization,
):
    fields = gene2vec_fields_regression_no_tokenization

    expression_vals = [1.231, 45.3453]
    mfi = MultiFieldInstance(
        data={"genes": ["ETS1", "SARM1"], "expressions": expression_vals},
        metadata={"cell_name": "fake"},
    )

    tokenizer = load_tokenizer("gene2vec")

    collator = MultiFieldCollator(
        tokenizer=tokenizer, fields=fields, pad_to_multiple_of=6
    )

    tokenized = collator([mfi])
    expression_input_ids = tokenized["input_ids"][0, 1]
    gene_input_ids = tokenized["input_ids"][0, 0]
    inverted_positions = expression_input_ids == -(gene_input_ids + 1)
    assert inverted_positions.tolist() == [True, False, False, True, True, True]


def test_hic_collator(snp2vec_fields):
    seq_len = 116
    batch_size = 16
    n_fields = 1
    tokenizer = get_snp2vec_tokenizer()

    dataset = generate_hic_dataset(batch_size * 10, seq_len, seq_len, seed=42)
    masker = Masker(
        change_ratio=0.15, mask_ratio=1.0, switch_ratio=0.0, tokenizer=tokenizer
    )

    label_dict = {"hic_contact": {0: 0}}
    data_collator = MultiFieldCollator(
        tokenizer,
        masker=masker,
        pad_to_multiple_of=10,
        fields=snp2vec_fields,
        label_columns=[
            LabelColumnInfo(label_column_name="hic_contact", is_regression_label=True)
        ],
        label_dict=label_dict,
        collation_strategy="multitask",
    )
    data_loader = DataLoader(
        dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=True
    )
    for batch in data_loader:
        input_ids = batch["input_ids"]
        labels = batch["labels"]["dna_chunks"]
        assert "hic_contact" in batch["labels"]
        assert (labels == -100).numpy().mean() > 0
        assert input_ids.shape == (
            batch_size,
            n_fields,
            round_up_to_nearest_multiple_of_10(seq_len * 2 + 2),
        )


def test_pad_zero_expression_strategy_in_collator():
    seq_len = 116
    batch_size = 16
    fields = [
        FieldInfo("genes"),
        FieldInfo(
            "expressions",
            is_masked=True,
            tokenization_strategy="continuous_value_encoder",
            decode_modes={"regression": {}},
        ),
    ]
    tokenizer = load_test_tokenizer()
    tokenizer.default_field = "genes"
    dataset = generate_dataset(
        batch_size * 10,
        seq_len,
        seq_len,
        seed=42,
        token_fields=["genes"],
        scalar_valued_fields=["expressions"],
        zero_fraction=0.1,
    )
    masker = Masker(
        change_ratio=0.15, mask_ratio=1.0, switch_ratio=0.0, tokenizer=tokenizer
    )

    data_collator = MultiFieldCollator(
        tokenizer,
        masker=masker,
        pad_to_multiple_of=10,
        fields=fields,
        pad_zero_expression_strategy={"strategy": "batch_wise"},
    )
    data_loader = DataLoader(
        dataset, collate_fn=data_collator, batch_size=batch_size, shuffle=True
    )

    for batch in data_loader:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        assert (labels["expressions"] == -100).numpy().mean() > 0
        assert (input_ids[:, 1, :] == 0).any()
        assert not (input_ids[:, 0, :] == 0).any()


def test_sequence_classification_labels_exposed_correctly():
    labels = [LabelColumnInfo("test_label")]
    label_dict = {"test_label": {"val1": 0, "val2": 1, "val3": 2}}
    dataset = generate_dataset(100, label_dict=label_dict)
    tokenizer = load_tokenizer("all_genes")
    fields = [FieldInfo("genes"), FieldInfo("expressions")]
    mfc = MultiFieldCollator(
        tokenizer,
        fields,
        label_dict=label_dict,
        label_columns=labels,
        collation_strategy="sequence_classification",
    )
    cols = mfc(dataset[:])
    observed_label_ids = {*cols["labels"]["test_label"].numpy()}
    available_label_ids = {*label_dict["test_label"].values()}
    assert observed_label_ids == available_label_ids


def test_sequence_classification_silent_labels_identified_correctly():
    labels = [LabelColumnInfo("test_label", silent_label_values=["val3"])]
    label_dict = {"test_label": {"val1": 0, "val2": 1, "val3": 2}}
    dataset = generate_dataset(100, label_dict=label_dict)
    tokenizer = load_tokenizer("all_genes")
    fields = [FieldInfo("genes"), FieldInfo("expressions")]
    mfc = MultiFieldCollator(
        tokenizer,
        fields,
        label_dict=label_dict,
        label_columns=labels,
        collation_strategy="sequence_classification",
    )
    cols = mfc(dataset[:])
    observed_label_ids = {*cols["labels"]["test_label"].numpy()}
    available_label_ids = {*label_dict["test_label"].values()}
    assert observed_label_ids - available_label_ids == {-100}
    silent_id = label_dict["test_label"][labels[0].silent_label_values[0]]
    assert available_label_ids - observed_label_ids == {silent_id}


def test_read_label_ids_for_multilabel():
    example = MultiFieldInstance(data={}, metadata={"test_label": "a|b|e"})
    label_info = LabelColumnInfo("test_label", is_multilabel=True)
    single_label_dict = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
    label_ids = MultiFieldCollator._label_id_from_example(
        example, label_info, single_label_dict
    )
    assert [1.0, 1.0, 0.0, 0.0, 1.0] == label_ids


def test_read_label_ids_for_regression():
    example = MultiFieldInstance(data={}, metadata={"test_label": 1.23})
    label_info = LabelColumnInfo("test_label", is_regression_label=True)
    single_label_dict = {}  # can be anything, not used
    label_id = MultiFieldCollator._label_id_from_example(
        example, label_info, single_label_dict
    )
    assert 1.23 == label_id


def test_read_label_ids_for_single_label():
    example = MultiFieldInstance(data={}, metadata={"test_label": "a"})
    label_info = LabelColumnInfo("test_label")
    single_label_dict = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
    label_id = MultiFieldCollator._label_id_from_example(
        example, label_info, single_label_dict
    )
    assert 0 == label_id


def test_missing_label_mapped_to_minus_100():
    example = MultiFieldInstance(data={}, metadata={"test_label": "mistake"})
    label_info = LabelColumnInfo("test_label")
    single_label_dict = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
    label_id = MultiFieldCollator._label_id_from_example(
        example, label_info, single_label_dict
    )
    assert -100 == label_id


def test_nan_mapped_to_minus_100():
    example = MultiFieldInstance(data={}, metadata={"test_label": "nan"})
    label_info = LabelColumnInfo("test_label")
    single_label_dict = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
    label_id = MultiFieldCollator._label_id_from_example(
        example, label_info, single_label_dict
    )
    assert -100 == label_id
