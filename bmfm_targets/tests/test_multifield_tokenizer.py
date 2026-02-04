import os
import tempfile
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest
import torch
from torch import tensor  # pylint: disable=E0611
from transformers import AutoConfig
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from bmfm_targets.config.main_config import SCBertMainConfig
from bmfm_targets.tests.helpers import generate_dataset, load_test_tokenizer
from bmfm_targets.tokenization import (
    MultiFieldInstance,
    MultiFieldTokenizer,
    load_tokenizer,
)
from bmfm_targets.tokenization.create.create_gene2vec_tokenizer import (
    create_new_mft_with_extra_cls_tokens,
)
from bmfm_targets.tokenization.multifield_tokenizer import set_custom_cls_post_processor

TEST_TOKENIZER_ROOT = Path(__file__).parent / "resources/tokenizers"
TOKENIZER_ROOT = Path(__file__).parents[1] / "tokenization"
OLD_TOKENIZER_PATH = TOKENIZER_ROOT / "gene2vec_vocab"


TEST_MULTIFIELD_TOKENIZER_PATH = TEST_TOKENIZER_ROOT / "multifield_test_tokenizer"

#
MODEL_SNAPSHOT_FOR_LOADING_SAVED_TOKENIZER = Path(
    "/dccstor/bmfm-targets/models/omics/transcriptome/scRNA/pretrain/bmfm.targets.slate.bert.110m.scRNA.pretrained.multifield.mask.multiloss.v2"
)

CCC_TOKENIZER_ACCESSIBLE = MODEL_SNAPSHOT_FOR_LOADING_SAVED_TOKENIZER.exists()


@pytest.fixture()
def test_tokenizer():
    return load_test_tokenizer()


def test_test_tokenzier_loads(test_tokenizer: MultiFieldTokenizer):
    assert isinstance(test_tokenizer, MultiFieldTokenizer)


def test_tokenizer_loaded_subtokenizers(test_tokenizer: MultiFieldTokenizer):
    assert len(test_tokenizer.tokenizers) == 5
    for field in test_tokenizer.tokenizers.keys():
        assert isinstance(
            test_tokenizer.get_field_tokenizer(field), PreTrainedTokenizerBase
        )


# def test_create_save_and_load_via_panglao_data_module(
#     pl_data_module_panglao_dynamic_binning,
# ):
#     tokenizer = pl_data_module_panglao_dynamic_binning.tokenizer
#     with tempfile.TemporaryDirectory() as tmpdir:
#         SCBertMainConfig.save_tokenizer(tokenizer, tmpdir)
#         loaded_tokenizer = load_tokenizer(tmpdir)
#     assert isinstance(tokenizer, MultiFieldTokenizer)
#     assert isinstance(loaded_tokenizer, MultiFieldTokenizer)
#     assert_tokenizers_match(tokenizer, loaded_tokenizer)


def test_create_save_and_load_scbert_save():
    tokenizer = load_tokenizer("gene2vec")
    with tempfile.TemporaryDirectory() as tmpdir:
        SCBertMainConfig.save_tokenizer(tokenizer, tmpdir)
        loaded_tokenizer = load_tokenizer(tmpdir)
    assert isinstance(tokenizer, MultiFieldTokenizer)
    assert isinstance(loaded_tokenizer, MultiFieldTokenizer)
    assert_tokenizers_match(tokenizer, loaded_tokenizer)


def test_create_save_and_load():
    tokenizer = load_tokenizer("gene2vec")
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer.save_pretrained(tmpdir)
        loaded_tokenizer = MultiFieldTokenizer.from_pretrained(tmpdir)
    assert isinstance(tokenizer, MultiFieldTokenizer)
    assert isinstance(loaded_tokenizer, MultiFieldTokenizer)
    assert_tokenizers_match(tokenizer, loaded_tokenizer)


@pytest.mark.skipif(
    reason="no access to saved checkpoint", condition=not CCC_TOKENIZER_ACCESSIBLE
)
def test_load_from_checkpoint_on_ccc():
    tokenizer = MultiFieldTokenizer.from_pretrained(
        MODEL_SNAPSHOT_FOR_LOADING_SAVED_TOKENIZER
    )
    tokenizer = load_tokenizer(MODEL_SNAPSHOT_FOR_LOADING_SAVED_TOKENIZER)
    assert isinstance(tokenizer, MultiFieldTokenizer)
    assert sorted(tokenizer.tokenizers.keys()) == ["expressions", "genes"]


@pytest.mark.skipif(
    reason="no access to saved checkpoint", condition=not CCC_TOKENIZER_ACCESSIBLE
)
def test_load_from_checkpoint_on_ccc_with_load_tokenizer():
    tokenizer = load_tokenizer(MODEL_SNAPSHOT_FOR_LOADING_SAVED_TOKENIZER)
    assert isinstance(tokenizer, MultiFieldTokenizer)
    assert sorted(tokenizer.tokenizers.keys()) == ["expressions", "genes"]


@pytest.mark.skipif(
    reason="no access to saved checkpoint", condition=not CCC_TOKENIZER_ACCESSIBLE
)
def test_load_from_checkpoint_on_ccc_autotokenizer_with_config():
    # notice - we are reading a model config from the tokenizer config.
    config = AutoConfig.from_pretrained(
        str(
            MODEL_SNAPSHOT_FOR_LOADING_SAVED_TOKENIZER
            / "multifield-tokenizer_config.json"
        )
    )
    tokenizer = MultiFieldTokenizer.from_old_multifield_tokenizer(
        MODEL_SNAPSHOT_FOR_LOADING_SAVED_TOKENIZER,
        config=config,
        # filename_prefix="multifield",
    )
    assert isinstance(tokenizer, MultiFieldTokenizer)
    assert sorted(tokenizer.tokenizers.keys()) == ["expressions", "genes"]


@pytest.mark.xfail(
    raises=ValueError, reason="hugging face support for filename_prefix is unclear"
)
def test_create_save_and_load_inside_prefix():
    tokenizer = load_tokenizer("gene2vec")
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer.save_pretrained(tmpdir, filename_prefix="multifield")
        loaded_tokenizer = MultiFieldTokenizer.from_pretrained(
            tmpdir, filename_prefix="multifield"
        )
    assert isinstance(tokenizer, MultiFieldTokenizer)
    assert isinstance(loaded_tokenizer, MultiFieldTokenizer)
    assert_tokenizers_match(tokenizer, loaded_tokenizer)


def test_convert_old_tokenizer_and_load():
    vocab_path = Path(__file__).parent / "test_vocab"
    multi_field_tokenizer_from_multivocab: MultiFieldTokenizer = (
        MultiFieldTokenizer.from_old_multifield_tokenizer(
            str(vocab_path),
            multifield_vocab_file=str(vocab_path / "multifield_vocab.json"),
            load_relative=True,
        )
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        multi_field_tokenizer_from_multivocab.save_pretrained(save_directory=tmpdir)
        final = MultiFieldTokenizer.from_pretrained(tmpdir)
    assert isinstance(final, MultiFieldTokenizer)
    assert isinstance(multi_field_tokenizer_from_multivocab, MultiFieldTokenizer)
    assert_tokenizers_match(final, multi_field_tokenizer_from_multivocab)


def test_convert_tokens_to_ids(test_tokenizer: MultiFieldTokenizer):
    local_ids = test_tokenizer.convert_field_tokens_to_ids(
        "genes", ["token1", "token2"]
    )

    assert local_ids == [5, 6]


def test_pad_token_id(test_tokenizer: MultiFieldTokenizer):
    pad_token = test_tokenizer.pad_token
    assert pad_token == "[PAD]"
    pad_token_id = test_tokenizer.pad_token_id
    assert pad_token_id == 2


def test_tokenization_single_instance_batch(test_tokenizer_fields):
    tokenizer = load_test_tokenizer()

    m1 = MultiFieldInstance(
        metadata={"cell_name": "cell1"},
        data={"genes": ["token1", "token2"], "expressions": [2, 3]},
    )

    batch = m1
    encoding = tokenizer(
        mfi=batch,
        fields=test_tokenizer_fields,
        # return_id="local",
        add_special_tokens=False,
        is_split_into_words=True,
        return_attention_mask=True,
        return_special_tokens_mask=True,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=10,
    )

    gene_ids = tensor([5, 6, 2, 2, 2, 2, 2, 2, 2, 2])
    expressions_ids = tensor([7, 8, 2, 2, 2, 2, 2, 2, 2, 2])
    attention_mask = tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    special_tokens_mask_genes = tensor([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    special_tokens_mask_expressions = tensor([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

    assert_tensors_equal(encoding["genes"]["input_ids"], gene_ids), encoding["genes"][
        "input_ids"
    ]
    assert_tensors_equal(encoding["expressions"]["input_ids"], expressions_ids)
    assert_tensors_equal(encoding["genes"]["attention_mask"], attention_mask)
    assert_tensors_equal(
        encoding["genes"]["special_tokens_mask"], special_tokens_mask_genes
    )
    assert_tensors_equal(encoding["expressions"]["attention_mask"], attention_mask)
    assert_tensors_equal(
        encoding["expressions"]["special_tokens_mask"], special_tokens_mask_expressions
    )


def test_tokenization_single_instance_add_special_tokens(test_tokenizer_fields):
    tokenizer = load_test_tokenizer()

    m1 = MultiFieldInstance(
        metadata={"cell_name": "cell1"},
        data={"genes": ["token1", "token2"], "expressions": [2, 3]},
    )

    batch = m1
    encoding = tokenizer(
        mfi=batch,
        fields=test_tokenizer_fields,
        # return_id="local",
        add_special_tokens=False,
        is_split_into_words=True,
        return_attention_mask=True,
        return_special_tokens_mask=True,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=10,
    )

    gene_ids = tensor([5, 6, 2, 2, 2, 2, 2, 2, 2, 2])
    expressions_ids = tensor([7, 8, 2, 2, 2, 2, 2, 2, 2, 2])
    attention_mask = tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    special_tokens_mask_genes = tensor([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    special_tokens_mask_expressions = tensor([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

    assert_tensors_equal(encoding["genes"]["input_ids"], gene_ids), encoding["genes"][
        "input_ids"
    ]
    assert_tensors_equal(encoding["expressions"]["input_ids"], expressions_ids)
    assert_tensors_equal(encoding["genes"]["attention_mask"], attention_mask)
    assert_tensors_equal(
        encoding["genes"]["special_tokens_mask"], special_tokens_mask_genes
    )
    assert_tensors_equal(encoding["expressions"]["attention_mask"], attention_mask)
    assert_tensors_equal(
        encoding["expressions"]["special_tokens_mask"], special_tokens_mask_expressions
    )


def test_tokenization_does_not_resplit(test_tokenizer_fields):
    tokenizer = load_test_tokenizer()

    # token1 should be translated to 5, GENE_NAME_XYZ is 16, all the others should be a single zero (UKN) between two fives
    m1 = MultiFieldInstance(
        metadata={"cell_name": "cell1"},
        data={
            "genes": [
                "token1",
                "tobesplit99",
                "token1",
                "token0.token 3",
                "token1",
                "GENE_NAME_XYZ",
                "token1",
                "token2-token3",
                "token1",
                "token4_token5",
                "token1",
                "token6:token7",
                "token1",
                "token8@token9",
                "token1",
            ],
            "expressions": [
                "2",
                "3",
                "3",
                "3",
                "3",
                "3",
                "3",
                "3",
                "3",
                "3",
                "3",
                "3",
                "3",
                "3",
                "3",
            ],
        },
    )

    batch = m1
    encoding = tokenizer(
        mfi=batch,
        fields=test_tokenizer_fields,
        # return_id="local",
        add_special_tokens=True,
        is_split_into_words=True,
        return_attention_mask=True,
        return_special_tokens_mask=True,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=18,
    )

    gene_ids = tensor([3, 5, 0, 5, 0, 5, 16, 5, 0, 5, 0, 5, 0, 5, 0, 5, 1, 2])
    expressions_ids = tensor([3, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 2])
    attention_mask = tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    special_tokens_mask_genes = tensor(
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    )
    special_tokens_mask_expressions = tensor(
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    )

    assert_tensors_equal(encoding["genes"]["input_ids"], gene_ids)
    assert_tensors_equal(encoding["expressions"]["input_ids"], expressions_ids)
    assert_tensors_equal(encoding["genes"]["attention_mask"], attention_mask)
    assert_tensors_equal(
        encoding["genes"]["special_tokens_mask"], special_tokens_mask_genes
    )
    assert_tensors_equal(encoding["expressions"]["attention_mask"], attention_mask)
    assert_tensors_equal(
        encoding["expressions"]["special_tokens_mask"], special_tokens_mask_expressions
    )


def assert_tensors_equal(result, expected, name=""):
    ref = pd.DataFrame(expected.detach().numpy().squeeze())
    res = pd.DataFrame(result.detach().numpy().squeeze())
    pdt.assert_frame_equal(ref, res, obj=name, by_blocks=True)


@pytest.mark.xfail(
    reason="__call__ function in PreTrainedTokenizerFast dose not allow to turn off splitting words into tokens"
)
def test_tokenize_xyz(test_tokenizer: MultiFieldTokenizer):
    vec = [
        # ["token2", "token2", "token4", "GENE_NAME_XYZ", "token2", "token2", "token4"],
        ["token2", "token2", "token4", "GENE.NAME.XYZ", "token2", "token2", "token4"],
    ]
    # "GENE_NAME_XYZ" is a token in the list
    vocab = test_tokenizer.get_field_vocab("genes")
    encoding = test_tokenizer.get_field_tokenizer("genes")(
        vec,
        return_attention_mask=False,
        return_special_tokens_mask=False,
        is_split_into_words=True,
        add_special_tokens=False,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=16,
    )
    # this will fail as GENE_NAME_XYZ will be split into five unknown tokens
    assert (encoding["input_ids"] != test_tokenizer.unk_token_id).all()


def test_special_token_mask_with_special_tokens(
    test_tokenizer: MultiFieldTokenizer, test_tokenizer_fields
):
    data1 = [
        MultiFieldInstance(
            data={
                "genes": ["token1", "token2", "token1", "token2"],
                "expressions": ["1", "2", "2", "2"],
            }
        ),
        MultiFieldInstance(data={"genes": ["token1"], "expressions": ["3"]}),
    ]
    data2 = [
        MultiFieldInstance(data={"genes": ["token1"], "expressions": ["4"]}),
        MultiFieldInstance(
            data={"genes": ["token1", "token2"], "expressions": ["5", "6"]}
        ),
    ]

    encoding = test_tokenizer(
        mfi=data1,
        mfi_pair=data2,
        fields=test_tokenizer_fields,
        return_attention_mask=True,
        return_special_tokens_mask=True,
        padding="max_length",
        return_tensors="pt",
        max_length=8,
    )

    enc = encoding["expressions"]
    input_ids = enc["input_ids"][0].tolist()
    special_tokens_mask = enc["special_tokens_mask"][0].tolist()

    mask_w_specials = test_tokenizer.get_field_tokenizer(
        "expressions"
    ).get_special_tokens_mask(input_ids, already_has_special_tokens=True)
    assert special_tokens_mask == mask_w_specials

    # only valid values for other parts of encoding
    assert torch.unique(enc["attention_mask"], sorted=True).tolist() == [0, 1]
    assert (
        enc["input_ids"][enc["attention_mask"] == 0] == test_tokenizer.pad_token_id
    ).all()
    assert torch.unique(enc["special_tokens_mask"], sorted=True).tolist() == [0, 1]

    assert torch.unique(enc["token_type_ids"], sorted=True).tolist() == [0, 1]


def test_special_token_ids(test_tokenizer: MultiFieldTokenizer):
    x = test_tokenizer.all_special_ids
    assert x == [0, 1, 2, 3, 4]


def test_truncation(test_tokenizer_fields):
    n_samples = 5
    max_length = 3
    tokenizer = load_test_tokenizer()
    dataset = generate_dataset(n_samples, 4, 8, seed=42, tokenizer=tokenizer)
    encoding = tokenizer(
        dataset[:],
        fields=test_tokenizer_fields,
        # return_id="local",
        return_attention_mask=True,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=max_length,
    )
    assert encoding["genes"]["input_ids"].shape == (n_samples, max_length)


def test_padding_on_samples_of_different_length(gene2vec_fields):
    n_samples = 50
    min_seq_len = 4
    max_seq_len = 10
    tokenizer = load_test_tokenizer()
    dataset = generate_dataset(
        n_samples, min_seq_len, max_seq_len, seed=42, tokenizer=tokenizer
    )

    encoding = tokenizer(
        dataset[:],
        fields=gene2vec_fields,
        # return_id="local",
        return_attention_mask=True,
        padding="longest",
        return_tensors="pt",
    )
    # lengths are correct
    enc = encoding["genes"]
    assert enc["input_ids"].shape == (n_samples, max_seq_len + 2)
    # at least some are padded
    assert tokenizer.pad_token_id in enc["input_ids"][:, -1]
    # only valid values for other parts of encoding
    assert torch.unique(enc["attention_mask"], sorted=True).tolist() == [0, 1]
    assert (
        enc["input_ids"][enc["attention_mask"] == 0] == tokenizer.pad_token_id
    ).all()
    assert torch.unique(enc["token_type_ids"], sorted=True).tolist() == [0]
    assert torch.unique(enc["special_tokens_mask"], sorted=True).tolist() == [0, 1]


def test_get_special_token_binary_mask(test_tokenizer: MultiFieldTokenizer):
    field = "expressions"
    special_token_count = len(test_tokenizer.all_special_tokens)
    special_tokens = test_tokenizer.all_special_ids
    vocab_size = test_tokenizer.field_vocab_size(field)
    mask = tensor(test_tokenizer.get_special_token_binary_mask(field))
    assert mask.shape == (vocab_size,)
    assert mask[special_tokens].all() == True
    assert mask[special_token_count:].all() == False


def test_adding_tokens():
    tokenizer = load_test_tokenizer()
    field = "expressions"
    token_to_add = "addedToken1"
    data = ["3", "4"]
    res = local_ids = tokenizer.convert_field_tokens_to_ids(field, data)
    assert res == [8, 9]
    data.append(token_to_add)
    res = local_ids = tokenizer.convert_field_tokens_to_ids(field, data)
    assert res == [8, 9, tokenizer.unk_token_id]
    voc_size = tokenizer.get_field_tokenizer(field).vocab_size
    # actually add the tokens
    tokenizer.add_tokens_for_field(field, token_to_add)
    res = local_ids = tokenizer.convert_field_tokens_to_ids(field, data)
    assert res == [8, 9, voc_size]
    # reset back to original state.
    tokenizer.reset_tokenizer_vocab(field)


def test_reset_tokenizer():
    tokenizer = load_test_tokenizer()
    field = "expressions"
    token_to_add = "addedToken1"
    data = ["3", "4"]
    res = local_ids = tokenizer.convert_field_tokens_to_ids(field, data)
    assert res == [8, 9]
    data.append(token_to_add)
    res = local_ids = tokenizer.convert_field_tokens_to_ids(field, data)
    assert res == [8, 9, tokenizer.unk_token_id]
    voc_size = tokenizer.get_field_tokenizer(field).vocab_size
    tokenizer.get_field_tokenizer(field).add_tokens(token_to_add)
    res = local_ids = tokenizer.convert_field_tokens_to_ids(field, data)
    assert res == [8, 9, voc_size]
    tokenizer.reset_tokenizer_vocab(field)
    res = local_ids = tokenizer.convert_field_tokens_to_ids(field, data)
    assert res == [8, 9, tokenizer.unk_token_id]


def test_save_and_load_tokenizer(test_tokenizer: MultiFieldTokenizer):
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "saved_tokenizer"
        test_tokenizer.save_pretrained(out_path)
        loaded_tokenizer = MultiFieldTokenizer.from_pretrained(out_path)
        assert_tokenizers_match(test_tokenizer, loaded_tokenizer)


def test_save_and_load_tokenizer_from_different_dir(
    test_tokenizer: MultiFieldTokenizer,
):
    """The paths written inside the tokenizer config file must be read correctly even if the cwd is not as it was when they had been saved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "saved_tokenizer"
        test_tokenizer.save_pretrained(out_path)
        old_path = os.getcwd()
        os.chdir("/")
        loaded_tokenizer = MultiFieldTokenizer.from_pretrained(out_path)
        os.chdir(old_path)
        assert_tokenizers_match(test_tokenizer, loaded_tokenizer)


def test_init_tokenizer_and_save():
    """Test that we can create a new multifiled tokenizer and not just load it, in a way that will allow us to save it back as pretrained and then load it as such."""
    mf_paths_dict = {
        "genes": TEST_TOKENIZER_ROOT / "genes",
        "label_expressions": TEST_TOKENIZER_ROOT / "label_expressions",
        "perturbations": TEST_TOKENIZER_ROOT / "perturbations",
    }
    new_tokenizer = MultiFieldTokenizer(
        field_to_tokenizer_map=mf_paths_dict, load_relative=False
    )
    assert isinstance(new_tokenizer, MultiFieldTokenizer)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "saved_new_tokenizer"
        new_tokenizer.save_pretrained(out_path)
        assert sorted(new_tokenizer.tokenizers.keys()) == sorted(mf_paths_dict.keys())
        loaded_tokenizer = MultiFieldTokenizer.from_pretrained(out_path)
        assert_tokenizers_match(new_tokenizer, loaded_tokenizer)


def assert_tokenizers_match(old_tokenizer, new_tokenizer):
    assert type(new_tokenizer) == type(old_tokenizer)
    assert isinstance(new_tokenizer, MultiFieldTokenizer)
    assert set(new_tokenizer.tokenizers.keys()) == set(old_tokenizer.tokenizers.keys())
    for field_name in new_tokenizer.tokenizers.keys():
        assert (
            new_tokenizer.get_field_tokenizer(field_name).get_vocab()
            == old_tokenizer.get_field_tokenizer(field_name).get_vocab()
        )


def test_load_all_tokenizer():
    all_tokenizer = load_tokenizer("all_genes")
    assert isinstance(all_tokenizer, MultiFieldTokenizer)
    assert len(all_tokenizer.tokenizers) == 4
    assert all(
        isinstance(
            all_tokenizer.get_field_tokenizer(field_name),
            MultiFieldTokenizer.SUB_TOKENIZER_CLASS,
        )
        for field_name in all_tokenizer.tokenizers.keys()
    )


def test_convert_vocab_to_tokenizer():
    old_tokenizer_path = OLD_TOKENIZER_PATH
    tokenizer = MultiFieldTokenizer.from_old_multifield_tokenizer(
        name_or_path=old_tokenizer_path, save_converted_tokenizer_back=False
    )
    assert sorted(tokenizer.tokenizers.keys()) == [
        "expressions",
        "genes",
        "label_expressions",
        "perturbations",
    ]

    for tok in tokenizer.tokenizers.values():
        assert isinstance(tok, MultiFieldTokenizer.SUB_TOKENIZER_CLASS)


def convert_subtokenizer(old_tokenizer_path, save_converted_tokenizer_back=False):
    tokenizer = MultiFieldTokenizer.convert_tokenizer_to_bert(
        name_or_path=old_tokenizer_path,
        save_converted_tokenizer_back=save_converted_tokenizer_back,
    )
    return tokenizer


def test_convert_all_pretrained_sub_tokenizers():
    all_subtokenizers = [
        "all_genes",
        "expressions",
        "gene2vec",
        "label_expressions",
        "perturbations",
        "snp2vec_tokenizer",
    ]
    for name in all_subtokenizers:
        path = TOKENIZER_ROOT / "pretrained" / name
        # print(name, path)
        tokenizer = convert_subtokenizer(path)
        assert isinstance(tokenizer, MultiFieldTokenizer.SUB_TOKENIZER_CLASS)


def test_convert_all_resource_sub_tokenizers():
    all_subtokenizers = {
        "genes": f"{TEST_TOKENIZER_ROOT}/genes",
        "expressions": f"{TEST_TOKENIZER_ROOT}/expressions",
        "perturbations": f"{TEST_TOKENIZER_ROOT}/perturbations",
    }

    for name, path in all_subtokenizers.items():
        tokenizer = convert_subtokenizer(path, save_converted_tokenizer_back=False)
        assert isinstance(tokenizer, MultiFieldTokenizer.SUB_TOKENIZER_CLASS)


def test_multifield_tokenizer_can_work_with_string_inputs(
    test_tokenizer, snp2vec_fields
):
    dna_sequence = "AGGCTGTGGCCACTACACCCACAATCTTCTGGGGGCCGGGTTTCTCCTACACCATAGAGACGGGTCCGGAAACGGGACAGAAGGCCCACCTTCCTCCCTCCGACGCCACCAATGAGGCCAACTAACCAGGAACCGAGGTAGAGAGGCCGCACAGCTGAGTCTCAGGCCGGTGCCATCTTAAGTGTGGGCGCCGCGACGAT"
    mfi = MultiFieldInstance(data={"dna_chunks": dna_sequence})
    output_from_multi_field_tokenizer = test_tokenizer(
        [mfi],
        fields=snp2vec_fields,
        add_special_tokens=True,
        is_split_into_words=False,
        return_attention_mask=True,
        return_special_tokens_mask=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=100,
    )["dna_chunks"]

    from transformers import BertTokenizerFast

    bert_tokenizer = BertTokenizerFast.from_pretrained(
        f"{TEST_TOKENIZER_ROOT}/dna_chunks",
        do_lower_case=False,
        tokenize_chinese_chars=False,
        clean_text=False,
        strip_accents=None,
    )
    output_from_bert_tokenizer = bert_tokenizer(
        dna_sequence,
        return_tensors="pt",
        add_special_tokens=True,
        is_split_into_words=False,
        return_attention_mask=True,
        return_special_tokens_mask=True,
        padding=True,
        truncation=True,
        max_length=100,
    )
    for key in output_from_bert_tokenizer.keys():
        assert torch.equal(
            output_from_bert_tokenizer[key], output_from_multi_field_tokenizer[key]
        )


def test_snp_tokenizer(test_tokenizer, snp2vec_fields):
    dna_sequence = "ACGNT兰兰裔均兰秋TTGCAGTGAGCC阳GATGCCTGTAATCCCAGCTC高正GA则GC"
    mfi = MultiFieldInstance(data={"dna_chunks": dna_sequence})
    output_from_multi_field_tokenizer = test_tokenizer(
        [mfi],
        fields=snp2vec_fields,
        add_special_tokens=True,
        is_split_into_words=False,
        return_attention_mask=True,
        return_special_tokens_mask=True,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=100,
    )["dna_chunks"]
    expected_ids = tensor(
        [3, 156, 8, 9, 243, 441, 446, 814, 998, 1207, 1514, 1644, 1652, 1, 2]
    )
    expected_tokens = [
        "[CLS]",
        "ACG",
        "N",
        "T",
        "兰兰",
        "裔均",
        "兰秋",
        "TTGCAGTGAGCC",
        "阳G",
        "ATGCCTGTAATCCCAGC",
        "TC高",
        "正G",
        "A则GC",
        "[SEP]",
        "[PAD]",
    ]
    print(output_from_multi_field_tokenizer)
    assert torch.equal(
        output_from_multi_field_tokenizer["input_ids"][0, :15], expected_ids
    )
    assert (
        test_tokenizer.convert_field_ids_to_tokens(
            expected_ids.tolist(), field="dna_chunks"
        )
        == expected_tokens
    )


def test_create_tokenizer_with_new_specials():
    mft = load_tokenizer("all_genes")
    pre_vocab_size = mft.field_vocab_size("genes")

    with tempfile.TemporaryDirectory() as tmpdir:
        create_new_mft_with_extra_cls_tokens(tmpdir, "all_genes")
        mft = load_tokenizer(tmpdir)
        post_vocab_size = mft.field_vocab_size("genes")

    assert post_vocab_size > pre_vocab_size

    subtok = mft.get_field_tokenizer("genes")
    # subtok.add_special_tokens({"additional_special_tokens": special_cls_tokens})

    tokenized = subtok("BRCA1", return_special_tokens_mask=True)
    input_ids = tokenized["input_ids"]
    assert subtok.convert_ids_to_tokens(input_ids[0]) == "[CLS]"
    assert subtok.convert_ids_to_tokens(input_ids[-1]) == "[SEP]"
    assert tokenized["special_tokens_mask"] == [1, 0, 1]

    added_cls = ["[CLS]", "[CLS_0]"]
    set_custom_cls_post_processor(subtok, added_cls)
    tokenized = subtok("BRCA1", return_special_tokens_mask=True)
    input_ids = tokenized["input_ids"]
    assert subtok.convert_ids_to_tokens(input_ids[0]) == "[CLS]"
    assert subtok.convert_ids_to_tokens(input_ids[1]) == "[CLS_0]"
    assert subtok.convert_ids_to_tokens(input_ids[-1]) == "[SEP]"
    assert tokenized["special_tokens_mask"] == [1, 1, 0, 1]


def test_load_tokenizer_with_prepend_tokens():
    mft = load_tokenizer("all_genes_v2", ["[CLS]", "[CLS_0]"])
    subtok = mft.get_field_tokenizer("genes")
    tokenized = subtok("BRCA1", return_special_tokens_mask=True)
    input_ids = tokenized["input_ids"]
    assert subtok.convert_ids_to_tokens(input_ids[0]) == "[CLS]"
    assert subtok.convert_ids_to_tokens(input_ids[1]) == "[CLS_0]"
    assert subtok.convert_ids_to_tokens(input_ids[-1]) == "[SEP]"
    assert tokenized["special_tokens_mask"] == [1, 1, 0, 1]
