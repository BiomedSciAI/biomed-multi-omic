import tempfile

import pytest

from ..tokenization import MultiFieldVocabulary


def make_dummy_multifield_vocab():
    vocab = MultiFieldVocabulary()

    # Add special tokens for each field
    field1_key = "field1"
    field2_key = "field2"

    # Add random tokens for field1
    random_tokens_field1 = ["token1", "token2", "token3"]
    for token in random_tokens_field1:
        vocab.add_token_to_vocab(field1_key, token)

    # Add random tokens for field2
    random_tokens_field2 = ["token4", "token5", "token6"]
    for token in random_tokens_field2:
        vocab.add_token_to_vocab(field2_key, token)
    return vocab


def test_save_load():
    with tempfile.TemporaryDirectory() as tmpdirname:
        vocab_before = make_dummy_multifield_vocab()
        vocab_before.save(tmpdirname + "/vocab")
        vocab_after = MultiFieldVocabulary.load(tmpdirname + "/vocab")
        assert vocab_before.token_to_ids == vocab_after.token_to_ids


def test_get_unk_token_for_unknown_id():
    vocab = make_dummy_multifield_vocab()
    token = vocab.get_local_token("field1", 1)
    assert token == "token2"
    token = vocab.get_local_token("field1", 111)
    assert token == "[UNK]"


def test_get_field_specific_ids():
    vocab = make_dummy_multifield_vocab()
    local_ids = vocab.get_field_specific_token_ids("field2", return_id="local")
    assert local_ids == [0, 1, 2]
    global_ids = vocab.get_field_specific_token_ids("field2", return_id="global")
    assert global_ids == [3, 4, 5]


def test_save_load_nonalphabetic_insert_order():
    pytest.skip("Unknown if this is required")
    with tempfile.TemporaryDirectory() as tmpdirname:
        vocab_before = make_dummy_multifield_vocab()
        vocab_before.add_tokens_to_vocab("field0", ["token7", "token8"])
        vocab_before.save(tmpdirname + "/vocab")
        vocab_after = MultiFieldVocabulary.load(tmpdirname + "/vocab")
        assert vocab_before.token_to_ids == vocab_after.token_to_ids
