import json
from pathlib import Path

import pytest

from bmfm_targets.tokenization.create.create_protein_coding_tokenizer import (
    create_protein_coding_tokenizer,
)
from bmfm_targets.tokenization.load import get_all_genes_v2_tokenizer
from bmfm_targets.tokenization.multifield_tokenizer import MultiFieldTokenizer
from bmfm_targets.tokenization.resources.reference_data import get_protein_coding_genes

_PC_VOCAB_DIR = (
    Path(__file__).parent.parent
    / "tokenization"
    / "protein_coding_vocab"
    / "tokenizers"
    / "genes"
)


@pytest.fixture(scope="module")
def pc_tokenizer(tmp_path_factory):
    out = tmp_path_factory.mktemp("pc_tok")
    create_protein_coding_tokenizer(str(out))
    return MultiFieldTokenizer.from_pretrained(str(out)), out


def test_vocab_content(pc_tokenizer):
    """Verify vocab contains correct symbols and specials."""
    mft, _ = pc_tokenizer
    genes_tok = mft.get_field_tokenizer("genes")
    vocab = set(genes_tok.get_vocab().keys())

    # Must use correct symbols (e.g. A1BG)
    assert set(get_protein_coding_genes()).issubset(vocab)
    # Specials must be present
    assert "[S]" in vocab
    assert "[T]" in vocab


def test_config_integrity(pc_tokenizer):
    """Verify PRE-TOKENIZER and formatting match reference exactly."""
    mft, path = pc_tokenizer
    ref = get_all_genes_v2_tokenizer()

    new_backend = mft.get_field_tokenizer("genes").backend_tokenizer
    ref_backend = ref.get_field_tokenizer("genes").backend_tokenizer

    # 1. Critical: pre_tokenizer must be preserved (null)
    assert str(new_backend.pre_tokenizer) == str(ref_backend.pre_tokenizer)

    # 2. Check JSON formatting style
    map_file = path / "tokenizers" / "genes" / "special_tokens_map.json"
    data = json.loads(map_file.read_text())

    # Entries must be objects
    assert isinstance(data["cls_token"], dict)
    assert isinstance(data["additional_special_tokens"][0], dict)
    assert data["additional_special_tokens"][0]["normalized"] is False


def test_genes_special_tokens_loaded_without_decoder_blob(pc_tokenizer):
    """
    The genes subtokenizer must expose its full special-token set even though
    tokenizer_config.json carries no added_tokens_decoder blob.

    The loader reads the tokens straight from tokenizer.json and re-registers the
    [CLS_*]/[S]/[T] specials from special_tokens_map.json, so all_special_tokens must
    match the reference all_genes_v2 tokenizer's genes field.
    """
    mft, path = pc_tokenizer

    cfg = json.loads(
        (path / "tokenizers" / "genes" / "tokenizer_config.json").read_text()
    )
    assert "added_tokens_decoder" not in cfg

    genes_specials = set(mft.get_field_tokenizer("genes").all_special_tokens)
    ref_specials = set(
        get_all_genes_v2_tokenizer().get_field_tokenizer("genes").all_special_tokens
    )
    assert genes_specials == ref_specials
