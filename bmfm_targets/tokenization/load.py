import logging
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from bmfm_targets.tokenization import MultiFieldTokenizer

GENE2VEC_VOCAB_PATH = Path(__file__).parent / "gene2vec_vocab"
ALL_GENES_VOCAB_PATH = Path(__file__).parent / "all_genes_vocab"
ALL_GENES_V2_VOCAB_PATH = Path(__file__).parent / "all_genes_v2_vocab"
SNP_VOCAB_PATH = Path(__file__).parent / "snp_vocab"
REF_VOCAB_PATH = Path(__file__).parent / "ref_vocab"
SNP_BPE_VOCAB_PATH = SNP_VOCAB_PATH / "tokenizers/dna_chunks"
REF_BPE_VOCAB_PATH = REF_VOCAB_PATH / "tokenizers/dna_chunks"

logger = logging.getLogger(__name__)


def get_gene2vec_tokenizer():
    """Load the gene2vec tokenizer."""
    return MultiFieldTokenizer.from_pretrained(GENE2VEC_VOCAB_PATH)


def get_all_genes_tokenizer():
    """Load the all_genes tokenizer."""
    return MultiFieldTokenizer.from_pretrained(ALL_GENES_VOCAB_PATH)


def get_all_genes_v2_tokenizer():
    """Load the all_genes tokenizer."""
    return MultiFieldTokenizer.from_pretrained(ALL_GENES_V2_VOCAB_PATH)


def get_snp2vec_tokenizer() -> MultiFieldTokenizer:
    return MultiFieldTokenizer.from_pretrained(SNP_VOCAB_PATH)


def get_ref2vec_tokenizer() -> MultiFieldTokenizer:
    return MultiFieldTokenizer.from_pretrained(REF_VOCAB_PATH)


def get_snp2vec_BPEtokenizer() -> PreTrainedTokenizerFast:
    snp2vec_tokenizer = AutoTokenizer.from_pretrained(
        SNP_BPE_VOCAB_PATH,
        do_lower_case=False,
        tokenize_chinese_chars=False,
        strip_accents=False,
        clean_text=False,
    )
    return snp2vec_tokenizer


def get_refgen2vec_BPEtokenizer() -> PreTrainedTokenizerFast:
    ref2vec_tokenizer = AutoTokenizer.from_pretrained(
        REF_BPE_VOCAB_PATH,
        do_lower_case=False,
        tokenize_chinese_chars=False,
        strip_accents=False,
        clean_text=False,
    )
    return ref2vec_tokenizer


def load_tokenizer(identifier="all_genes", prepend_tokens=None) -> MultiFieldTokenizer:
    """
    Load a tokenizer from a pretrained model or a directory containing a tokenizer.

    Args:
    ----
        identifier (str, optional): name of packaged tokenizer ("gene2vec" or "all_genes"
            or path to pretrained model or tokenizer directory. Defaults to "gene2vec".
    """
    tokenizer = _load_tokenizer_from_identifier(identifier)
    if prepend_tokens is not None:
        from .multifield_tokenizer import set_custom_cls_post_processor

        for subtok in tokenizer.tokenizers.values():
            set_custom_cls_post_processor(subtok, prepend_tokens)
    return tokenizer


def _load_tokenizer_from_identifier(identifier) -> MultiFieldTokenizer:
    if identifier == "gene2vec":
        return get_gene2vec_tokenizer()
    if identifier == "all_genes":
        return get_all_genes_tokenizer()
    if identifier == "all_genes_v2":
        return get_all_genes_v2_tokenizer()
    if identifier == "snp2vec":
        return get_snp2vec_tokenizer()
    if identifier == "ref2vec":
        return get_ref2vec_tokenizer()
    return _load_tokenizer_from_path_and_maybe_convert(identifier)


def _load_tokenizer_from_path_and_maybe_convert(tokenizer_path):
    try:
        return MultiFieldTokenizer.from_pretrained(tokenizer_path)
    except FileNotFoundError:
        logger.info(f"Multifield tokenizer not found in {tokenizer_path}")
        logger.info("Attempting to load from multifield_vocab.json and convert")
        converted_tokenizer = MultiFieldTokenizer.from_old_multifield_tokenizer(
            tokenizer_path, save_converted_tokenizer_back=True
        )
        logger.info(f"Converted Multifield tokenizer saved to {tokenizer_path}")
        return converted_tokenizer
