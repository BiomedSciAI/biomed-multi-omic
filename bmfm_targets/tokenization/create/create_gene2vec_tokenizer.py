"""
Functions for creating gene vocabs from various resources:
- gene2vec file
- cellxgene_soma database
- stored h5ad files.
"""

import json
import logging
from functools import reduce
from os import environ
from pathlib import Path

import pandas as pd
from cellxgene_census import open_soma
from scanpy import read_h5ad

from bmfm_targets.tests import helpers
from bmfm_targets.tokenization.load import get_gene2vec_tokenizer

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)


def make_cellxgene_vocab(vocab_dir, uri=None, expression_bins=50):
    gene_list = get_cellxgene_soma_gene_list(uri)
    make_gene_vocab(vocab_dir, gene_list, expression_bins)


def get_cellxgene_soma_gene_list(uri=None):
    census = open_soma(uri=uri)
    experiment = census["census_data"]["homo_sapiens"]
    gene_data = experiment.ms["RNA"].var.read().concat().to_pandas()
    gene_list = sorted(gene_data.feature_name)
    return gene_list


def get_vocab_from_dataset(h5ad_path):
    ad = read_h5ad(h5ad_path, backed="r")
    return {*ad.var_names}


def get_test_data_vocabs():
    """Load vocabs from snippets of data stored in tests and defined in `helpers`."""
    paths = [getattr(helpers, i) for i in helpers.__dict__ if "Paths" in i]
    roots = [
        *(getattr(x, next(filter(lambda y: "root" in y, x.__dict__))) for x in paths)
    ]
    h5ad_paths = []
    for dsroot in roots:
        try:
            h5ad_paths.append(next((dsroot / "h5ad").glob("*")))
        except:
            print("no h5ad in dir ", dsroot)
            continue
    vocabs = {p.parent: get_vocab_from_dataset(p) for p in h5ad_paths}
    vocabs = {k: v for k, v in vocabs.items() if len(v) > 1}
    return vocabs


def get_data_vocabs_from_environment():
    """
    Pull all gene vocabulary from datasets that are stored in environment variables.

    This assumes that the environment variables are correctly set and that the
    datasets have already been processed.
    """
    data_path_vars = [
        "BMFM_TARGETS_PANGLAO_DATA",
        "BMFM_TARGETS_ZHENG68K_DATA",
        "BMFM_TARGETS_HUMANCELLATLAS_DATA",
        "BMFM_TARGETS_CELLXGENE_DATA",
        "BMFM_TARGETS_SCIBD_DATA",
        "BMFM_TARGETS_SCIBD300K_DATA",
        "BMFM_TARGETS_TIL_DATA",
    ]
    vocabs = {}
    for path_var in data_path_vars:
        name = path_var.split("_")[2].lower()
        root_path = environ.get(path_var)
        if root_path is None:
            logger.warning(f"Expected '{path_var}' not found!")
            continue
        # panglao has d
        processed_dir = "processed_dynamic" if name == "panglao" else "processed"
        h5ad_path = Path(root_path) / processed_dir / "train" / "processed_final.h5ad"
        if not h5ad_path.exists():
            logger.warning(f"No file found at f{str(h5ad_path)}")
            continue
        dataset_vocab = get_vocab_from_dataset(h5ad_path)
        logger.info(f"{name}: Found {len(dataset_vocab)} gene names in dataset")
        vocabs[name] = dataset_vocab
    return vocabs


def get_union_all_vocabs_gene_list(
    bulk_missing_genes_path="/dccstor/bmfm-targets/data/omics/transcriptome/bulkRNA/ALL/HTS/raw_genes_missing_in_vocab.csv",
):
    """Create gene list that unions all of the gene names we have encountered so far."""
    gene2vec_vocab = {*get_gene2vec_tokenizer().get_field_vocab("genes")[5:]}
    logger.info(f"gene2vec: Found {len(gene2vec_vocab)} gene names")
    bulk_rna_vocab = {*pd.read_csv(bulk_missing_genes_path, index_col=0).squeeze()}
    logger.info(f"bulkrna: Found {len(bulk_rna_vocab)} previously missing gene names")
    cellxgene_gene_list = get_cellxgene_soma_gene_list()
    logger.info(f"cellxgene_soma: Found {len(cellxgene_gene_list)} gene names")
    big_vocabs = {
        "cellxgene": {*cellxgene_gene_list},
        "gene2vec": {*gene2vec_vocab},
        "bulkrna": bulk_rna_vocab,
    }

    data_vocabs = get_data_vocabs_from_environment()
    all_vocabs = {**data_vocabs, **big_vocabs}

    union_all_vocabs = reduce(lambda x, y: x | y, all_vocabs.values())
    logger.info(
        f"After unioning {len(all_vocabs)} sources: Found {len(union_all_vocabs)} gene names."
    )
    return sorted(union_all_vocabs)


def make_all_genes_vocab(vocab_dir, expression_bins=50):
    gene_list = get_union_all_vocabs_gene_list()
    make_gene_vocab(vocab_dir, gene_list, expression_bins)


def make_gene2vec_vocab(
    vocab_dir,
    gene2vec_path="/dccstor/bmfm-targets/data/omics/transcriptome/gene_embeddings/gene2vec_dim_200_iter_9.txt",
    expression_bins=10,
):
    gene_list = [row.split()[0] for row in open(gene2vec_path)]
    make_gene_vocab(vocab_dir, gene_list, expression_bins)


def make_gene_vocab(
    vocab_dir,
    gene_list,
    expression_bins=10,
):
    special_tokens_map = {
        "cls_token": "[CLS]",
        "mask_token": "[MASK]",
        "pad_token": "[PAD]",
        "sep_token": "[SEP]",
        "unk_token": "[UNK]",
    }
    tokenizer_config = {
        "clean_up_tokenization_spaces": True,
        "cls_token": "[CLS]",
        "mask_token": "[MASK]",
        "model_max_length": 1000000000000000019884624838656,
        "pad_token": "[PAD]",
        "sep_token": "[SEP]",
        "strip_accents": None,
        "tokenizer_class": "MultiFieldTokenizer",
        "unk_token": "[UNK]",
    }
    vocab_dir = Path(vocab_dir)
    special_tokens = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]
    expressions = [str(i) for i in range(expression_bins)]
    multifield_vocab = {
        "genes": special_tokens + gene_list,
        "expressions": special_tokens + expressions,
    }
    with open(vocab_dir / "multifield_vocab.json", "w") as f:
        json.dump(multifield_vocab, f)
    with open(vocab_dir / "special_tokens_map.json", "w") as f:
        json.dump(special_tokens_map, f)
    with open(vocab_dir / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f)


def add_special_cls(tokenizer, added_cls_tokens, outdir):
    """
    Add special tokens before vocab, preserving original special token IDs.

    Args:
    ----
        tokenizer: Original tokenizer (BertTokenizerFast/WordLevel)
        added_cls_tokens: List of new special tokens to add
        outdir: Directory to save modified tokenizer
    """
    from transformers import AutoTokenizer

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save original to get JSON files
    tokenizer.save_pretrained(outdir)

    # Load tokenizer.json
    tok_path = outdir / "tokenizer.json"
    with open(tok_path) as f:
        tok_data = json.load(f)

    # Extract original vocab and find insertion point
    vocab = tok_data["model"]["vocab"]
    orig_specials = set(tokenizer.all_special_tokens)

    # Find first non-special token ID (insertion point)
    insert_id = min(id for token, id in vocab.items() if token not in orig_specials)
    n_new = len(added_cls_tokens)

    # Rebuild vocab: shift non-specials, insert new specials
    new_vocab = {}
    for token, id in vocab.items():
        if token in orig_specials:
            new_vocab[token] = id  # Keep original special IDs
        elif id < insert_id:
            new_vocab[token] = id  # Keep tokens before insertion point
        else:
            new_vocab[token] = id + n_new  # Shift regular vocab

    # Insert new special tokens
    for i, token in enumerate(added_cls_tokens):
        new_vocab[token] = insert_id + i

    tok_data["model"]["vocab"] = new_vocab

    # Update added_tokens to include new specials
    added_tokens = tok_data.get("added_tokens", [])
    existing_tokens = {t["content"] for t in added_tokens}

    for i, token in enumerate(added_cls_tokens):
        if token not in existing_tokens:
            added_tokens.append(
                {
                    "id": insert_id + i,
                    "content": token,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                }
            )

    # Sort by ID and update
    added_tokens.sort(key=lambda x: x["id"])
    tok_data["added_tokens"] = added_tokens

    # Save modified tokenizer.json
    with open(tok_path, "w") as f:
        json.dump(tok_data, f, indent=2, ensure_ascii=False)

    # Update special_tokens_map.json
    special_map_path = outdir / "special_tokens_map.json"
    if special_map_path.exists():
        with open(special_map_path) as f:
            special_map = json.load(f)

        # Add new tokens to additional_special_tokens
        if "additional_special_tokens" not in special_map:
            special_map["additional_special_tokens"] = []

        existing_add = set(special_map["additional_special_tokens"])
        for token in added_cls_tokens:
            if token not in existing_add:
                special_map["additional_special_tokens"].append(token)

        with open(special_map_path, "w") as f:
            json.dump(special_map, f, indent=2, ensure_ascii=False)

    # Update tokenizer_config.json with new vocab size and added_tokens_decoder
    config_path = outdir / "tokenizer_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

        config["vocab_size"] = len(new_vocab)

        # Update added_tokens_decoder
        if "added_tokens_decoder" not in config:
            config["added_tokens_decoder"] = {}

        for i, token in enumerate(added_cls_tokens):
            token_id = str(insert_id + i)
            config["added_tokens_decoder"][token_id] = {
                "content": token,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            }

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    # Reload and properly register special tokens, then save again
    reloaded = AutoTokenizer.from_pretrained(outdir)
    reloaded.add_special_tokens({"additional_special_tokens": added_cls_tokens})
    reloaded.save_pretrained(outdir)


def create_new_mft_with_extra_cls_tokens(
    new_tokenizer_path, initial_tokenizer_identifier="all_genes"
):
    from ..load import load_tokenizer

    mft = load_tokenizer(initial_tokenizer_identifier)
    special_cls_tokens = [f"[CLS_{i}]" for i in range(31)]

    for k, subtok in mft.tokenizers.items():
        new_tokenizer_path = Path(new_tokenizer_path)
        outdir = new_tokenizer_path / "tokenizers" / k
        outdir.mkdir(parents=True, exist_ok=True)
        add_special_cls(subtok, special_cls_tokens, outdir)
