import json
import logging
from pathlib import Path

from bmfm_targets.tokenization.load import get_all_genes_v2_tokenizer
from bmfm_targets.tokenization.resources.reference_data import get_protein_coding_genes

logger = logging.getLogger(__name__)


def create_subset_tokenizer(
    base_mft, field: str, allowed_tokens: set, output_dir: Path
):
    field_dir = output_dir / "tokenizers" / field
    field_dir.mkdir(parents=True, exist_ok=True)

    # 1. Determine Vocabulary
    tok = base_mft.get_field_tokenizer(field)
    decoder = tok.backend_tokenizer.get_added_tokens_decoder()
    # Get all specials (explicit list + hidden in decoder like [S])
    specials = set(tok.all_special_tokens) | {
        t.content for t in decoder.values() if t.special
    }

    # Build new vocab: Specials (ordered by old ID) -> Existing Allowed -> New Allowed
    old_vocab = sorted(tok.get_vocab().items(), key=lambda x: x[1])
    new_tokens = list(
        dict.fromkeys(
            [t for t, _ in old_vocab if t in specials]
            + [t for t, _ in old_vocab if t in allowed_tokens]
            + sorted(allowed_tokens)
        )
    )
    new_vocab_map = {t: i for i, t in enumerate(new_tokens)}

    # 2. Save Baseline & Overwrite Vocab
    tok.save_pretrained(str(field_dir))
    (field_dir / "vocab.txt").write_text("\n".join(new_tokens))

    # 3. Update tokenizer.json (Preserves pre_tokenizer, updates IDs)
    json_path = field_dir / "tokenizer.json"
    data = json.loads(json_path.read_text())
    data["model"]["vocab"] = new_vocab_map

    # Rebuild added_tokens list ensuring object format and correct IDs
    tmpl = {
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True,
    }
    existing_added = {t["content"]: t for t in data.get("added_tokens", [])}

    data["added_tokens"] = []
    for t in new_tokens:
        if t in specials:
            # Use existing attributes or default template, update ID
            entry = existing_added.get(t, {"content": t, **tmpl})
            entry["id"] = new_vocab_map[t]
            data["added_tokens"].append(entry)

    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    # 4. Sanitize config (remove decoder to prevent ID collisions on load)
    cfg_path = field_dir / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg.pop("added_tokens_decoder", None)
    cfg_path.write_text(json.dumps(cfg, indent=2))

    # 5. Enforce Object Style in special_tokens_map.json
    stm_path = field_dir / "special_tokens_map.json"
    if stm_path.exists():
        stm = json.loads(stm_path.read_text())
        # Convert all entries to full objects
        fmt = (
            lambda x: x
            if isinstance(x, dict)
            else {
                "content": x,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
            }
        )
        stm = {
            k: [fmt(i) for i in v] if isinstance(v, list) else fmt(v)
            for k, v in stm.items()
        }
        stm_path.write_text(json.dumps(stm, indent=2))


def create_protein_coding_tokenizer(output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning reference tokenizer...")
    base_mft = get_all_genes_v2_tokenizer()
    base_mft.save_pretrained(str(out))

    logger.info("Subsetting 'genes' field...")
    create_subset_tokenizer(base_mft, "genes", set(get_protein_coding_genes()), out)
    logger.info("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    create_protein_coding_tokenizer(parser.parse_args().output_dir)
