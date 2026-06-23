#!/usr/bin/env python3
"""
Download checkpoint from HuggingFace, convert to SafeTensors, and upload to new repo.

Setup:
    export HF_TOKEN="hf_..."  # Get token from https://huggingface.co/settings/tokens

Usage:
    python create_vllm_compatible_hf_model_repo.py --source-repo "ibm-research/biomed.rna.llama.47m.wced.multitask.v1" --target-repo "ibm-research/biomed.rna.llama.47m.wced.multitask.v1.vllm"
    python create_vllm_compatible_hf_model_repo.py --source-repo "ibm-research/biomed.rna.llama.32m.mlm.multitask.v1" --target-repo "ibm-research/biomed.rna.llama.32m.mlm.multitask.v1.vllm"

"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import save_file


def to_json_serializable(obj):
    """Recursively convert objects to JSON-serializable format."""
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        return to_json_serializable(obj.to_dict())
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return to_json_serializable(obj.__dict__)
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [to_json_serializable(item) for item in obj]
    if isinstance(obj, str | int | float | bool | type(None)):
        return obj
    return str(obj)


def add_vllm_required_keys(config_dict: dict) -> dict:
    """
    Modify config for vLLM compatibility.

    - Adds architectures key required for vLLM plugin registration
    - Preserves original model_type (scllama)
    - Sets encoder/decoder flags for proper model initialization
    - Adds vocab_size at root level for vLLM dummy tokenizer creation
    """
    # Set standard architecture for vLLM plugin
    config_dict["architectures"] = ["BiomedRnaForSequenceEmbedding"]

    # Keep original model_type as scllama (matches local working config)
    config_dict["model_type"] = "scllama"

    # Set model architecture flags
    config_dict["use_cache"] = False
    config_dict["is_encoder_decoder"] = False
    config_dict["is_decoder"] = False
    config_dict["add_cross_attention"] = False

    # Extract vocab_size from fields and add to root level
    # This allows vLLM to create a dummy tokenizer automatically
    if "fields" in config_dict and len(config_dict["fields"]) > 0:
        # Use the genes field vocab_size (first field)
        genes_field = next(
            (f for f in config_dict["fields"] if f.get("field_name") == "genes"), None
        )
        if genes_field and "vocab_size" in genes_field:
            config_dict["vocab_size"] = genes_field["vocab_size"]

    return config_dict


def create_readme(
    source_repo: str, target_repo: str, original_readme: str | None = None
) -> str:
    """Create README with reference to original model, preserving YAML frontmatter."""
    if not original_readme:
        # Create minimal README if none exists
        return f"""---
license: apache-2.0
library_name: biomed-multi-omic
pipeline_tag: feature-extraction
tags:
- Biology
- RNA
- vLLM
---

# {target_repo.split('/')[-1]}

SafeTensors version of [{source_repo}](https://huggingface.co/{source_repo}) for vLLM.

> **Note**: Converted for use with the [vLLM BiomedRNA plugin](https://github.com/BiomedSciAI/biomed-multi-omic/tree/main/vllm).

## Usage
```python
from biomed_rna_plugin import get_vllm_biomed_rna_model
llm = get_vllm_biomed_rna_model(model_path="{target_repo}")
```

## Original Model
See: [{source_repo}](https://huggingface.co/{source_repo})
"""

    # Parse original README to extract and preserve YAML frontmatter
    lines = original_readme.split("\n")
    if lines[0].strip() == "---":
        # Find end of YAML frontmatter
        yaml_end = -1
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                yaml_end = i
                break

        if yaml_end > 0:
            # Preserve YAML, add note after it
            yaml_section = "\n".join(lines[: yaml_end + 1])
            rest_of_readme = "\n".join(lines[yaml_end + 1 :])
            note = f"""

> **Note**: This model is converted from [{source_repo}](https://huggingface.co/{source_repo}) to SafeTensors format for use with the [vLLM BiomedRNA plugin](https://github.com/BiomedSciAI/biomed-multi-omic/tree/main/vllm).
"""
            return yaml_section + note + rest_of_readme

    # No YAML frontmatter found, just prepend note
    note = f"""> **Note**: This model is converted from [{source_repo}](https://huggingface.co/{source_repo}) to SafeTensors format for use with the [vLLM BiomedRNA plugin](https://github.com/BiomedSciAI/biomed-multi-omic/tree/main/vllm).

"""
    return note + original_readme


def main():
    parser = argparse.ArgumentParser(description="Convert and upload checkpoint to HF")
    parser.add_argument(
        "--source-repo", default="ibm-research/biomed.rna.llama.47m.wced.multitask.v1"
    )
    parser.add_argument("--target-repo", required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--commit-message", default="Convert to SafeTensors for vLLM")
    args = parser.parse_args()

    print(f"Converting {args.source_repo} → {args.target_repo}")

    # Download and load checkpoint
    cache_dir = snapshot_download(args.source_repo, allow_patterns=["*.ckpt"])
    ckpt_file = next(Path(cache_dir).glob("*.ckpt"))
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "model"
        output_dir.mkdir(parents=True)

        # Save SafeTensors
        save_file(ckpt["state_dict"], str(output_dir / "model.safetensors"))

        # Save metadata
        metadata = {
            k: to_json_serializable(ckpt[k])
            for k in ["hyper_parameters", "training_config", "label_dict"]
            if k in ckpt
        }
        if metadata:
            (output_dir / "checkpoint_metadata.json").write_text(
                json.dumps(metadata, indent=2)
            )

        # Save and modify config
        model_config = ckpt["hyper_parameters"]["model_config"]
        model_config.save_pretrained(output_dir)

        config_path = output_dir / "config.json"
        config_dict = json.loads(config_path.read_text())
        config_dict = add_vllm_required_keys(config_dict)
        config_path.write_text(json.dumps(config_dict, indent=2))

        # Copy tokenizer files (including subdirectories)
        print("Copying tokenizer files...")
        try:
            tok_cache = snapshot_download(
                args.source_repo, allow_patterns=["tokenizers/**/*"]
            )

            # Copy entire tokenizers directory if it exists
            tok_src = Path(tok_cache) / "tokenizers"
            if tok_src.exists():
                tok_dst = output_dir / "tokenizers"
                shutil.copytree(tok_src, tok_dst)

                # Count files copied
                file_count = sum(1 for _ in tok_dst.rglob("*") if _.is_file())
                print(f"  ✓ Copied tokenizers directory with {file_count} files")
            else:
                print("  ⚠ No tokenizers directory found in source repo")

        except Exception as e:
            print(f"  ⚠ Warning: Could not copy tokenizer files: {e}")

        # Create README
        try:
            readme_cache = snapshot_download(
                args.source_repo, allow_patterns=["README.md"]
            )
            original = (Path(readme_cache) / "README.md").read_text()
        except Exception:
            original = None
        (output_dir / "README.md").write_text(
            create_readme(args.source_repo, args.target_repo, original)
        )

        # Upload (uses HF_TOKEN env var)
        api = HfApi()
        api.create_repo(args.target_repo, private=args.private, exist_ok=True)
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=args.target_repo,
            commit_message=args.commit_message,
        )

    print(f"✓ Uploaded to https://huggingface.co/{args.target_repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
