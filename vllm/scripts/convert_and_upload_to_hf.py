#!/usr/bin/env python3
"""
Download checkpoint from HuggingFace, convert to SafeTensors, and upload to new repo.

Usage:
    python convert_and_upload_to_hf.py --target-repo "username/model-name" --token "hf_..."
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
    if isinstance(obj, (list, tuple)):
        return [to_json_serializable(item) for item in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


def modify_config(config_dict: dict) -> dict:
    """
    Modify config for vLLM compatibility.
    
    PLACEHOLDER: Add your custom modifications here.
    Example: config_dict["use_cache"] = False
    """
    return config_dict


def create_readme(source_repo: str, target_repo: str, original_readme: str | None = None) -> str:
    """Create README with reference to original model."""
    header = f"""---
# Converted from [{source_repo}](https://huggingface.co/{source_repo})
# For vLLM plugin: https://github.com/BiomedSciAI/biomed-multi-omic/tree/main/vllm
---

"""
    if original_readme:
        return header + original_readme
    
    return f"""{header}# {target_repo.split('/')[-1]}

SafeTensors version of [{source_repo}](https://huggingface.co/{source_repo}) for vLLM.

## Usage
```python
from vllm_biomed_rna_plugin import get_vllm_biomed_rna_model
llm = get_vllm_biomed_rna_model(model_path="{target_repo}")
```
"""


def main():
    parser = argparse.ArgumentParser(description="Convert and upload checkpoint to HF")
    parser.add_argument("--source-repo", default="ibm-research/biomed.rna.llama.47m.wced.multitask.v1")
    parser.add_argument("--target-repo", required=True)
    parser.add_argument("--token", default=None)
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
        metadata = {k: to_json_serializable(ckpt[k]) 
                   for k in ["hyper_parameters", "training_config", "label_dict"] 
                   if k in ckpt}
        if metadata:
            (output_dir / "checkpoint_metadata.json").write_text(json.dumps(metadata, indent=2))

        # Save and modify config
        model_config = ckpt["hyper_parameters"]["model_config"]
        model_config.save_pretrained(output_dir)
        
        config_path = output_dir / "config.json"
        config_dict = json.loads(config_path.read_text())
        config_dict = modify_config(config_dict)
        config_path.write_text(json.dumps(config_dict, indent=2))

        # Copy tokenizer files
        try:
            tok_cache = snapshot_download(args.source_repo, 
                                         allow_patterns=["tokenizer*", "special_tokens_map.json", "vocab.json"])
            for pattern in ["tokenizer*", "special_tokens_map.json", "vocab.json"]:
                for file in Path(tok_cache).glob(pattern):
                    shutil.copy(file, output_dir)
        except Exception:
            pass

        # Create README
        try:
            readme_cache = snapshot_download(args.source_repo, allow_patterns=["README.md"])
            original = (Path(readme_cache) / "README.md").read_text()
        except Exception:
            original = None
        (output_dir / "README.md").write_text(create_readme(args.source_repo, args.target_repo, original))

        # Upload
        api = HfApi(token=args.token)
        api.create_repo(args.target_repo, private=args.private, exist_ok=True)
        api.upload_folder(folder_path=str(output_dir), repo_id=args.target_repo, 
                         commit_message=args.commit_message)

    print(f"✓ Uploaded to https://huggingface.co/{args.target_repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Made with Bob
