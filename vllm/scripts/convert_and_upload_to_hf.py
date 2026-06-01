#!/usr/bin/env python3
"""
Download checkpoint from HuggingFace, convert to SafeTensors, modify config, and upload to new HF repo.

This script:
1. Downloads the original checkpoint from HuggingFace
2. Converts .ckpt to model.safetensors
3. Extracts and saves config.json
4. Allows config modifications (placeholder for custom changes)
5. Uploads everything to a new HuggingFace repository
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


def convert_checkpoint_to_safetensors(checkpoint: dict, output_dir: Path) -> Path:
    """
    Convert PyTorch Lightning checkpoint to SafeTensors format.

    Args:
    ----
        checkpoint: Loaded checkpoint dict with 'state_dict' key
        output_dir: Directory to save model.safetensors

    Returns:
    -------
        Path to the created model directory
    """
    # Save state_dict as model.safetensors
    state_dict = checkpoint["state_dict"]
    safetensors_file = output_dir / "model.safetensors"
    save_file(state_dict, str(safetensors_file))
    print(
        f"✓ Saved model.safetensors ({safetensors_file.stat().st_size / (1024**2):.1f} MB)"
    )

    # Save checkpoint metadata if available
    metadata = {}
    for key in ["hyper_parameters", "training_config", "label_dict"]:
        if key in checkpoint:
            metadata[key] = _to_json_serializable(checkpoint[key])

    if metadata:
        metadata_file = output_dir / "checkpoint_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print("✓ Saved checkpoint_metadata.json")

    return output_dir


def _to_json_serializable(obj):
    """Recursively convert objects to JSON-serializable format."""
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        return _to_json_serializable(obj.to_dict())
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return _to_json_serializable(obj.__dict__)
    if hasattr(obj, "item"):  # numpy/torch scalar
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_json_serializable(item) for item in obj]
    if isinstance(obj, str | int | float | bool | type(None)):
        return obj
    return str(obj)


def modify_config(config_dict: dict) -> dict:
    """
    Modify config.json for vLLM compatibility.
    
    PLACEHOLDER: Add your custom config modifications here.
    
    Args:
    ----
        config_dict: Original config dictionary
        
    Returns:
    -------
        Modified config dictionary
    """
    # Example modifications (customize as needed):
    # config_dict["some_key"] = "some_value"
    # config_dict["architectures"] = ["BiomedRnaForSequenceEmbedding"]
    
    print("\n" + "="*80)
    print("PLACEHOLDER: Config Modification Section")
    print("="*80)
    print("Add your config modifications in the modify_config() function")
    print("Current config keys:", list(config_dict.keys()))
    print("="*80 + "\n")
    
    # Add any modifications here
    # For example:
    # if "use_cache" in config_dict:
    #     config_dict["use_cache"] = False
    
    return config_dict


def main():
    parser = argparse.ArgumentParser(
        description="Convert checkpoint to SafeTensors and upload to HuggingFace"
    )
    parser.add_argument(
        "--source-repo",
        type=str,
        default="ibm-research/biomed.rna.llama.47m.wced.multitask.v1",
        help="Source HuggingFace repository ID",
    )
    parser.add_argument(
        "--target-repo",
        type=str,
        required=True,
        help="Target HuggingFace repository ID (e.g., 'your-username/model-name')",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the target repository private",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Convert checkpoint to SafeTensors format for vLLM",
        help="Commit message for the upload",
    )
    args = parser.parse_args()

    print("="*80)
    print("HuggingFace Checkpoint Converter & Uploader")
    print("="*80)
    print(f"Source repo: {args.source_repo}")
    print(f"Target repo: {args.target_repo}")
    print(f"Private: {args.private}")
    print("="*80 + "\n")

    # Step 1: Download checkpoint from source repo
    print("[1/5] Downloading checkpoint from HuggingFace...")
    cache_dir = snapshot_download(
        repo_id=args.source_repo,
        allow_patterns=["*.ckpt"],
    )
    print(f"✓ Downloaded to: {cache_dir}")

    # Step 2: Load checkpoint
    print("\n[2/5] Loading checkpoint...")
    ckpt_file = list(Path(cache_dir).glob("*.ckpt"))[0]
    print(f"Loading: {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    print("✓ Checkpoint loaded")

    # Step 3: Convert to SafeTensors and save config
    print("\n[3/5] Converting to SafeTensors format...")
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "model"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert checkpoint to safetensors
        convert_checkpoint_to_safetensors(ckpt, output_dir)

        # Extract and save model config
        print("\n[4/5] Extracting and modifying config...")
        model_config = ckpt["hyper_parameters"]["model_config"]
        
        # Save original config first
        model_config.save_pretrained(output_dir)
        print("✓ Saved original config.json")
        
        # Load config for modification
        config_path = output_dir / "config.json"
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Apply modifications (placeholder)
        config_dict = modify_config(config_dict)
        
        # Save modified config
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        print("✓ Saved modified config.json")

        # Copy tokenizer files if they exist in the source repo
        print("\nChecking for tokenizer files...")
        try:
            tokenizer_cache = snapshot_download(
                repo_id=args.source_repo,
                allow_patterns=["tokenizer*", "special_tokens_map.json", "vocab.json"],
            )
            for file in Path(tokenizer_cache).glob("tokenizer*"):
                shutil.copy(file, output_dir)
                print(f"✓ Copied {file.name}")
            for file in Path(tokenizer_cache).glob("special_tokens_map.json"):
                shutil.copy(file, output_dir)
                print(f"✓ Copied {file.name}")
            for file in Path(tokenizer_cache).glob("vocab.json"):
                shutil.copy(file, output_dir)
                print(f"✓ Copied {file.name}")
        except Exception as e:
            print(f"Note: Could not copy tokenizer files: {e}")

        # Copy and modify README/model card
        print("\nProcessing README/model card...")
        try:
            readme_cache = snapshot_download(
                repo_id=args.source_repo,
                allow_patterns=["README.md"],
            )
            readme_file = Path(readme_cache) / "README.md"
            if readme_file.exists():
                # Read original README
                with open(readme_file, "r") as f:
                    original_readme = f.read()
                
                # Prepend reference text
                reference_text = f"""---
# This model is a copy of [{args.source_repo}](https://huggingface.co/{args.source_repo})
# Converted to SafeTensors format for use with the vLLM BiomedRNA plugin
# Plugin repository: https://github.com/BiomedSciAI/biomed-multi-omic/tree/main/vllm
---

"""
                modified_readme = reference_text + original_readme
                
                # Save modified README
                with open(output_dir / "README.md", "w") as f:
                    f.write(modified_readme)
                print("✓ Copied and modified README.md")
            else:
                # Create minimal README if none exists
                minimal_readme = f"""# {args.target_repo.split('/')[-1]}

This model is a copy of [{args.source_repo}](https://huggingface.co/{args.source_repo}) converted to SafeTensors format for use with the vLLM BiomedRNA plugin.

**Plugin repository:** https://github.com/BiomedSciAI/biomed-multi-omic/tree/main/vllm

## Usage

```python
from vllm_biomed_rna_plugin import get_vllm_biomed_rna_model

llm = get_vllm_biomed_rna_model(model_path="{args.target_repo}")
```

For more information, see the original model: [{args.source_repo}](https://huggingface.co/{args.source_repo})
"""
                with open(output_dir / "README.md", "w") as f:
                    f.write(minimal_readme)
                print("✓ Created README.md")
        except Exception as e:
            print(f"Note: Could not process README: {e}")

        # Step 5: Upload to HuggingFace
        print(f"\n[5/5] Uploading to HuggingFace: {args.target_repo}")
        api = HfApi(token=args.token)
        
        # Create repo if it doesn't exist
        try:
            api.create_repo(
                repo_id=args.target_repo,
                private=args.private,
                exist_ok=True,
            )
            print(f"✓ Repository created/verified: {args.target_repo}")
        except Exception as e:
            print(f"Note: {e}")

        # Upload all files
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=args.target_repo,
            commit_message=args.commit_message,
        )
        print(f"✓ Uploaded to https://huggingface.co/{args.target_repo}")

    print("\n" + "="*80)
    print("✓ CONVERSION AND UPLOAD COMPLETE!")
    print("="*80)
    print(f"\nYour model is now available at:")
    print(f"https://huggingface.co/{args.target_repo}")
    print("\nYou can now use it with vLLM:")
    print(f'  llm = get_vllm_biomed_rna_model(model_path="{args.target_repo}")')
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())

# Made with Bob
