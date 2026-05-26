#!/usr/bin/env python3
"""save .ckpt as safe_ternsorts and config.json."""
import argparse
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/dccstor/bmfm-targets1/users/sivanra/models"),
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="ibm-research/biomed.rna.llama.47m.wced.multitask.v1",
    )
    args = parser.parse_args()

    cache_dir = snapshot_download(
        repo_id=args.model_id,
        allow_patterns=["*.ckpt"],
    )

    model_name = args.model_id.replace("ibm-research/", "")
    output_dir = args.output_dir / model_name

    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_file = list(Path(cache_dir).glob("*.ckpt"))[0]
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    convert_checkpoint_to_safetensors(ckpt, output_dir)

    model_config = ckpt["hyper_parameters"]["model_config"]
    model_config.save_pretrained(output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
