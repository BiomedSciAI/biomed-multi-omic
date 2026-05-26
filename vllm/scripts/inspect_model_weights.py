#!/usr/bin/env python3
"""Inspect model weights and structure from SafeTensors files."""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

try:
    from safetensors import safe_open
except ImportError:
    print("ERROR: safetensors not installed. Install with: pip install safetensors")
    sys.exit(1)


def analyze_weights(model_path: str, output_file: str = None):
    """Analyze model weights structure."""
    print(f"\nAnalyzing: {model_path}")

    with safe_open(model_path, framework="pt") as f:
        keys = list(f.keys())
        prefixes = defaultdict(list)
        total_params = 0

        for key in keys:
            tensor = f.get_tensor(key)
            numel = tensor.numel()
            total_params += numel
            prefix = key.split(".")[0] if "." in key else key
            prefixes[prefix].append((key, tensor.shape, numel))

        print(
            f"Total: {len(keys)} tensors, {total_params:,} params (~{total_params / 1e6:.1f}M)"
        )

        print("\nModule Structure:")
        for prefix in sorted(prefixes.keys()):
            params = prefixes[prefix]
            prefix_params = sum(numel for _, _, numel in params)
            print(f"  {prefix}/: {len(params)} tensors, {prefix_params:,} params")

            submodules = defaultdict(list)
            for key, shape, numel in params:
                parts = key.split(".")
                subprefix = ".".join(parts[:2]) if len(parts) > 1 else key
                submodules[subprefix].append((key, shape, numel))

            for subprefix in sorted(submodules.keys())[:5]:
                sub_params = submodules[subprefix]
                sub_count = sum(numel for _, _, numel in sub_params)
                print(
                    f"    {subprefix}: {len(sub_params)} tensors, {sub_count:,} params"
                )

            if len(submodules) > 5:
                print(f"    ... and {len(submodules) - 5} more submodules")

        print("\nSample Keys:")
        for i, key in enumerate(sorted(keys)[:10], 1):
            tensor = f.get_tensor(key)
            shape_str = "x".join(map(str, tensor.shape))
            print(f"  {i}. {key}: {shape_str}")
        if len(keys) > 10:
            print(f"  ... and {len(keys) - 10} more")

        output_file = Path(
            output_file or Path(model_path).parent / "model_weight_keys.txt"
        )
        with open(output_file, "w") as out:
            out.write(f"Model: {model_path}\n")
            out.write(f"Total: {len(keys)} keys, {total_params:,} params\n\n")
            for key in sorted(keys):
                tensor = f.get_tensor(key)
                shape_str = "x".join(map(str, tensor.shape))
                out.write(f"{key}: {shape_str}\n")

        print(f"\nFull key list saved to: {output_file}")


DEFAULT_MODEL_PATH = "/dccstor/bmfm-targets1/users/sivanra/models/biomed.rna.llama.47m.wced.multitask.v1/model.safetensors"


def main():
    parser = argparse.ArgumentParser(
        description="Inspect SafeTensors model weights and structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect default BiomedRNA model
  python scripts/inspect_model_weights.py

  # Inspect custom model
  python scripts/inspect_model_weights.py /path/to/model.safetensors

  # Save output to custom location
  python scripts/inspect_model_weights.py --output keys.txt
        """,
    )

    parser.add_argument(
        "model_path",
        nargs="?",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to model.safetensors file (default: {DEFAULT_MODEL_PATH})",
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Output file for full key list (default: model_weight_keys.txt in model directory)",
    )

    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)

    analyze_weights(str(model_path), args.output)


if __name__ == "__main__":
    main()

# Made with Bob
