#!/usr/bin/env python
"""
Migrate old MLM/SeqCls/SeqLabel checkpoints to multitask format.

Usage:
    python run/migrate_checkpoints_to_multitask.py --input old.ckpt --output new.ckpt
    python run/migrate_checkpoints_to_multitask.py --input-dir ./old/ --output-dir ./new/
    python run/migrate_checkpoints_to_multitask.py --input old.ckpt --output new.ckpt --label-name cell_type
"""

import argparse
from pathlib import Path

import torch


def detect_checkpoint_type(state_dict: dict) -> tuple[str, str | None]:
    """
    Detect checkpoint type from state_dict keys.

    Note: MLM and sequence_labeling have identical state dicts and convert the same way,
    so we treat them as "mlm_or_seqlabel" for migration purposes.

    Returns
    -------
        (checkpoint_type, label_column_name_if_found)
    """
    keys = set(state_dict.keys())

    # Modern multitask has nested predictions head (with or without base_model prefix from LoRA)
    if any("cls.predictions.predictions" in k for k in keys):
        return ("multitask", None)

    # Old multitask_classifier has classifiers.{label_name} structure
    if any("classifiers." in k for k in keys):
        # Extract label name from first classifier key
        for key in keys:
            if "classifiers." in key:
                parts = key.split(".")
                idx = parts.index("classifiers")
                if idx + 1 < len(parts):
                    return ("multitask_classifier", parts[idx + 1])
        return ("multitask_classifier", None)

    # Old sequence_classification has classifier (singular) head
    if any("classifier" in k for k in keys):
        return ("sequence_classification", None)

    # Old MLM/SeqLabel has cls.predictions (but not nested)
    if any(
        "cls.predictions" in k and "cls.predictions.predictions" not in k for k in keys
    ):
        return ("mlm_or_seqlabel", None)

    raise ValueError(f"Unknown checkpoint type. Keys: {list(keys)[:10]}")


def convert_mlm_to_multitask(state_dict: dict) -> dict:
    """Convert MLM checkpoint to multitask format."""
    new_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("cls.predictions") and not key.startswith(
            "cls.predictions.predictions"
        ):
            # Nest MLM head under predictions.predictions
            new_key = key.replace("cls.predictions", "cls.predictions.predictions", 1)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    # Detect model prefix from checkpoint keys
    # ModernBERT has "scmodernbert.*" prefix, scBERT has "scbert.*"
    is_modernbert = any(k.startswith("scmodernbert.") for k in state_dict.keys())
    model_prefix = "scmodernbert" if is_modernbert else "scbert"

    # Initialize pooler as identity to preserve first-token behavior
    # MLM models don't have pooler (add_pooling_layer=False), but multitask models need it
    # Infer hidden_size from embeddings
    hidden_size = None
    for key, value in state_dict.items():
        if "embeddings" in key and "weight" in key and value.ndim >= 2:
            hidden_size = value.shape[-1]
            break

    if hidden_size is not None:
        pooler_weight = torch.eye(hidden_size)
        pooler_bias = torch.zeros(hidden_size)

        # Add pooler under submodule name (e.g., scbert.pooler or scmodernbert.pooler)
        new_state_dict[f"{model_prefix}.pooler.dense.weight"] = pooler_weight
        new_state_dict[f"{model_prefix}.pooler.dense.bias"] = pooler_bias

    return new_state_dict


def convert_seqcls_to_multitask(
    state_dict: dict, label_name: str | None = None
) -> dict:
    """
    Convert sequence classification checkpoint to multitask format.

    Args:
    ----
        state_dict: Checkpoint state dict
        label_name: Label column name (required if not auto-detected from MultiTaskClassifier)
    """
    new_state_dict = {}

    # Try to auto-detect from MultiTaskClassifier structure
    detected_type, detected_label = detect_checkpoint_type(state_dict)

    if detected_type == "multitask_classifier":
        # Unwrap MultiTaskClassifier
        # Detect if this is llama (has base_model.core.*) or scbert (has base_model.embeddings.*)
        is_llama = any("base_model.core." in k for k in state_dict.keys())

        for key, value in state_dict.items():
            if key.startswith("base_model.model."):
                # Remove base_model.model prefix -> becomes scbert.*
                new_key = key.replace("base_model.model.", "scbert.")
                new_state_dict[new_key] = value
            elif key.startswith("base_model.core.") and is_llama:
                # Llama: base_model.core.* -> core.*
                new_key = key.replace("base_model.", "")
                new_state_dict[new_key] = value
            elif key.startswith("base_model."):
                # SCBert: base_model.* -> scbert.*
                new_key = key.replace("base_model.", "scbert.")
                new_state_dict[new_key] = value
            elif key.startswith("classifiers."):
                # Map classifiers.{label}.* to cls.label_predictions.predictions.label_decoders.{label}.decoder.*
                parts = key.split(".")
                if len(parts) >= 3:
                    label = parts[1]
                    rest = ".".join(parts[2:])
                    new_key = f"cls.label_predictions.predictions.label_decoders.{label}.decoder.{rest}"
                    new_state_dict[new_key] = value
        return new_state_dict

    # Standard SeqCls conversion
    if label_name is None:
        raise ValueError(
            "label_name required for SeqCls conversion. "
            "Provide via --label-name argument or use a MultiTaskClassifier checkpoint."
        )

    for key, value in state_dict.items():
        if "classifier" in key:
            # Map classifier to label_predictions
            new_key = key.replace(
                "classifier",
                f"cls.label_predictions.predictions.label_decoders.{label_name}.decoder",
            )
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    # MLM head will be initialized from scratch if needed
    return new_state_dict


def convert_seqlabel_to_multitask(state_dict: dict) -> dict:
    """Convert sequence labeling checkpoint to multitask format."""
    # Similar to MLM conversion
    return convert_mlm_to_multitask(state_dict)


def migrate_checkpoint(
    input_path: str, output_path: str, label_name: str | None = None
):
    """Main migration function."""
    print(f"Loading checkpoint from {input_path}")
    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)

    # Handle both raw state_dict and full checkpoint
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        is_full_ckpt = True
    else:
        state_dict = ckpt
        is_full_ckpt = False

    # Detect type
    ckpt_type, detected_label = detect_checkpoint_type(state_dict)
    print(f"Detected checkpoint type: {ckpt_type}")
    if detected_label:
        print(f"Auto-detected label column: {detected_label}")

    if ckpt_type == "multitask":
        print("Already in multitask format, copying...")
        torch.save(ckpt, output_path)
        return

    # Convert
    if ckpt_type == "mlm_or_seqlabel":
        new_state_dict = convert_mlm_to_multitask(state_dict)
    elif ckpt_type == "sequence_classification":
        new_state_dict = convert_seqcls_to_multitask(state_dict, label_name=label_name)
    elif ckpt_type == "multitask_classifier":
        new_state_dict = convert_seqcls_to_multitask(state_dict)  # auto-detects label
    else:
        raise ValueError(f"Unknown checkpoint type: {ckpt_type}")

    # Save
    if is_full_ckpt:
        ckpt["state_dict"] = new_state_dict
        torch.save(ckpt, output_path)
    else:
        torch.save(new_state_dict, output_path)

    print(f"Migrated checkpoint saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate old model checkpoints to multitask format"
    )
    parser.add_argument("--input", help="Input checkpoint path")
    parser.add_argument("--output", help="Output checkpoint path")
    parser.add_argument(
        "--label-name",
        help="Label column name (required for non-MultiTaskClassifier SeqCls checkpoints)",
    )
    parser.add_argument("--input-dir", help="Input directory for batch processing")
    parser.add_argument("--output-dir", help="Output directory for batch processing")

    args = parser.parse_args()

    if args.input and args.output:
        migrate_checkpoint(args.input, args.output, args.label_name)
    elif args.input_dir and args.output_dir:
        # Batch processing
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for ckpt_file in input_dir.glob("*.ckpt"):
            output_file = output_dir / ckpt_file.name
            migrate_checkpoint(str(ckpt_file), str(output_file), args.label_name)
    else:
        parser.print_help()

# Made with Bob
