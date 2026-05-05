#!/usr/bin/env bash
# Zero-shot embeddings from a v2 (March 2026) Llama checkpoint.
#
# v2 checkpoints were trained with scale_adapt encoder + log-normalized input,
# max_length 8192. They're smaller (~32-47M params) and use Llama arch instead
# of BERT. Set CHECKPOINT to either:
#   ibm-research/biomed.rna.llama.32m.mlm.multitask.v1   (MLM + multitask)
#   ibm-research/biomed.rna.llama.47m.wced.multitask.v1  (WCED + multitask)
#
# Usage:
#   INPUT_FILE=/path/to/data.h5ad WORKING_DIR=/path/to/out \
#     CHECKPOINT=ibm-research/biomed.rna.llama.47m.wced.multitask.v1 \
#     bash predict_llama_v2.sh

set -euo pipefail

: "${INPUT_FILE:?set INPUT_FILE=/path/to/data.h5ad}"
: "${WORKING_DIR:=./outputs/predict_llama_v2}"
: "${CHECKPOINT:=ibm-research/biomed.rna.llama.47m.wced.multitask.v1}"

bmfm-targets-run -cd run -cn predict \
  input_file="$INPUT_FILE" \
  working_dir="$WORKING_DIR" \
  checkpoint="$CHECKPOINT" \
  data_module.log_normalize_transform=true \
  data_module.max_length=2048
