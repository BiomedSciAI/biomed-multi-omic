#!/usr/bin/env bash
# Zero-shot embeddings + multitask predictions from the WCED+multitask v1 checkpoint.
#
# This checkpoint was trained with log-normalized input (log_normalize_transform=true)
# and has classification heads for cell_type, tissue, tissue_general, donor_id,
# suspension_type. The predict task dumps the CLS embedding + per-head predictions.
#
# Outputs land in $WORKING_DIR as embeddings.csv + predictions.csv.
#
# Usage:
#   INPUT_FILE=/path/to/data.h5ad WORKING_DIR=/path/to/out bash predict_wced_multitask.sh

set -euo pipefail

: "${INPUT_FILE:?set INPUT_FILE=/path/to/data.h5ad}"
: "${WORKING_DIR:=./outputs/predict_wced_multitask}"

bmfm-targets-run -cd run -cn predict \
  input_file="$INPUT_FILE" \
  working_dir="$WORKING_DIR" \
  checkpoint=ibm-research/biomed.rna.bert.110m.wced.multitask.v1 \
  data_module.log_normalize_transform=true \
  data_module.max_length=1024
