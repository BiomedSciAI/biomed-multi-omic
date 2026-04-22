#!/usr/bin/env bash
# Zero-shot embeddings + predictions from the MLM+RDA v1 checkpoint on an h5ad.
#
# This checkpoint expects RAW counts (not log-normalized) and applies its own
# RDA transform internally. The two must-have overrides vs. the finetune/other
# predict commands are:
#   data_module.log_normalize_transform=false   (RDA wants raw)
#   ++data_module.rda_transform=auto_align      (optional; default is downsample)
#
# Outputs land in $WORKING_DIR as embeddings.csv + predictions.csv.
#
# Usage:
#   INPUT_FILE=/path/to/data.h5ad WORKING_DIR=/path/to/out bash predict_mlm_rda.sh

set -euo pipefail

: "${INPUT_FILE:?set INPUT_FILE=/path/to/data.h5ad}"
: "${WORKING_DIR:=./outputs/predict_mlm_rda}"

bmfm-targets-run -cd run -cn predict \
  input_file="$INPUT_FILE" \
  working_dir="$WORKING_DIR" \
  checkpoint=ibm-research/biomed.rna.bert.110m.mlm.rda.v1 \
  data_module.log_normalize_transform=false \
  data_module.max_length=1024 \
  ++data_module.rda_transform=auto_align
