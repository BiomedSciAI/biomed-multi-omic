#!/usr/bin/env bash
# Zero-shot embeddings + predictions from a DNA (ModernBERT) checkpoint on a csv.
#
# DNA inference expects a directory containing {train,test,dev}.csv. Column 0
# is the nucleotide sequence; remaining columns are labels (ignored at predict
# time but must be present structurally).
#
# Usage:
#   INPUT_DIR=/path/to/csvs INPUT_FILENAME=test.csv WORKING_DIR=/path/to/out \
#     bash dna_predict.sh

set -euo pipefail

: "${INPUT_DIR:?set INPUT_DIR=/path/to/csv_directory}"
: "${INPUT_FILENAME:=test.csv}"
: "${WORKING_DIR:=./outputs/dna_predict}"
: "${CHECKPOINT:=ibm-research/biomed.dna.modernbert.113m.v1}"
: "${DATASET_NAME:=coreprom}"

bmfm-targets-run -cd run -cn dna_predict \
  input_directory="$INPUT_DIR" \
  input_filename="$INPUT_FILENAME" \
  working_dir="$WORKING_DIR" \
  checkpoint="$CHECKPOINT" \
  dataset_name="$DATASET_NAME"
