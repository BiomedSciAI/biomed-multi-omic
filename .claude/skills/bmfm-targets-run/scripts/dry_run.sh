#!/usr/bin/env bash
# Level-1 validator: renders the fully-resolved Hydra config without running
# training. This catches most config authoring errors (missing fields, bad
# interpolations, wrong override operators) in under a second.
#
# Pass the exact same args you'd give `bmfm-targets-run`, e.g.:
#
#   bash scripts/dry_run.sh -cd run -cn finetune \
#       input_file=/tmp/data.h5ad \
#       working_dir=/tmp/run \
#       checkpoint=ibm-research/biomed.rna.bert.110m.wced.multitask.v1 \
#       label_column_name=celltype \
#       split_column_name=split_random
#
# Exits nonzero if resolution fails (errors go to stderr as normal).
#
# For a Level-2 smoke test (actually run a handful of steps on CPU), use:
#
#   bmfm-targets-run <your args> \
#     task.max_epochs=1 ++task.max_steps=5 \
#     data_module.num_workers=0 \
#     task.accelerator=cpu task.precision=32

set -euo pipefail

if ! command -v bmfm-targets-run >/dev/null 2>&1; then
  echo "ERROR: bmfm-targets-run not on PATH. Activate the venv first." >&2
  exit 127
fi

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <all the args you'd pass to bmfm-targets-run>" >&2
  echo "       e.g. $0 -cd run -cn predict input_file=... checkpoint=..." >&2
  exit 2
fi

exec bmfm-targets-run "$@" --cfg job --resolve
