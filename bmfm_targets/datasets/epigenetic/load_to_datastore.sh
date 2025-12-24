#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# ---- config ----
PY_SCRIPT="${HOME}/bmfm-targets/bmfm_targets/datasets/epigenetic/load_bigwig.py"
GTF="/dccstor/bmfm-targets/data/omics/epigenome/gencode.v38.annotation.gtf.gz"
DATASTORE="/dccstor/bmfm-targets/data/omics/epigenome/datastore.parquet"
PROMOTER_UP=1000
PROMOTER_DOWN=1000

# -----------------
# BLACKLIST/WHITELIST CONFIG: Use comma-separated lists.
# An empty variable means no restriction.

# FEATURE CONTROL:
# Example: To blacklist 'methylation' and 'h3k27ac': "methylation,h3k27ac"
FEATURE_BLACKLIST="methylation"
FEATURE_WHITELIST="" # If this is set, only features on this list are processed

# BIOSAMPLE CONTROL:
# Example: To blacklist 'K562' and 'HepG2': "K562,HepG2"
BIOSAMPLE_BLACKLIST="K562"
BIOSAMPLE_WHITELIST=""
# -----------------

bw_files=( /dccstor/bmfm-targets/data/omics/epigenome/*/*.bw )

if [ "${#bw_files[@]}" -eq 0 ]; then
  echo "ERROR: no .bw files found under /dccstor/bmfm-targets/data/omics/epigenome/*/*.bw"
  exit 1
fi

for bw in "${bw_files[@]}"; do
  # guard in case something odd slipped in
  [ -f "$bw" ] || { echo "Skipping non-file: $bw"; continue; }

  feature=$(basename "$(dirname "$bw")")            # parent dir -> feature name

  # --- FEATURE BLACKLIST CHECK ---
  # Check if $feature is in the comma-separated FEATURE_BLACKLIST.
  # The surrounding commas (e.g., ,methylation,) ensure an exact word match.
  if [ -n "$FEATURE_BLACKLIST" ] && [[ ",${FEATURE_BLACKLIST}," =~ ,${feature}, ]]; then
    echo "Skipping (BLACKLISTED feature): $feature ($bw)"
    continue
  fi

  # --- FEATURE WHITELIST CHECK ---
  # If a whitelist is provided, the feature MUST be on the list.
  if [ -n "$FEATURE_WHITELIST" ] && [[ ! ",${FEATURE_WHITELIST}," =~ ,${feature}, ]]; then
    echo "Skipping (NOT in WHITELIST feature): $feature ($bw)"
    continue
  fi

  filename=$(basename "$bw")
  stem="${filename%.*}"                              # remove extension
  biosample_name="${stem##*_}"                              # last underscore-delimited token

  # --- BIOSAMPLE BLACKLIST CHECK ---
  if [ -n "$BIOSAMPLE_BLACKLIST" ] && [[ ",${BIOSAMPLE_BLACKLIST}," =~ ,${biosample_name}, ]]; then
    echo "Skipping (BLACKLISTED biosample): $biosample_name ($bw)"; continue
  fi

  # --- BIOSAMPLE WHITELIST CHECK ---
  if [ -n "$BIOSAMPLE_WHITELIST" ] && [[ ! ",${BIOSAMPLE_WHITELIST}," =~ ,${biosample_name}, ]]; then
    echo "Skipping (NOT in WHITELIST biosample): $biosample_name ($bw)"; continue
  fi

  printf "Loading %s (%s) from %s\n" "$feature" "$biosample_name" "$bw"

  python "$PY_SCRIPT" \
    --gtf "$GTF" \
    --datastore "$DATASTORE" \
    --biosample_name "$biosample_name" \
    --promoter_up "$PROMOTER_UP" \
    --promoter_down "$PROMOTER_DOWN" \
    --bigwig "$bw" \
    --feature_name "$feature"
done
