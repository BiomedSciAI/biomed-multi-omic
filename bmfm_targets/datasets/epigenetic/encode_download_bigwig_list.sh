#!/bin/bash

# This script reads data_quadruplets.txt and downloads the specified files
# into an organized 'epigenome' directory.

# --- Configuration ---
DATA_FILE=$1
BASE_DIR="/dccstor/bmfm-targets/data/omics/epigenome/"

# --- Validation ---
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: $DATA_FILE not found!"
    exit 1
fi

# --- Download Loop ---
echo "--- Starting Download Process ---"

# Read the data file line by line, skipping the header
tail -n +2 "$DATA_FILE" | while IFS=$'\t' read -r CELL_LINE MEASUREMENT EXPERIMENT_ID FILE_ID; do

    # Skip empty lines
    [ -z "$FILE_ID" ] && continue

    # Determine the subdirectory by removing suffixes like _plus or _minus
    # e.g., "methylation_plus" becomes "methylation"
    SUB_DIR=$(echo "$MEASUREMENT" | sed 's/_[^_]*$//')

    # Construct the full path and URL
    DEST_DIR="$BASE_DIR/$SUB_DIR"
    OUTPUT_FILE="$DEST_DIR/${MEASUREMENT}_${CELL_LINE}.bw"
    URL="https://www.encodeproject.org/files/${FILE_ID}/@@download/${FILE_ID}.bigWig"

    # Create the directory if it doesn't exist
    mkdir -p "$DEST_DIR"

    # Skip if file already exists
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Skipping $MEASUREMENT ($FILE_ID) - already exists"
        continue
    fi

    echo "Downloading $MEASUREMENT ($FILE_ID) to $OUTPUT_FILE"

    # Use wget with continue option in case of interruption
    wget -c -O "$OUTPUT_FILE" "$URL"

    # Check if wget was successful
    if [ $? -eq 0 ]; then
        echo "✓ Download successful."
    else
        echo "✗ Error downloading $FILE_ID. Please check the URL or your connection."
    fi
    echo "---"
done

echo "--- Download Script Finished ---"
