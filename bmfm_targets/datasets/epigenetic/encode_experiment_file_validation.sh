#!/bin/bash

# This script validates that the experiment and file IDs in data_quadruplets.txt
# exist on the same line within a master experiment TSV file.

# --- Configuration ---
DATA_FILE="./encode_file_list_h1.tsv"
METADATA_TSV="./encode_h1_experiments_bw.tsv"

# Input validation
if [ ! -f "$METADATA_TSV" ]; then
    echo "Error: Metadata file not found at '$METADATA_TSV'"
    exit 1
fi

if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Data file not found at '$DATA_FILE'"
    exit 1
fi

echo "--- Starting Validation using '$METADATA_TSV' ---"
TOTAL_VALID=0
TOTAL_FAILED=0

while IFS=$'\t' read -r CELL_LINE MEASUREMENT EXPERIMENT_ID FILE_ID; do
    # Skip empty lines
    [ -z "$EXPERIMENT_ID" ] && continue

    # First try: match all three (measurement + experiment_id + file_id)
    if grep -F "$EXPERIMENT_ID" "$METADATA_TSV" | grep -F "$FILE_ID" | grep -F "$MEASUREMENT" > /dev/null; then
        echo "[SUCCESS] Found exact match for: $MEASUREMENT ($FILE_ID in $EXPERIMENT_ID)"
        ((TOTAL_VALID++))

    # Second try: match just experiment_id + file_id (measurement doesn't match exactly)
    elif grep -F "$MEASUREMENT" "$METADATA_TSV" | grep -F "$FILE_ID" > /dev/null; then
        echo "[PARTIAL SUCCESS] Found experiment/file match but measurement differs: $MEASUREMENT ($FILE_ID in $EXPERIMENT_ID)"

        # Extract and show the actual measurement name from the 4th column
        ACTUAL_MEASUREMENT=$(grep -F "$EXPERIMENT_ID" "$METADATA_TSV" | grep -F "$FILE_ID" | cut -f4,7 | head -1)
        echo "    Requested measurement: '$MEASUREMENT'"
        echo "    Actual measurement:    '$ACTUAL_MEASUREMENT'"
        ((TOTAL_VALID++))

    # Final case: experiment_id and file_id don't match at all
    else
        echo "[TOTAL FAILURE] Could not find experiment/file combination: $MEASUREMENT ($FILE_ID in $EXPERIMENT_ID)"

        # Show lines matching just experiment_id
        echo "--> Lines containing EXPERIMENT_ID '$EXPERIMENT_ID':"
        grep -F "$EXPERIMENT_ID" "$METADATA_TSV" || echo "    (No lines found)"
        echo ""

        # Show lines matching just file_id
        echo "--> Lines containing FILE_ID '$FILE_ID':"
        grep -F "$FILE_ID" "$METADATA_TSV" || echo "    (No lines found)"
        echo ""

        ((TOTAL_FAILED++))
    fi
done < <(tail -n +2 "$DATA_FILE")

echo "--- Validation Complete ---"
echo "Summary: $TOTAL_VALID Succeeded, $TOTAL_FAILED Failed."

exit $((TOTAL_FAILED > 0))
