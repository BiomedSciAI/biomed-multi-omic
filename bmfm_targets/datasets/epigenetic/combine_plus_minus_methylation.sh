#!/usr/bin/env bash

# This script combines the plus and minus methylation bigwig files to a single bigwig file
# It requires the python packages `bigWigToBedGraph`, `bedtools` and `bedGraphToBigWig` which can be installed with conda

set -euo pipefail

CELL_LINE=$1
PLUS="/dccstor/bmfm-targets/data/omics/epigenome/methylation/methylation_plus_$CELL_LINE.bw"
MINUS="/dccstor/bmfm-targets/data/omics/epigenome/methylation/methylation_minus_$CELL_LINE.bw"
OUT="/dccstor/bmfm-targets/data/omics/epigenome/methylation/methylation_$CELL_LINE.bw"
CHROMSIZES="/dccstor/bmfm-targets/data/omics/epigenome/hg38.chrom.sizes"

echo "[1/4] Converting bigWigs to bedGraphs..."
bigWigToBedGraph "$PLUS" plus.bedGraph
bigWigToBedGraph "$MINUS" minus.bedGraph

echo "[2/4] Shifting minus strand by -1 bp and sorting..."
bedtools shift -i minus.bedGraph -g "$CHROMSIZES" -s -1 > minus.shifted.bedGraph
cat plus.bedGraph minus.shifted.bedGraph \
  | LC_ALL=C sort -k1,1 -k2,2n > combined.sorted.bedGraph

echo "[3/4] Merging CpG sites with mean score..."
bedtools merge -i combined.sorted.bedGraph -c 4 -o mean > merged.bedGraph

echo "[4/4] Writing final bigWig..."
echo "[4/4] Writing final bigWig..."
grep -v "^chrEBV" merged.bedGraph > merged.filtered.bedGraph
bedGraphToBigWig merged.filtered.bedGraph "$CHROMSIZES" "$OUT"

rm plus.bedGraph minus.bedGraph minus.shifted.bedGraph combined.sorted.bedGraph merged.bedGraph merged.filtered.bedGraph
echo "✅ Done: $OUT"
