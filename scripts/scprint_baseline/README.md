# scPRINT baseline

`embed_scprint.py` embeds cells with a frozen [scPRINT](https://github.com/cantinilab/scPRINT) checkpoint and writes the result to `obsm["scPRINT"]` of an h5ad. It is a standalone script that does not import `bmfm_targets`.

## Why a separate environment

scPRINT pins `torch==2.2.0`, `torchmetrics==1.6.0`, `numpy<2` and `python<3.13`. `bmfm_targets` needs `torchmetrics==1.1.0` and `transformers>=5`, so the two cannot share an environment. Run this script in its own one, then score `obsm["scPRINT"]` in the main environment like any other embedding (for example with `bmfm_targets.evaluation`).

## Setup

```bash
python -m venv scprint-env && source scprint-env/bin/activate
pip install scprint 'lamindb[bionty]'
lamin init --storage <dir> --schema bionty
hf download jkobject/scPRINT large-v1.ckpt --local-dir .
```

## Run

```bash
python embed_scprint.py data.h5ad --ckpt large-v1.ckpt                # human
python embed_scprint.py data.h5ad --ckpt large-v1.ckpt --organism NCBITaxon:10090   # mouse
```

`--output` writes to a new file instead of overwriting the input.

## Input checks

- Raw integer counts must be in `X` or a layer. Log-normalised input is rejected, because scPRINT conditions on sequencing depth.
- `var_names` may be gene symbols or Ensembl IDs (auto-detected).
- The organism is not inferred. Pass `--organism` for non-human data.
- Cells are not filtered (`min_nnz_genes=0`), so the metrics describe the same cells as the other methods.

## Gene-index alignment

scdataloader's `Collator` computes embedding rows from positions in the local bionty gene table. If that table differs from the checkpoint's gene list, every gene reads the wrong embedding row. Nothing raises, and the embedding looks normal. The script compares the table with the checkpoint, registers any retired Ensembl IDs it is missing, and refuses to embed if they still differ (`--no-align-ontology` raises instead of repairing).

This matters. With the 335,182-gene `ji9krimq` (scPRINT-2) checkpoint, the local table was 8 genes short. `avg_bio` was 0.310 on the 9 sceval datasets before the repair and 0.505 after.

## Defaults

`large-v1.ckpt`, `how="random expr"` and the `cell_type_ontology_term_id` class token, as in scPRINT's own `cell_emb.default_benchmark`. The smaller `medium-v1.5` checkpoint scored 0.093 lower on `avg_bio` on the same datasets.
