# Data preparation and active inspection

Before writing any config, look at what the user actually has. This file contains the snippets.

## Inspecting an h5ad

Always run this before picking a label column or confirming data format:

```python
import scanpy as sc
ad = sc.read_h5ad("<user path>")
print("shape:", ad.shape)
print("obs columns:")
print(ad.obs.dtypes)
print("\ncandidate label columns (categorical / small unique count):")
for col in ad.obs.columns:
    n = ad.obs[col].nunique()
    if n <= 100:
        print(f"  {col}: {n} unique  (dtype={ad.obs[col].dtype})")
print("\nis there a split column?")
split_cols = [c for c in ad.obs.columns if c.startswith("split")]
print(" ", split_cols or "no; will need to create one")
print("\nare counts raw or normalized?  max(X)=", ad.X.max(), "; any decimals?", (ad.X.data % 1 != 0).any() if hasattr(ad.X, "data") else "(dense)")
```

What to infer from the output:

- **Label column**: small unique count + categorical dtype = classification; float dtype = regression.
- **Split column**: only needed for **training / finetuning**. `predict` runs over the full h5ad — don't invent a split column for it. If training and a column is present, use its name in `split_column_name=`. If training and absent, create one (see below). Values must be the literal strings `"train"`, `"dev"`, `"test"`.
- **Raw vs. normalized**: the framework ships a helper — `from bmfm_targets.datasets.datasets_utils import guess_if_raw; guess_if_raw(ad.X.data)` returns True/False. Use it instead of eyeballing. If raw → most checkpoints want `data_module.log_normalize_transform=true`; the v1 RDA checkpoint wants `log_normalize_transform=false` (RDA normalizes internally). If already log-normalized, set `log_normalize_transform=false`.
- **Gene identifiers**: `ad.var_names[:5]` should be gene symbols (e.g. `ENSG00000141510`, `TP53`). If they're Ensembl ids and the tokenizer expects symbols, the package has a mapping — see `bmfm_targets/transforms/protein_coding_gene_mapping_*.json`.

## Creating a split column

The split column values must be the literal strings `"train"`, `"dev"`, `"test"` — not `"val"`, `"validation"`, `"eval"`, or integer codes. The framework does a string-equality lookup against those three. Rows with other values are dropped silently.

**Quick random split** (80/10/10):

```python
import scanpy as sc
import bmfm_targets.datasets.datasets_utils as du

ad = sc.read_h5ad("<user path>")
ad.obs["split_random"] = du.get_random_split(
    ad.obs, {"train": 0.8, "dev": 0.1, "test": 0.1}, random_state=42
)
ad.write_h5ad("<user path>")
```

This is the snippet from the repo README. It writes back in-place; make sure the user is OK with that, or copy the file first.

**Stratified split** (preserves class balance across train/dev/test — better for imbalanced labels):

Use `DatasetTransformer` with a `stratifying_label`:

```python
from bmfm_targets.datasets.dataset_transformer import DatasetTransformer

dt = DatasetTransformer(
    source_h5ad_file_name="<user path>",
    split_weights={"train": 0.8, "dev": 0.1, "test": 0.1},
    stratifying_label="celltype",  # or whichever column you want to stratify on
    transforms=[],  # empty list = no gene transforms, just split creation
    random_state=42,
)
processed = dt.process_datasets()
processed.write_h5ad("<user path>")
# Split column will be named split_stratified_celltype
```

Set `split_column_name=split_stratified_<label>` in the config.

**User-named split**: if the user has their own split column (named anything), pass it via `split_column_name=<their_name>` at the CLI. No code needed.

## Split strategy: prefer held-out-batch over random whenever possible

Large pretrained RNA encoders absorb a lot of donor / batch / assay structure during pretraining. A random split at finetune time puts cells from the same donor in both train and eval — the model then "generalizes" by recognizing donor-specific tokens it has already seen, and the reported metric is not what the user thinks it is. This is most visible on CellXGene-derived data.

Guidance when proposing a split:

- If the user's h5ad has a grouping column — `donor_id`, `dataset_id`, `batch`, `assay`, `sample_id` — with **≥ 4 distinct values**, steer them towards a **held-out-group split**: hold one or more groups entirely out of train. Use `DatasetTransformer` with a group-aware strategy or write the column manually (pick ~80% of groups for train, split the rest into dev/test).
- If the grouping column has only **2–3 values**, honestly tell the user the data is too narrow for a grouped split — one group in test means the test set is a single donor and the result is anecdotal. In that case random stratified is as good as it gets, but call out the limitation.
- Only fall back to `split_random` when there's no grouping column at all, and say so in the summary you hand to the user.

## Gotcha: label-name clash with a pretrained decoder

A clash happens **if and only if** the user's `label_column_name` is identical to one already in the checkpoint's `label_columns`. When names match, state-dict loading pushes the pretrained head weights into the user's fresh head — if `n_unique_values` happens to match it loads silently with the wrong label-index semantics, and if it doesn't it fails with a shape mismatch. When the names differ, a fresh head is added and you don't need to do anything.

Per-checkpoint head names (always re-check the provenance yaml; this is a snapshot):

- **v1 BERT WCED multitask** (`biomed.rna.bert.110m.wced.multitask.v1`): `cell_type`, `tissue`, `tissue_general`, `donor_id`
- **v2 Llama 32m MLM multitask**: `cell_type_ontology_term_id`, `tissue`, `tissue_general`, `donor_id`
- **v2 Llama 47m WCED multitask**: `cell_type_ontology_term_id`, `tissue`, `tissue_general`, `donor_id`, `sex`
- v1 MLM-RDA, v1 WCED (non-multitask), v1 MLM multitask have no label heads or different sets — read the yaml.

If the user's column name matches one of those, rename it (`cell_type` → `my_cell_type`, `donor_id` → `my_donor_id`, …) in `ad.obs` before training, and point `label_column_name` at the new name. If it doesn't match, proceed — there is nothing to fix.

There is no "close enough" — names either match exactly or they don't.

## Gotcha: `expose_zeros: true` requires a zero-padding strategy

`expose_zeros: true` on `dataset_kwargs` tells the data module to include non-expressed genes as zero-valued tokens. Pretraining the v2 checkpoints relies on this. But if you set it without also setting `pad_zero_expression_strategy`, every cell's input gets swamped by tens of thousands of zero tokens and the model learns to predict "zero" for everything.

Always pair it with a strategy. The v2 checkpoints use `batch_wise` with `interleave_zero_ratio: 0.9` and `expressed_sample_ratio: 0.2`:

```yaml
data_module:
  dataset_kwargs:
    expose_zeros: true
    pad_zero_expression_strategy: batch_wise
    interleave_zero_ratio: 0.9
    expressed_sample_ratio: 0.2
```

If you're unsure what the user needs, read the v2 checkpoint yaml and copy its values — those have been tuned. Do NOT set `expose_zeros: true` alone in any config you hand the user.

## Inspecting a DNA csv directory

For DNA finetuning the user needs `<dir>/{train,test,dev}.csv`. Check:

```bash
ls <dir>
head -3 <dir>/train.csv
wc -l <dir>/*.csv
```

Then in Python:

```python
import pandas as pd
df = pd.read_csv("<dir>/train.csv")
print(df.head())
print(df.dtypes)
print("num rows:", len(df))
print("label columns (all but first, which is the sequence):")
for c in df.columns[1:]:
    if df[c].dtype.kind in "fi":
        kind = "regression" if df[c].dtype.kind == "f" else "classification"
    else:
        kind = "classification"
    print(f"  {c}: {kind} ({df[c].nunique()} unique, dtype={df[c].dtype})")
```

Rules for the csv format:

- **Column 0 MUST be the sequence** (A/C/G/T string). The framework reads it positionally.
- **Label columns follow**, named however the user wants — they get referenced by name in `label_columns` in the yaml.
- Trailing columns (e.g. `seq_id`) are fine — they just get ignored.
- All three files (`train.csv`, `test.csv`, `dev.csv`) must have the same columns.

## Gene identifier mismatch

If the h5ad has Ensembl ids (`ENSG...`) but the tokenizer is gene-symbol-based (the default for scRNA), the framework applies a mapping transform — it's already enabled by default in most data_modules via `RenameGenesTransform` (see `bmfm_targets/datasets/dataset_transformer.py`). If the user has custom identifiers (e.g. organism other than human), they'll need to provide their own gene map; flag this early rather than silently producing low-quality embeddings.

## Labels file (`labels.json`)

For `predict` and `finetune` the framework writes `<working_dir>/labels.json` listing the label set seen during training. On a second run pointing at the same `working_dir`, it reuses this file — useful for consistent label ordering across runs, but a source of confusion when the user changes label columns and forgets to clear the working_dir. When in doubt, delete `labels.json` or pick a fresh `working_dir`.

## Working directory guidance

- If the user provides `working_dir`, use it.
- Otherwise default to `./outputs/<task_class>_<timestamp>/` in the current repo.
- Avoid `/tmp` for anything the user will want to inspect later — macOS clears `/tmp` on reboot.
- The directory will be created if it doesn't exist; existing contents (especially `labels.json`) get reused, which is both a feature and a foot-gun.
