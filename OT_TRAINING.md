# scRNA → ChIP Population-OT Training

Population-level optimal transport (OT) loss for scRNA→ChIP translation.
Instead of per-cell MSE, the model minimises a **debiased Sinkhorn divergence**
between the predicted ChIP population for a celltype (the batch of decoded
scRNA cells) and the true ChIP population for that celltype (all ChIP-seq
samples for that tissue in the h5ad).

Population OT is a **standard composed loss** (`name: population_ot`): it is just
another entry in the trainer's `losses:` list, summed by `calculate_losses`
alongside any WCED reconstruction loss. There is no bespoke training module or
`ot_*` trainer flags — the standard `MultiTaskTrainingModule` drives everything.

## How it works

1. **Homogeneous batching** — each mini-batch contains scRNA cells from a
   single tissue/celltype, sampled with replacement via
   `ConditionHomogeneousBatchSampler` (enabled by `data_module.use_ot_batching`).

2. **ChIP population injection** — at `setup()` time, `scRNA2ChIPDataModule`
   groups all ChIP cells by `celltype_column` and stacks them into a
   `chip_populations` dict (`celltype → FloatTensor [M, n_genes]`).  The
   `_ChipCollator` then injects the matching population into every batch at
   `batch["labels"]["chip_population"]` (so `calculate_losses`, which only
   forwards `labels`, can reach it).

3. **OT loss** — the `population_ot` loss task pairs a `WCEDPopulationSource`
   (extracts the predicted `[B, V]` cloud from the WCED decode token and the
   `[M, V]` reference population from `labels["chip_population"]`) with a
   `PopulationOTObjective` (debiased Sinkhorn divergence). With a WCED `mse`
   loss entry too, the total is the weighted mean of the two loss tasks:
   ```
   loss = (  weight_mse * mse_reconstruction
           + weight_ot  * SinkhornDivergence(pred [B, V], chip_population [M, V])
          ) / (weight_mse + weight_ot)
   ```
   Note: `calculate_losses` combines the entries as a **weighted mean** (it
   divides by the sum of weights), not a raw weighted sum — so with the default
   equal weights the total is `(mse + ot) / 2`. When a loss is skipped on a step
   (e.g. OT returns `None` because `B < 2` or `M < 2`), its weight drops out of
   the denominator for that step.
   Both clouds live in the same gene space (gene-aligned by the dataset).
   `SD(X,Y) = OT(X,Y) − ½OT(X,X) − ½OT(Y,Y)`; `OT(Y,Y)` is cached per target
   population (device-stable content key) since it is constant across steps.
   The objective returns `None` (loss skipped) when `B < 2` or `M < 2`.

`population_ot` is a batch-level loss with no per-sample alignment, so its
objective declares `contributes_sample_metrics == False`: it is excluded from
the per-sample metric machinery and never collides with the WCED MSE metrics.

## Input h5ad format

The h5ad must contain both scRNA and ChIP cells together:

| obs column     | values            | notes                          |
|----------------|-------------------|--------------------------------|
| `data_type`    | `"scRNA"`, `"ChIP"` | used to split scRNA vs ChIP  |
| `tissue_label` | e.g. `"liver"`    | or whatever `celltype_column` points to |
| `tissue_split` | `"train"`, `"dev"`, `"test"` | split column        |

Genes (`var_names`) must be identical for scRNA and ChIP rows (gene-aligned).

## OT-only vs OT + WCED multitask

Both modes are selected purely by which entries appear in the trainer's
`losses:` list — there are no separate mode flags.

**OT only** (no per-gene reconstruction supervision) — a single `population_ot`
loss entry:
```yaml
trainer:
  losses:
    - field_name: label_expressions
      name: population_ot
      wced_target: all_genes
      population_key: chip_population
      eps: 1.0
      n_iters: 100
      cost: euclidean
      link_function: null
      weight: 1.0
```
No `sequence_label_extractor` is needed for OT-only training.

**OT + WCED multitask** (recommended — WCED reconstruction regularises the
decoder while OT drives population-level translation) — add an `mse` entry and
wire `WCEDMasker` so per-gene labels are present:

`WCEDMasker` scatters per-gene expression values into the full tokenizer vocab
space and exposes `labels["label_expressions"] = {"all", "input", "non_input"}`
tensors `[B, vocab_size]`. Pass it as `sequence_label_extractor` when
constructing the DataModule; the `_ChipCollator` injects
`labels["chip_population"]` `[M, vocab_size]` after the masker runs, so every
batch carries both.

```yaml
trainer:
  losses:
    - field_name: label_expressions
      name: mse
      ignore_zero: true
      link_function: exp
      wced_target: all_genes
      weight: 1.0
    - field_name: label_expressions
      name: population_ot
      wced_target: all_genes
      population_key: chip_population
      eps: 1.0
      n_iters: 100
      cost: euclidean
      link_function: null
      weight: 1.0
```

Total loss = `(weight_mse * mse + weight_ot * sinkhorn_divergence) / (weight_mse + weight_ot)`
(a weighted mean — see the note above).
Start with equal weights; increase the `population_ot` `weight:` if the model
memorises individual cells without population-level alignment.

## Wiring into a YAML config

See the shipped end-to-end examples:

- `run/scrna2chip_ot_train.yaml` — OT-only.
- `run/scrna2chip_ot_wced_train.yaml` — OT + WCED multitask.

The data-module side enables homogeneous batching and population grouping:

```yaml
data_module:
  use_ot_batching: true
  celltype_column: tissue_label   # obs column to group ChIP population by
  batch_size: 16
  collation_strategy: sequence_labeling
  sequence_label_extractor:       # only for OT+WCED — enables WCED label dict
    _target_: bmfm_targets.training.masking.strategy.WCEDMasker
    _partial_: true
    value_field_name: label_expressions
  ...
```

## Scale reconciliation

The `population_ot` `link_function` is optional and defaults to identity
(`null`). Reconciling the scale between predicted and ChIP space is
preprocessing's responsibility — do not rely on a link function to fix it.

## Key files

| File | Purpose |
|------|---------|
| `bmfm_targets/training/losses/objectives.py` | `PopulationOTObjective` — debiased Sinkhorn divergence + OT(Y,Y) cache |
| `bmfm_targets/training/losses/sources.py` | `WCEDPopulationSource` — extracts `[B, V]` pred + `[M, V]` population |
| `bmfm_targets/training/losses/compat.py` | parses `population_ot` loss dicts into `LossTask`s |
| `bmfm_targets/training/losses/ot/sinkhorn.py` | `sinkhorn_cost` (log-domain, fp32-safe) |
| `bmfm_targets/datasets/samplers.py` | `ConditionHomogeneousBatchSampler` |
| `bmfm_targets/training/data_module.py` | `scRNA2ChIPDataModule` + `_ChipCollator` (`use_ot_batching`) |

## Tests

| Test file | Covers |
|-----------|--------|
| `bmfm_targets/tests/test_population_ot_loss.py` | objective + source + `loss_dict_to_task` (Unit A) |
| `bmfm_targets/tests/test_chip_collator.py` | collator / data-module population plumbing (Unit B) |
| `bmfm_targets/tests/test_unit_c.py` | module unification, metric exclusion, OT(Y,Y) cache, integration (Unit C) |

```bash
.venv/bin/python -m pytest -o log_cli=false \
  bmfm_targets/tests/test_population_ot_loss.py \
  bmfm_targets/tests/test_chip_collator.py \
  bmfm_targets/tests/test_unit_c.py
```

## Per-loss hyperparameters (`population_ot` entry)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `weight` | 1.0 | Relative weight of the OT term in the weighted-mean loss |
| `eps` | 1.0 | Sinkhorn entropic regularisation |
| `n_iters` | 100 | Sinkhorn iterations |
| `cost` | `"euclidean"` | `"euclidean"` or `"sqeuclidean"` |
| `link_function` | `null` | Optional transform on `pred` before the cost (e.g. `"exp"`); default identity |
| `population_key` | `"chip_population"` | key under `batch["labels"]` holding the `[M, V]` population |
| `celltype_column` | `"tissue_label"` | (data_module) obs column for celltype grouping |
