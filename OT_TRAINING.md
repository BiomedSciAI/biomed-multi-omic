# scRNA → ChIP Population-OT Training

Population-level optimal transport (OT) loss for scRNA→ChIP translation.
Instead of per-cell MSE, the model minimises a **debiased Sinkhorn divergence**
between the predicted ChIP population for a celltype (the batch of decoded
scRNA cells) and the true ChIP population for that celltype (all ChIP-seq
samples for that tissue in the h5ad).

## How it works

1. **Homogeneous batching** — each mini-batch contains scRNA cells from a
   single tissue/celltype, sampled with replacement via
   `ConditionHomogeneousBatchSampler`.

2. **ChIP population injection** — at `setup()` time, `scRNA2ChIPDataModule`
   groups all ChIP cells by `celltype_column` and stacks them into a
   `chip_populations` dict (`celltype → FloatTensor [M, n_genes]`).  Each
   batch gets the matching `chip_population` tensor injected as
   `batch["chip_population"]`.

3. **OT loss** — `ScrnaToChipTranslationModule.training_step` runs the
   standard WCED reconstruction losses, then adds:
   ```
   loss = wced_weight * wced_loss
        + ot_weight  * SinkhornDivergence(pred [B, seq_len],
                                          chip_population [M, seq_len])
   ```
   Both the predicted and true populations live in the same gene space
   (gene-aligned by the dataset).

## Input h5ad format

The h5ad must contain both scRNA and ChIP cells together:

| obs column     | values            | notes                          |
|----------------|-------------------|--------------------------------|
| `data_type`    | `"scRNA"`, `"ChIP"` | used to split scRNA vs ChIP  |
| `tissue_label` | e.g. `"liver"`    | or whatever `celltype_column` points to |
| `split_random` | `"train"`, `"dev"`, `"test"` | split column        |

Genes (`var_names`) must be identical for scRNA and ChIP rows (gene-aligned).

## Running the hello-world

```bash
PYTHONPATH=/u/dmichael/scRNA2ChIP python run/scrna2chip_ot_hello_world.py
```

(The conda env is an editable install of `bmfm-targets`; prefix with `PYTHONPATH`
to shadow it with this repo's version of `bmfm_targets`.)

This builds a tiny synthetic h5ad, runs one training step, and prints the
loss. Requires no GPU and no real data.

## OT-only vs OT + WCED multitask

Both modes are supported — controlled by the `losses` list in the trainer section
and the `wced_weight` / `ot_weight` knobs.

**OT only** (no per-gene reconstruction supervision):
```yaml
trainer:
  losses: []          # no WCED loss tasks
  extra_kwargs:
    ot_weight: 1.0
    wced_weight: 0.0  # or just omit — wced_weight * 0 = 0
```

**OT + WCED multitask** (recommended — WCED reconstruction regularises the
decoder while OT drives population-level translation):

`WCEDMasker` is the key: it scatters per-gene expression values into the full
tokenizer vocab space and splits them into `"input"` (genes that appear in the
input sequence) and `"non_input"` / `"all"` subsets.  Pass it as
`sequence_label_extractor` when constructing the DataModule; it is stored as
`self.masker` and called by `MultiFieldCollator`, so every batch contains both
`batch["labels"]["label_expressions"]["all"]` (shape `[B, vocab_size]`) and
`batch["chip_population"]` (shape `[M, vocab_size]`).

```yaml
trainer:
  losses:
    - field_name: label_expressions
      name: mse
      wced_target: all_genes
  extra_kwargs:
    ot_weight: 1.0    # relative weight of Sinkhorn divergence term
    wced_weight: 1.0  # relative weight of per-gene MSE reconstruction term
```

Total loss = `wced_weight * wced_loss + ot_weight * sinkhorn_divergence`.
Start with equal weights; increase `ot_weight` if the model memorises
individual cells without population-level alignment.

## Wiring into a YAML config

Full example with both losses:

```yaml
# trainer section
trainer:
  learning_rate: 1.0e-5
  training_module_class: bmfm_targets.training.modules.scrna_to_chip.ScrnaToChipTranslationModule
  losses:
    - field_name: label_expressions
      name: mse
      wced_target: all_genes
  extra_kwargs:
    ot_weight: 1.0
    wced_weight: 1.0
    ot_eps: 1.0
    ot_n_iters: 100

# data_module section
data_module:
  use_ot_batching: true
  celltype_column: tissue_label   # obs column to group ChIP population by
  batch_size: 16
  sequence_label_extractor:       # enables WCED label dict in every batch
    _target_: bmfm_targets.training.masking.WCEDMasker
    lookup_field_name: genes
    value_field_name: label_expressions
  ...
```

## Key files

| File | Purpose |
|------|---------|
| `bmfm_targets/training/modules/scrna_to_chip.py` | `ScrnaToChipTranslationModule` — OT training_step |
| `bmfm_targets/training/losses/ot/sinkhorn.py` | `sinkhorn_divergence` (log-domain, fp32-safe) |
| `bmfm_targets/training/losses/ot/monge_gap.py` | Monge gap (available but not used by default) |
| `bmfm_targets/datasets/samplers.py` | `ConditionHomogeneousBatchSampler` |
| `bmfm_targets/training/data_module.py` | `scRNA2ChIPDataModule` with `use_ot_batching` flag |
| `run/scrna2chip_ot_hello_world.py` | Synthetic end-to-end smoke test |

## Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `ot_weight` | 1.0 | Weight of OT loss term |
| `wced_weight` | 1.0 | Weight of WCED reconstruction term |
| `ot_eps` | 1.0 | Sinkhorn entropic regularisation |
| `ot_n_iters` | 100 | Sinkhorn iterations |
| `cost` | `"euclidean"` | `"euclidean"` or `"sqeuclidean"` |
| `celltype_column` | `"tissue_label"` | obs column for celltype grouping |
