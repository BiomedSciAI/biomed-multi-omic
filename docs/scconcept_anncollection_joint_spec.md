# Spec: Joint WCED + contrastive training (no streaming required)

Goal: a single training run that applies **WCED masking and contrastive paired-view learning
together**, over multi-h5ad pretraining data. Streaming is **not** required — the contrastive
data fits the memory budget, so we take the simplest path: reuse the existing (map-style)
`ContrastiveDataModule` unchanged and feed it multi-h5ad data.

> **Base branch:** verified against `origin/scconcept-from-main` (commit `d4eb2d5`), **not**
> `main`. The contrastive components (`ContrastiveDataModule`, `_PairedViewCollator`,
> `ConditionHomogeneousBatchSampler`, `ContrastiveObjective`, `ContrastiveMetricsCallback`) do not
> exist on `main`. Apply this from a checkout of that branch.

---

## What already works today

`run/scconcept_pretrain.yaml` already runs contrastive pretraining end-to-end on a **single**
processed h5ad via `ContrastiveDataModule`. That path gives us, for free:

- **Map-style dataset with obs.** `ContrastiveDataModule` (base `DataModule`,
  `DATASET_FACTORY = BaseRNAExpressionDataset`) exposes `dataset.h5ad.obs`, which
  `train_dataloader` reads to build the per-cell condition `pd.Series` (`data_module.py:1346`).
- **Condition-homogeneous sampling.** `ConditionHomogeneousBatchSampler` is passed as
  `batch_sampler` to an ordinary `DataLoader` — samples a condition uniformly, then `batch_size`
  cells from it. No streaming constraints apply here.
- **Paired-view collation.** `_PairedViewCollator` wraps `self.collate_fn`, splitting each cell's
  gene panel into two views and emitting `2N` instances in block layout
  `[A_0..A_{N-1}, B_0..B_{N-1}]`.
- **WCED composes with the pairing.** WCED masking is enabled by passing a `WCEDMasker` as the
  DataModule's `masking_strategy` (`data_module.py:258-259`); it runs *inside*
  `MultiFieldCollator.__call__` on the tokenized batch. `_PairedViewCollator` wraps that same
  WCED-enabled `collate_fn`, so masking sees the already-split panels and correctly separates
  input vs. non-input genes. No changes to any collator or masker are needed.
- **Losses coexist.** `ContrastiveObjective` (`objectives.py:622`) reads the `_contrastive`
  decoder suffix; WCED objectives read `_wced`. Both heads live on the same model. `both_batch:
  true` matches the collator's block layout (2N×2N combined form).

So the only real gaps are: **(1)** getting multi-h5ad data into that map-style DataModule, and
**(2)** turning on WCED alongside contrastive in the config. Both are small.

---

## Change 1: feed multi-h5ad data to `ContrastiveDataModule`

`ContrastiveDataModule` today points `dataset_kwargs.processed_data_source` at one h5ad. Two ways
to give it more, in order of preference:

### Recommended: concatenate to a single processed h5ad (no code)

Pre-concatenate the source h5ads once (offline), aligning genes on the intersection, and point
`processed_data_source` at the result:

```python
import anndata as ad, scanpy as sc
adatas = [sc.read(p) for p in source_paths]
combined = ad.concat(adatas, join="inner", label="dataset")   # inner join on genes
combined.write_h5ad("<processed>/scconcept_combined.h5ad")
```

Requirements: the condition column (e.g. `celltype`) must be present in `obs`, and `join="inner"`
harmonizes the gene var space (the same thing `AnnCollection(join_vars="inner")` did). Cost: one
combined h5ad on disk. This needs **zero new code** — only the yaml `processed_data_source` and,
if the run does its own processing, `transform_datasets`.

### Fallback: map-style AnnCollection-backed dataset (only if concatenation is undesirable)

If you must keep the h5ads separate (disk pressure, frequent additions), add a small map-style
`Dataset` that wraps `get_ann_collection(paths, join_obs="outer")` and:

- returns `MultiFieldInstance` for `dataset[i]` via `ann_collection[i]` (random access; backed),
- exposes an `.h5ad` shim whose `.obs` is `ann_collection.obs` (the one attribute
  `ContrastiveDataModule.train_dataloader` touches), so the DataModule consumes it **unchanged**.

This is ~50–80 lines and reuses the *existing* `ConditionHomogeneousBatchSampler` and
`_PairedViewCollator` with their exact semantics (uniform-over-conditions sampling, no
mixed-condition batches, fresh shuffle each epoch). Prefer concatenation unless you have a
concrete reason not to.

---

## Change 2: turn on WCED alongside contrastive in the config

Extend the working `scconcept_pretrain.yaml`. Three additions:

**a. WCED decode_mode on the `expressions` field** (keep the existing `contrastive` mode):

```yaml
- _target_: bmfm_targets.config.FieldInfo
  field_name: expressions
  tokenization_strategy: continuous_value_encoder
  encoder_kwargs:
    kind: mlp_with_special_token_embedding
    zero_as_special_token: true
  decode_modes:
    contrastive:
      projection_dim: 128
      learnable_scale: true
      decode_from: cls
    wced:
      vocab_field: genes
      logit_outputs: [mse, is_zero_bce]
```

> When resuming from a checkpoint that already carries `wced`,
> `main_config._merge_fields` (`main_config.py:182-212`) merges any *new* YAML decode_modes onto
> the checkpoint field. For a fresh field (as above) just declare both modes directly.

**b. A `WCEDMasker` as the DataModule `masking_strategy`** — this is what actually activates WCED
masking in the collator (`data_module.py:258`). Keep `mlm: false`:

```yaml
data_module:
  _target_: bmfm_targets.training.data_module.ContrastiveDataModule
  # ... existing keys ...
  mlm: false
  masking_strategy:
    _target_: bmfm_targets.training.masking.strategy.WCEDMasker
    # ... WCEDMasker args as used by the existing WCED multitask runs ...
```

**c. Both loss families in `trainer.losses`:**

```yaml
trainer:
  losses:
    - field_name: expressions
      name: mse
      wced_target: all_genes
      weight: 1.0
    - field_name: expressions
      name: is_zero_bce
      wced_target: all_genes
      weight: 1.0
    - field_name: expressions
      name: contrastive
      both_batch: true
      weight: 0.1   # tune; contrastive loss is on a different scale from WCED
```

---

## Notes / things to check

- **Effective batch is 2N.** `_PairedViewCollator` doubles `n_cells_per_batch` cells to `2N`
  instances. Size learning rate / memory for `2N`, not `N`.
- **Copy `WCEDMasker` args from a known-good WCED run.** The exact `WCEDMasker` constructor args
  (label sets, mask fractions) should mirror the existing WCED multitask config rather than being
  guessed here — that is the one detail worth verifying against a working WCED yaml before running.
- **Gene-space alignment is the real cost of concatenation.** An inner join drops genes absent
  from any source h5ad. If sources have very different panels, check how many genes survive before
  committing to the combined file.

---

## Summary

| Component | Status | Work needed |
|---|---|---|
| Contrastive on one h5ad (`ContrastiveDataModule`) | **works today** | none |
| `_PairedViewCollator` + WCED masking composing | **works today** | none |
| WCED + contrastive losses coexisting | **works today** | none |
| `contrastive` / `wced` decode_modes | **works today** | declare in yaml |
| Multi-h5ad input | **missing** | concatenate to one h5ad (no code); or map-style AnnCollection dataset (~50–80 lines) |
| Joint WCED + contrastive yaml | **missing** | ~15 lines (decode_mode + masker + losses) |

**Minimal path:** concatenate the source h5ads to one processed h5ad, point
`scconcept_pretrain.yaml` at it, add the WCED decode_mode + `WCEDMasker` + WCED losses. No new
code.

---

## Out of scope: streaming

An earlier draft targeted AnnCollection **streaming** (litdata `StreamingDataset` /
`StreamingDataLoader`), which forced condition-homogeneous batching to fight litdata's iteration
order (`batch_sampler` is ignored on `IterableDataset`; ordered indices only survive with
`shuffle=False`, which freezes epoch order). Since streaming is not required, that entire line of
work — a `ContrastiveStreamingDataModule`, `get_obs_series`, and condition-grouped/condition-chunk
index builders — is dropped. Revisit only if the contrastive dataset outgrows the memory budget;
at that point the streaming-native design is **condition-homogeneous litdata chunks +
`shuffle=True`** (which keeps per-epoch reshuffling), *not* a pre-sorted `shuffle=False` index.
