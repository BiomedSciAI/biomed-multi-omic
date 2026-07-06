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

---

# Pseudobulk supervised target + per-tissue metrics (held-out-tissue generalization)

This section covers the follow-up work (`add option for pseudobulk target; generalize
gene perturbation metrics`) and answers the practical question: **if you want to train
on some tissues and generate realistic ChIP for held-out tissues, does this help, and
how do you use it correctly?**

## TL;DR — does it help generalization?

- **The pseudobulk target is a denoising of the supervised signal, not a new
  generalization mechanism.** It replaces the noisy per-cell MSE target (a single random
  same-tissue ChIP draw) with the deterministic per-tissue *mean* ChIP. Cleaner target →
  lower-variance gradients → a smoother learned map, which usually generalizes a little
  better. It does **not**, by itself, make a model that fails to transfer suddenly
  transfer.
- **It can make the degenerate "tissue-ID → stored mean" shortcut *more* attractive,** not
  less. Because every cell of a tissue now maps to one constant vector, a high-capacity
  model can minimize training loss by recognizing the tissue and emitting its memorized
  mean. That solution has **zero** held-out value (a never-seen tissue has no stored
  mean). With only a handful of training tissues this is the default failure mode.
- **Held-out generalization is governed by things this change does not touch:** (1) whether
  scRNA→ChIP is genuinely a *tissue-invariant per-gene* relationship in your data (e.g.
  active-mark ChIP tracking local expression), (2) how *many/diverse* training tissues you
  have (a new tissue must interpolate between seen ones), (3) gene-aligned features so the
  transferable expression↔mark coupling is even learnable, (4) regularization.
- **Net:** use the pseudobulk target as the clean target for the *mean* ChIP profile, but
  judge success only against the baselines in
  [Evaluating held-out tissues](#evaluating-held-out-tissues-correctly) — a bare `agg_pcc`
  will look high for trivial reasons.

## What "realistic ChIP samples" means decides which loss you need

| Goal | Right tool | Why |
|------|-----------|-----|
| **One representative profile per held-out tissue** (a pseudobulk prediction) | `chip_pairing_strategy: pseudobulk` MSE; OT optional | The conditional mean *is* the target; pseudobulk MSE estimates it directly and cleanly. |
| **A realistic *population* of single-sample ChIP profiles** (capturing biological spread) | `population_ot` is essential | MSE (random or pseudobulk) drives every cell toward the tissue mean → outputs collapse to a point → no sample diversity. Only the OT distribution-matching term preserves spread. |

> **Critical interaction:** pseudobulk MSE and `population_ot` pull in opposite directions.
> Pseudobulk MSE wants every prediction *at the mean* (a point mass); OT wants the predicted
> cloud to *match the spread* of real ChIP. If you want diverse samples, keep the OT weight
> meaningful and treat pseudobulk MSE as a centering/stabilizing regularizer, not the
> dominant term. If you only want the per-tissue mean profile, you can drop OT entirely and
> train pseudobulk MSE alone.

## Enabling the pseudobulk pairing target

Default is unchanged (`"random"`). Opt in via `dataset_kwargs`:

```yaml
data_module:
  dataset_kwargs:
    new_field: label_expressions
    split_column_name: tissue_split
    chip_pairing_strategy: pseudobulk   # default "random"
```

- `"random"`: each scRNA cell is paired with one **random** same-tissue ChIP sample (the
  original behavior; injects noise, acts as a crude augmentation).
- `"pseudobulk"`: each scRNA cell is paired with the **deterministic per-tissue mean** ChIP
  profile (`make_group_means(..., avg_row_label=None)` over the split's ChIP cells, keyed on
  `label_columns`, or `tissue_label` if none). Deterministic → reproducible targets.

Grouping key: a single `label_columns` entry is used directly; multiple columns are joined
with `"__"` into a composite key. Same key is rebuilt at lookup time, so it stays consistent.

## Enabling per-tissue val/test metrics

The aggregated pseudobulk metrics (`agg_pcc`, `agg_mae`) were previously hardcoded to gene
perturbations. They are now generic over a configurable group column.

```yaml
trainer:
  enable_perturbation_metrics: true      # GATE — defaults to False; metrics are OFF unless set
  perturbation_group_column: tissue_label
```

- `enable_perturbation_metrics: true` does two things: it selects
  `SequenceLabelingTrainingModule` as the training module (instead of
  `MultiTaskTrainingModule`) **and** un-gates the metric computation in
  `_shared_test_val_on_end`. Without it, no per-tissue metrics are produced.
- `perturbation_group_column: tissue_label` groups predictions by tissue and appends
  `tissue_label` to `sample_metadata_keys` so it is tracked into the predictions DataFrame.
- `discrimination_score` (gene-specific, decodes `A_B` combinatorial knockouts via
  `split("_")`) is **auto-skipped** when `perturbation_group_column != "perturbed_genes"`.
- The ChIP predictions DataFrame has no `input_expressions`/`baseline_expressions` columns,
  so baseline/delta metrics are skipped automatically; only `agg_pcc`/`agg_mae` are emitted.

## Evaluating held-out tissues correctly

The ground-truth pseudobulks (`group_means`) come from `data_module.get_dataset_instance()`.
**The metric only works if every tissue you predict on is also present as ChIP in that
`group_means` AnnData.** If a predicted tissue is missing, the whole metric call currently
raises `KeyError: 'label_expressions'` (see [Known gaps](#known-gaps--gotchas)).

**Recommended recipe — evaluate held-out generalization in a dedicated test run:**

1. **Split by tissue.** In `tissue_split`, assign each held-out tissue entirely to `test`
   (and/or `dev`). Training never sees held-out ChIP — the iterated/paired cells are
   split-restricted (and the pseudobulk *pairing target* uses only the current split's
   ChIP), so there is no leak.
2. **Keep held-out ChIP in the evaluation data.** The h5ad must contain, for each held-out
   tissue, both its **scRNA** cells (to generate predictions) and its **ChIP** cells (the
   ground-truth pseudobulk). The dataset now builds `group_means` over ChIP from **all
   splits**, so a held-out tissue present in the h5ad is scored even during training-time
   validation. If a held-out tissue has *no* ChIP at all, you can only *generate* — there is
   nothing to score against (and see the `KeyError` gotcha below).
3. **Run the evaluation** with the metrics gate on:
   ```yaml
   trainer:
     enable_perturbation_metrics: true
     perturbation_group_column: tissue_label
   ```
   The per-tissue `agg_pcc` / `agg_mae` for the held-out tissues are your generalization
   numbers, alongside the automatic baseline below.

**Do not trust a bare `agg_pcc`.** ChIP profiles across tissues share a large common
component (housekeeping / "which genes have any signal"), so `agg_pcc` over raw profiles is
usually high (0.8–0.95) even for a trivial predictor. The dataset now appends an
`Average_Perturbation_Train` row to `group_means` (the unweighted mean ChIP over training
tissues), so the metrics **automatically** report:

- `baseline_agg_pcc_from_avg_perturbation` / `baseline_agg_mae_from_avg_perturbation` — the
  *average-training-tissue baseline*. **The model is only useful where its `agg_pcc` beats
  this.** A scatter of `agg_pcc` vs the baseline is also logged.

One baseline still worth checking manually:

- **scRNA self-baseline:** if ChIP and scRNA are gene-aligned and ChIP tracks expression,
  correlate the held-out tissue's true ChIP against its own scRNA pseudobulk. The model must
  beat this to add value over "expression ≈ ChIP."

## Known gaps & gotchas

1. **`KeyError` only when a predicted tissue has no ChIP anywhere.** `group_means` now spans
   all splits, so held-out tissues present in the h5ad are scored fine. But if you predict on
   a tissue that has *no* ChIP cells in the data at all, `get_grouped_predictions` drops the
   `label_expressions` column and `log_perturbation_specific_metrics` then raises
   `KeyError: 'label_expressions'` (it does not skip ungrounded groups). For pure generation
   on a tissue with no reference ChIP, leave `enable_perturbation_metrics` off.
2. **`exp_before_mean` is hardcoded `False`.** The pseudobulk mean (pairing target, metric
   ground truth, and the average baseline row) is computed in linear space. If your ChIP `.X`
   is log1p-transformed, that mean is biased — keep ChIP in linear space (the shipped configs
   use `log_normalize_transform: false`). All three uses share the setting, so they stay
   mutually consistent.
3. **Tests are not committed.** `bmfm_targets/tests/test_datasets_utils.py` and
   `bmfm_targets/tests/test_scrna2chip_pairing.py` exist locally but are untracked, so CI
   does not run them yet. `git add` them to protect `make_group_means`, the pairing
   behavior, the all-split `group_means`, and the average baseline.

## Minimal config deltas

**Train (held-out split, pseudobulk MSE + OT, with monitoring):**
```yaml
data_module:
  dataset_kwargs:
    split_column_name: tissue_split        # held-out tissues -> test in this column
    chip_pairing_strategy: pseudobulk
trainer:
  enable_perturbation_metrics: true        # per-tissue agg_pcc + avg-baseline at val/test
  perturbation_group_column: tissue_label
  losses:
    - {field_name: label_expressions, name: mse, link_function: exp, wced_target: all_genes, weight: 1.0}
    - {field_name: label_expressions, name: population_ot, wced_target: all_genes,
       population_key: chip_population, eps: 1.0, n_iters: 100, cost: euclidean, link_function: null, weight: 1.0}
```

**Held-out evaluation (test run):** same `enable_perturbation_metrics` /
`perturbation_group_column`, pointed at an h5ad whose `test` split contains the held-out
tissues' scRNA **and** ChIP. Compare reported `agg_pcc` against the manual baselines above.
