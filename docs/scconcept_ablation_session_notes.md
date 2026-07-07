# scConcept Ablation — Session Notes

## Goal

Run three ablations from scratch (no fine-tuning, no cell type supervision) on the 1% cellxgene h5ad dump to determine which loss component drives representation quality:

| Ablation | Config | Loss |
|---|---|---|
| WCED only | `scconcept_ablation_wced.yaml` | MSE + is_zero_bce on all genes |
| Contrastive only | `scconcept_ablation_contrastive.yaml` | InfoNCE panel-split |
| Both | `scconcept_ablation_both.yaml` | WCED + InfoNCE simultaneously |

Data: `/proj/bmfm/omics/data/scRNA/cellxgene/2025-01-30/cellxgene_random_1pct_h5ad/processed.h5ad`  
Branch: `scconcept-from-main` (merged from `main` at start of sessions)

---

## What we started with

- Three ablation YAMLs that used the old `AnnCollectionDataModule` API; main had merged a new per-split `processed_data_source` feature (PR #42)
- `scconcept_ablation_both.yaml` was structured as sequential fine-tuning, not simultaneous training
- `ContrastiveDataModule` only overrode `train_dataloader`; `val_dataloader` inherited the base class random-batch implementation

---

## What we did (across two sessions)

### Config fixes
- Updated all three configs to use `DataModule` / `ContrastiveDataModule` with the new `processed_data_source` + `split_column_name: split` dataset kwargs
- Removed cell type `label_columns` and HCE loss from all three — pure reconstruction ablations
- Rewrote `scconcept_ablation_both.yaml` from scratch as simultaneous WCED + contrastive (not sequential fine-tuning), `checkpoint: null`

### Val dataloader fix
- Diagnosed that val `recall@1` was flat at random baseline (~1.6%) while train recall saturated near 100%
- Root cause: val was using the base class random-batch dataloader, not condition-homogeneous sampling; cross-type in-batch negatives made it harder than train
- Fix: added `val_dataloader()` override to `ContrastiveDataModule` calling the same `_contrastive_dataloader` helper as train

### OOM fix on `scconcept_ablation_both`
- Jobs 7735–7738 crashed on `ModuleNotFoundError: No module named 'czbenchmarks'` (top-level import in `callbacks.py`); fixed with lazy import
- Job 7740 (first run with val fix) crashed with CUDA OOM: `Tried to allocate 9.44 GiB, 1.77 GiB free`
- Root cause: `accumulate_grad_batches: 2` keeps the previous training step's activations live while validation runs; the WCED decoder's `[2N × seq_len × 19071]` logit tensor from train was still on GPU when val tried to allocate the same
- Fix: `accumulate_grad_batches: 1` in all three configs (must match for comparable training schedules; effective batch size is already 2N=128 from `_PairedViewCollator`)

### pyproject.toml fix
- `cellxgene_census[experimental]<=1.17.0` blocked install since 1.18.0 is the only available version; removed upper bound

---

## State at end of session

### Running jobs

| Job | Config | Status |
|---|---|---|
| 7976 | `scconcept_ablation_contrastive` | Submitted |
| 7977 | `scconcept_ablation_both` | Submitted |
| WCED | `scconcept_ablation_wced` | **Not yet submitted** |

Previous job 7737 (contrastive, older code) ran to completion (`max_epochs=10`); job 7738 (both) was still showing in `bjobs` as running but had OOM'd — likely stale LSF state.

### Key config state (all three ablations)

- `accumulate_grad_batches: 1`
- `n_cells_per_batch: 64` (effective batch 128 after `_PairedViewCollator`)
- `max_epochs: 10`, `limit_val_batches: 20`, `val_check_interval: 0.2`
- `precision: 16-mixed`
- No `label_columns`, no cell type head
- ClearML project: `bmfm-targets/scconcept_ablation`

### Key code state (`bmfm_targets/training/data_module.py`)

`ContrastiveDataModule` has:
- `_obs_conditions(dataset)` — builds condition Series from `cell_type_ontology_term_id` (or specified columns)
- `_contrastive_dataloader(dataset, collate_fn)` — `ConditionHomogeneousBatchSampler` + `_PairedViewCollator`
- `train_dataloader()` and `val_dataloader()` both call `_contrastive_dataloader`

### Submit command pattern (use this, not `source activate`)

```bash
bsub -J <job_name> -o /u/dmichael/.lsf/cluster/%J.out -e /u/dmichael/.lsf/cluster/%J.err \
  -R 'select[ngpus>=1] span[hosts=1]' -M 128G -gpu 'num=1:mode=exclusive_process' \
  "/proj/bmfm/users/dmichael/envs/biomedmultiomic/bin/bmfm-targets-run -cd /u/dmichael/biomed-multi-omic/run -cn <config_name>"
```

---

## Pending (superseded — see Session 3 below)

1. ~~Submit WCED ablation job~~
2. ~~Verify 7976/7977~~
3. **Evaluate final checkpoints** on downstream tasks (cell type classification, embedding quality) to compare the three ablations

---

# Session 3 (2026-07-03): root cause of the "overfitting", fixes, aligned re-run

## TL;DR

The "massive overfitting" was **not** a real property of the scConcept loss. The
condition-grouped batch sampler was a **silent no-op**, so batches were always random
and the contrastive representation never learned to generalize (val recall pinned at
the ~1% random baseline while the temperature-scaled val loss ran away). After fixing
the plumbing, aligning the paper's dataset-grouped batching, and two runtime fixes,
**all three arms now train and generalize** (contrastive val recall@1 ≈ 0.6).

## Root cause (corrected diagnosis)

- Earlier hypothesis "cell-type-homogeneous batching causes overfitting" was **wrong** —
  it assumed the sampler grouped by cell type. It never grouped by anything.
- `ContrastiveDataModule._obs_conditions` read `dataset.h5ad.obs`, but the RNA dataset
  exposes obs as `dataset.metadata` (there is no `.h5ad` attr). `hasattr` was always
  False → returned a single `"all"` group → `ConditionHomogeneousBatchSampler` grouped
  nothing → **random batches**, `condition_columns` a dead config since commit a6dafc0.
- The runaway val loss was a temperature artifact: `logit_scale` (the multiplier) grew
  unbounded (→65) and the InfoNCE val loss is scaled by it, so misclassified val pairs
  incurred exploding penalty while val **recall** just sat at random.

## Paper (biorxiv 2025.10.14.682419, Methods) — the intended recipe

- Positive pair = two disjoint gene-panel views of the same cell; negatives = other
  cells in the batch **plus** the remaining cells in the anchor's own view (the
  `both_batch` 2N×2N form).
- Mini-batches are built from **cells drawn exclusively from a single dataset** (NOT
  cell type): related cell types within a dataset are the organic hard negatives, and
  this avoids letting the model use inter-dataset technical differences as a shortcut.
  Reference code: `within_group_sampling: "dataset"` in every training config.

## Fixes (all committed/pushed to `origin/scconcept-from-main`; editable install → live)

| Commit | Fix | Ablation impact |
|---|---|---|
| `39c7f74` | `_obs_conditions` → read `dataset.metadata`; **raise** (listing available obs cols) if a `condition_columns` entry is missing instead of silently falling back. Also: apply `_PairedViewCollator.feature_dropout` (was dead), reseed collator RNG per worker (was pickled → identical across workers), clamp `logit_scale` to ln(100). | enables grouping; neutral otherwise |
| `be6c277` | wced arm made pure reconstruction (drop the stray HCE cell-type loss + `celltypeont` ontology, which also crashed on a missing gitignored `ontology.graphml`). | restores intended ablation |
| `87d2aad` | Align all three arms to the vanilla WCED baseline (`llama_medium_wced_h5ad_nolabels`): `num_workers 16`, `max_epochs 11`, `log_normalize_transform true`, effective batch 64 cells/step (wced `batch_size 8→64`; contrastive/both `n_cells_per_batch 64`, accumulate 1), `limit_val_batches 20`. | arms differ only in loss |
| `ea51ccf` | WCED decoder decodes **token 0 only** (was run over all 1024 positions → 9.4 GiB tensor). Only token 0 is consumed (`WCEDFieldSource.decode_token_index=0`), so gradients already flowed only through it. **Verified numerically identical (max abs diff 0.0).** Fixes the `both`-arm OOM and ~halves wced per-step FLOPs. | numerically neutral |
| `110731f` | Collation crash fix: `_batchwise/_random_pad_zero` used `list(itemgetter(*idx)(values))`, which returns a scalar (not a tuple) when `idx` has length 1 → `'float' object is not iterable`. The `both` arm hits it because paired-view panels can be tiny. Replaced with a comprehension correct for any length. | numerically neutral |

Config: `condition_columns: [dataset_id]` in contrastive/both (and `scconcept_pretrain.yaml`).
Per-arm differences kept intentionally: WCED masking (wced/both), contrastive panel-split,
`expose_zeros` (all for wced; null for contrastive/both).

## Current runs (as of ~16:40, 2026-07-03)

| Job | Arm | Host | Status | Notes |
|---|---|---|---|---|
| 7987 | contrastive | c05-03 | **DONE** (11 epochs, step 75744) | **val recall@1 = 0.70 (best 0.74), recall@5 = 0.91 (best 0.97)**; recall climbed to the end — converged well |
| 7991 | wced | b05-02 | RUN, ~step 2.3k | val MSE ≈ 0.17 stable; **slow throughput (data-loading bound)** |
| 7992 | both | b05-02 | **DONE** (11 epochs, step 75744) | val recall@1 **0.65** (best 0.69), recall@5 **0.89** (best 0.93); WCED val MSE **0.178**; contrastive val loss 1.12 |

**Preliminary ablation signal** (val, paired-view recall; downstream eval is the real comparison):
contrastive-alone recall@1 **0.70** > both **0.65** — adding WCED reconstruction slightly lowered
contrastive recall while also reconstructing well (MSE 0.178). wced-alone has no paired-view recall.

A persistent Monitor (id in session) watches all three; alerts only on crash/completion.
Metrics also stream to ClearML `bmfm-targets/scconcept_ablation`. Curves live in
`training_runs/<arm>/lightning_logs/version_*` (newest = current run).

## Open items / heads-up

- **wced GPU utilization is poor** — it is data-loading bound, not compute bound. Cause:
  `expose_zeros: all` pushes ~19,071 genes/cell through the `batch_wise` pad-zero transform
  on CPU workers; contrastive/both use `expose_zeros: null` and run ~19× more steps/hour.
  Safe (ablation-neutral) levers to try: higher `num_workers`/`prefetch_factor`, or a
  cheaper vectorized pad-zero transform. Changing `expose_zeros`/`max_length`/batch would
  affect the comparison — don't, without a decision.
- `logit_scale` climbs toward the ln(100) clamp (contrastive ~37, both ~27.5); clamp is a
  safety cap, not yet binding.
- **Evaluate** the final checkpoints on downstream tasks; judge contrastive generalization
  by **val recall on paired views**, not the temperature-scaled loss.
- **Checkpoint selection gotcha (contrastive/both):** `val_loss` is confounded by the growing
  `logit_scale`, so it rises over training even as recall improves — a `val_loss`-monitored
  "best" checkpoint picks an early, worse model (e.g. contrastive `epoch=1 val_loss=0.22` <
  `epoch=10 val_loss=0.74`, but epoch 10 has higher recall). Use `last.ckpt` or select by
  val recall for the contrastive/both arms.
- The repeat-pad path in `ConditionHomogeneousBatchSampler` duplicates cells when a group
  is smaller than `batch_size` (false negatives). Rare for `dataset_id` on the full dump;
  the train split may push a few small datasets under 64. Consider skipping/merging
  under-full groups.

## Submit pattern (unchanged)

```bash
bsub -J <cfg> -o /u/dmichael/.lsf/cluster/%J.out -e /u/dmichael/.lsf/cluster/%J.err \
  -R 'select[ngpus>=1] span[hosts=1]' -M 128G -gpu 'num=1:mode=exclusive_process' \
  "/proj/bmfm/users/dmichael/envs/biomedmultiomic/bin/bmfm-targets-run -cd /u/dmichael/biomed-multi-omic/run -cn <cfg>"
```

---

# Session 4 (2026-07-06/07): zero-shot benchmark of the final 3-arm checkpoints

## Goal

Answer the Session 3 open item — evaluate the final checkpoints on downstream tasks —
using the full 12-dataset zero-shot benchmark suite (`bmfm_targets/evaluation/benchmark_configs/`),
not the earlier July-3 preliminary runs.

## Which checkpoints

Identified via the three ClearML task IDs the user linked (project
`bmfm-targets/scconcept_ablation`, all `completed`):

| Arm | ClearML task ID | jobid tag | Checkpoint used |
|---|---|---|---|
| contrastive | `893c90f188d245f78f393e2ab8c02208` | 10027 | `.../scconcept_ablation_contrastive/last-v2.ckpt` |
| wced | `b6e62211ec124725840f84b82f5e62f3` | 10028 | `.../scconcept_ablation_wced/last.ckpt` |
| both | `fca4575028b44e9191117879074a5dfb` | 10029 | `.../scconcept_ablation_both/last-v2.ckpt` |

All under `/proj/bmfm/users/dmichael/training_runs/`. Used `last`/`last-v2` (not the
val_loss-best checkpoints) per the Session 3 checkpoint-selection gotcha — `logit_scale`
confounds `val_loss` for the contrastive/both arms.

## benchmark_run.sh fixes (committed)

- `bmfm-targets-scbert` → `bmfm-targets-run` (this repo's console script; the old name
  is the inner-source binary — see `.claude/environment.md`).
- Active `datasets` array had been left on the 2-dataset fine-tune subset
  (`multiple_sclerosis`, `myeloid`) from a prior session; restored the full 12-dataset
  zero-shot list as default (`covid_19 heart_atlas immune_all_human lung_atlas pbmc_10k
  cell_lines dc human_pbmc immune_atlas pancrm multiple_sclerosis myeloid`).

Launched with `--do_zs=true --do_ft=false` for all three checkpoints → 36 LSF jobs
(12 datasets × 3 arms), zero-shot only (`task=predict`, `~model ~tokenizer
~trainer.losses ~fields` — reconstructed from the checkpoint).

## Two environment gaps found and fixed (both in `biomedmultiomic` env, not code bugs)

1. **`ModuleNotFoundError: czbenchmarks` / `scib`** — `CziBenchmarkCallback` /
   `BatchIntegrationCallback` (`training/callbacks.py`) lazily import `czbenchmarks` and
   `scib.metrics`; neither was installed in `biomedmultiomic` (only in the inner-source
   `conda_targets` env). Fix: `pip install -e ".[benchmarking]"` — pyproject.toml already
   declares this extra (`scib`, `scib-metrics`, `cz-benchmarks` from the
   `sivanravidos/cz-benchmarks` fork). Confirmed end-to-end afterward (job 10433 DONE).
   **Gotcha:** installing while jobs are already running doesn't help them — the fix has
   to land before each job's Python process reaches the lazy import, and NFS propagation
   to compute nodes lags the login node by a couple of minutes, so several jobs raced the
   install and had to be resubmitted.
2. **`corrupted size vs. prev_size` (glibc heap corruption, core dump)** — hit only on
   the larger atlas datasets (`immune_all_human`, `immune_atlas`, etc.) in the
   sklearn-based `SGDCallback`/benchmark path. Classic OpenBLAS-thread-count bug: the
   node has far more cores visible than the job's LSF request (`-n 8`), and OpenBLAS's
   thread-metadata workaround ("precompiled NUM_THREADS exceeded") is fragile under
   that mismatch. Fix: cap threads to match the allocation via
   `bsub -env 'all, OPENBLAS_NUM_THREADS=8, OMP_NUM_THREADS=8, MKL_NUM_THREADS=8'`.

## Outcome

All 36 (arm × dataset) combinations completed successfully after retries (some
datasets needed a second or third submission due to the two issues above, not because
of anything dataset- or checkpoint-specific). Results are in ClearML under
`bmfm-targets/evaluate/scconcept_{contrastive,wced,both}/20260706-*_scconcept_ablation_zs`.

## Open items

- Pull the `CziBenchmarkCallback`/`SGDCallback`/`BatchIntegrationCallback` metrics from
  ClearML across the three arms and compare — this is the actual answer to "does
  scConcept backbone beat WCED on standard tasks" (Session 2 open question #1).
- Consider committing `OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS`/`MKL_NUM_THREADS` into
  `benchmark_run.sh`'s `PREFIX_CMD` directly so future runs don't need to rediscover the
  malloc-corruption fix.
