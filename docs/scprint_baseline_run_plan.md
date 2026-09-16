# scPRINT baseline run plan (manuscript revision, R3 major comment 2)

Status as of 2026-09-16. Branch `scprint-baseline-fixes` (off `add-scprint-baseline`).

**Goal:** `avg_batch_correction` and `avg_bio_conservation` for scPRINT on the sceval
datasets used in the manuscript, comparable to the existing rows in
`baseline_metrics_all.csv`. Embeddings computed on GPU via LSF. Originals never
overwritten.

---

## 1. What is already done

`bmfm_targets/evaluation/baselines.py` — `compute_scprint` rewritten to match
scPRINT's own published integration benchmark
(`scprint.tasks.cell_emb.default_benchmark`). Eight defects in the original
commit `d37ccf4` were fixed:

| # | Defect in `d37ccf4` | Fix |
|---|---|---|
| 1 | read `obsm["scprint_emb"]`; scPRINT writes `obsm["scprint"]` (`cell_emb.py:236`). `KeyError` was swallowed by `except Exception: return None`, so every dataset silently logged "skipped" | read `"scprint"`; raise `RuntimeError` listing available keys |
| 2 | no `scdataloader.Preprocessor` call — HUGO symbols fed to a `Collator` keyed on Ensembl IDs ⇒ ~zero gene overlap | `Preprocessor(is_symbol=…, force_preprocess=True, skip_validate=True, do_postp=False)`, mapping rate logged |
| 3 | organism hardcoded `NCBITaxon:9606` | `organism` arg, `SCPRINT_MOUSE`, `--scprint-organism`, hard check against `model.organisms` |
| 4 | model never moved to GPU — `Embedder` reads `model.device`, so inference ran on CPU | `model.to(device)`, device defaults to cuda |
| 5 | `dtype=torch.float32` passed into a CUDA autocast context (rejected) | dropped; scPRINT's `float16` default stands |
| 6 | `how="most var"` + 4-token `pred_embedding` | `how="random expr"`, `pred_embedding=("cell_type_ontology_term_id",)` |
| 7 | `Embedder` writes an intermediate h5ad into `./data/` | run inside `tempfile.TemporaryDirectory` with `os.chdir` in try/finally |
| 8 | counts check used `guess_if_raw(adata.X.data)` (wrong for dense) and only when a `counts` layer existed | `_counts_source()` scans X + every layer for non-negative integrality; **raises** if none qualifies |

Only a missing `scprint` install returns `None`. Every form of misuse raises, so a
broken baseline can never be recorded as a skipped one.

`bmfm_targets/tests/test_scprint_baseline.py` rewritten — the old suite mocked
`Embedder` to write `obsm["scprint_emb"]` itself, so it passed *because* of defect
1. New suite asserts the real key, the `Preprocessor` call, canonical Embedder
settings, `model.to(device)`, organism threading, input non-mutation, and that
log-normalised input raises.

---

## 2. Datasets

`BMFM_TARGETS_SCEVAL_DATA=/proj/bmfm/omics/data/scRNA/finetune/` (set in `~/.bashrc`).
Each dataset lives at `$BMFM_TARGETS_SCEVAL_DATA/<name>/h5ad/original.h5ad`.

All 9 runnable datasets use `data_module/sceval_default.yaml`, so columns are
uniform: **`label_column_name: CellType`**, **`batch_column_name: batch`**.

### Run these 9 (verified genuine raw counts: integer, non-negative, no `counts` layer)

| dataset | in existing `X_*` FM comparison set? |
|---|---|
| `cell_lines` | yes |
| `covid_19` | yes |
| `heart_atlas` | yes |
| `human_pbmc` | yes |
| `immune_atlas` | yes — but see note |
| `lung_atlas` | yes |
| `pbmc_10k` | yes |
| `immune_all_human` | no FM baselines to compare against |
| `pancrm` | no FM baselines to compare against |

All 9 are human ⇒ `--scprint-organism NCBITaxon:9606` (the default).

> **`immune_atlas` caveat:** `var_names` are mixed identifiers (`MIR1302-2HG`
> alongside `ENSG00000238009.6`). Symbol auto-detection will pick `is_symbol=True`
> and mapping will be partial. **Report the logged mapping rate for this dataset
> explicitly** in the writeup.

### Excluded from the 12-dataset zero-shot list (3), user-confirmed

Confirmed by the user: *"right those three are available as lognormed only, original
counts not available."* scPRINT conditions on sequencing depth, so embedding
log-normalised data would produce a plausible-looking wrong number.

- **`dc`** — `original.h5ad` max 598415.1, non-integer, no layers / raw / backup counts.
- **`multiple_sclerosis`** — no `original.h5ad`; all files log1p, 3000 HVG.
- **`myeloid`** — no `original.h5ad`; all files log1p, 3000 HVG.

`multiple_sclerosis` and `myeloid` are the *fine-tune* pair (`benchmark_run.sh:117`),
use `celltype`/`str_batch` columns, and appear nowhere in `baseline_metrics_all.csv`.
`dc` is in that CSV with classical baselines only and no `X_*` FM rows.

### Separate data-integrity finding to report (not blocking)

`perirhinal_cortex/h5ad/original.h5ad` is **byte-identical** to
`lung_atlas/h5ad/original.h5ad`: same size (1375890925 bytes), same md5 over the
first 100 MB (`666a2f9572ecf8e3180669015ed92fb2`), same 47 cell types, same leading
genes (`TSPAN6/TNMD/DPM1`). Both `backup_original*.h5ad` pairs match too, back to
Jan 5 2026. The manuscript's `perirhinal_cortex` row is therefore `lung_atlas`
scored twice (values agree to ~1e-7). Not in the chosen 12-dataset list, so it does
not block this run, but it needs to be raised.

---

## 3. Environment (not yet built)

scPRINT cannot go into `biomedmultiomic` (Python 3.13, torch 2.12.1): it pins
`torch==2.2.0` and `lamindb==0.76.12`. It needs a dedicated env.

```bash
export H2=/proj/bmfm/users/dmichael
conda create -y -p $H2/envs/scprint python=3.11
$H2/envs/scprint/bin/pip install scprint 'lamindb[bionty]'

# scdataloader.Preprocessor needs lamindb+bionty for symbol -> Ensembl mapping
$H2/envs/scprint/bin/lamin init --storage $H2/lamindb_scprint --schema bionty

# checkpoint (download from the login node — network confirmed reachable)
export HF_HOME=$H2/.cache/huggingface
$H2/envs/scprint/bin/hf download jkobject/scPRINT medium-v1.5.ckpt \
  --local-dir $H2/scprint_baseline/ckpt
```

Reference for correct usage: `scprint/tasks/cell_emb.py` from the scprint 1.6.4
wheel (`default_benchmark`). A copy was extracted to `/tmp/scpr_probe/ext/` during
investigation; re-extract from the wheel if that tmp dir is gone.

Work dir: `/proj/bmfm/users/dmichael/scprint_baseline/` (created).

---

## 4. Runner (not yet written)

`/proj/bmfm/users/dmichael/scprint_baseline/embed_one.py` — one dataset per
invocation. Contract:

- read `$BMFM_TARGETS_SCEVAL_DATA/<ds>/h5ad/original.h5ad` **read-only**;
- call `bmfm_targets.evaluation.baselines.compute_scprint` directly (do *not* call
  `run_baselines`, which writes back to `adata_path` by default);
- save to a **new** file `$WORK/emb/<ds>_scprint.h5ad` holding only
  `obs[["CellType","batch"]]` + `obsm["scPRINT"]` — small, and enough for scoring;
- never open `original.h5ad` for write. This is the user's hard constraint
  (*"transiently without overwriting the files"*, *"Separate files, originals untouched"*).

Note the two repos both provide package `bmfm_targets`; `compute_scprint` must be
imported from **this** repo (`/u/dmichael/biomed-multi-omic`), e.g. via
`PYTHONPATH=/u/dmichael/biomed-multi-omic`, since the scprint env will not have it
installed.

## 5. LSF submission (not yet written)

The user was explicit twice: the embedding run happens on GPU via LSF. Copy the
`PREFIX_CMD` pattern from `bmfm_targets/evaluation/benchmark_configs/benchmark_run.sh:142`:

```bash
PREFIX_CMD=(
  bsub
  -o "$HOME/.lsf/cluster/%J.out"
  -e "$HOME/.lsf/cluster/%J.err"
  -n 8
  -R 'select[ngpus>=1] span[hosts=1]'
  -M 128G
  -gpu 'num=1:mode=exclusive_process'
)
```

One job per dataset, 9 jobs. Logs land in `$HOME/.lsf/cluster/`.

Cluster checked 2026-09-16: `bsub` at
`/opt/share/IBMLSF-fp15/10.1/linux4.18-glibc2.28-x86_64/bin/bsub`, `normal` queue
open (1440 pending / 389 running), GPU hosts `zu-a100-{b,c}05-0{1,2}` with
`NVIDIAA100_SXM4` 80 GB cards, several idle. The `PREFIX_CMD` above does not name a
queue, so jobs land in `normal`.

## 6. Scoring (not yet run)

Must reuse the **canonical manuscript scorer** verbatim for comparability:
`/u/dmichael/bmfm-targets/run/compute_baseline_metrics.py` (inner-source, read-only).

```
Benchmarker(
    bio_conservation_metrics=BioConservation(isolated_labels=False),
    batch_correction_metrics=BatchCorrection(),
    n_jobs=-1,
)
bm.prepare(); bm.benchmark()
bm.get_results(min_max_scale=False, clean_names=True)
```

CLI: `--adata-path --batch-col --label-col --output --append --pre-integrated --dataset-name`.
Shebang pins `$H2/envs/targets3-13/bin/python`.
`EXCLUDED_OBSM_KEYS = {X_pca, X_umap, X_tsne, X_diffmap, X_draw_graph_fr}`.

**Do not** use this repo's `embeddings.py:calculate_batch_integration_metrics` — it
is built on `scib.metrics` and emits differently-named `avg_bio`/`avg_batch`. The
manuscript numbers come from `scib_metrics.Benchmarker`. Mixing the two gives
incomparable values.

`min_max_scale=False` scores each `obsm` embedding independently, which is why
scoring scPRINT transiently is legitimate: it cannot perturb the already-published
baseline values.

Target table to join — `/proj/bmfm/omics/data/scRNA/finetune/baseline_metrics_all.csv`:

```
dataset,baseline,avg_batch_correction,avg_bio_conservation,avg_total,KMeans NMI,
KMeans ARI,Silhouette label,cLISI,BRAS,iLISI,KBET,Graph connectivity,PCR comparison
```

Write scPRINT rows to a **separate** CSV in `$WORK/`, not by appending to the
manuscript table.

---

## 7. Remaining checklist

- [ ] build `$H2/envs/scprint` + `lamin init`
- [ ] download `medium-v1.5.ckpt`
- [ ] smoke-test `compute_scprint` on a ~2k-cell subset of `pbmc_10k`; confirm the
      logged gene-mapping rate is high and the embedding is non-degenerate
- [ ] write `embed_one.py` + `submit_scprint.sh`
- [ ] submit 9 GPU jobs, verify all 9 embeddings exist with the right row counts
- [ ] score all 9 with the inner-source scorer
- [ ] report per-dataset `avg_batch_correction` / `avg_bio_conservation`, the
      3 exclusions, the `immune_atlas` mapping rate, and the
      `perirhinal_cortex`/`lung_atlas` duplication

## 8. Untracked files to leave alone

Unrelated to this work, do not commit:
`bmfm_targets/evaluation/benchmark_configs/data_module/bmmc_xsite.yaml`,
`bmfm_targets/tokenization/resources/scconcept_panels.py`,
`bmfm_targets/tokenization/resources/scconcept_panels/`.
