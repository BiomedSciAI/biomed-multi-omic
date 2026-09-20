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
# large-v1 is the checkpoint to actually benchmark with (see §7): 101.1M params,
# d_model 512 / nlayers 16, vs medium-v1.5's 18.7M / 256 / 8. Same 44,756-gene
# vocabulary and same organisms, so the gene-alignment machinery is unchanged.
$H2/envs/scprint/bin/hf download jkobject/scPRINT large-v1.ckpt \
  --local-dir $H2/scprint_baseline/ckpt
```

Both checkpoints carry a stale absolute `precpt_gene_emb` parquet path, which is why
`baselines.py` loads with `scPrint.load_from_checkpoint(ckpt_path,
precpt_gene_emb=None)`.

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

> **CORRECTION (2026-09-17). This section had the convention backwards.** It
> originally said to use `scib_metrics.Benchmarker` and **not** this repo's
> `embeddings.py:calculate_batch_integration_metrics`. That is wrong, and it is the
> reason the first set of results (below) was not comparable to the manuscript.
>
> There are three distinct conventions in play:
>
> 1. `scib_metrics.benchmark.Benchmarker` — KMeans NMI/ARI; `avg_bio` also folds in
>    cLISI, `avg_batch` also folds in BRAS/iLISI/kBET/PCR. This produced
>    `baseline_metrics_all.csv` and the original `scprint_metrics.csv`. It is an
>    **internal** table and is *not* the convention of the published zero-shot
>    tables.
> 2. `bmfm_targets.evaluation.embeddings.calculate_batch_integration_metrics` —
>    `scib` + Leiden, `avg_bio = mean(NMI, ARI, ASW_label)`,
>    `avg_batch = mean(graph_conn, ASW_label/batch)`. **This is the Methods /
>    published-table convention.**
> 3. The scGPT zero-shot tutorial's `scib_eval` — same two formulas, plus
>    `weighted = 0.6*avg_bio + 0.4*avg_batch`.
>
> (2) and (3) agree to 3 decimals on every dataset scored under both, and (3)
> reproduces the published scGPT `cell_lines` row exactly (NMI 0.7567, ARI 0.7230,
> ASW_label 0.6834 → `avg_bio` 0.7210 vs published 0.721; graph_conn 0.9994,
> ASW_label/batch 0.9020 → `avg_batch` 0.9507 vs published 0.951). So (3) is the
> like-for-like reference, and convention (1) must not be used for any number that
> is placed beside a published one.
>
> Scorers: `$WORK/score_paper_convention.py` (2) and
> `$WORK/score_tutorial_convention.py` (3).

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

- [x] build `$H2/envs/scprint` + `lamin init`
- [x] download `medium-v1.5.ckpt`
- [x] download `large-v1.ckpt` (101.1M params — the checkpoint the paper's zero-shot
      claim actually refers to; `CKPT=` / `PRED_EMB=` in `submit_scprint.sh` select it)
- [x] smoke-test `compute_scprint` on a ~2k-cell subset of `pbmc_10k`; confirm the
      logged gene-mapping rate is high and the embedding is non-degenerate
- [x] write `embed_one.py` + `submit_scprint.sh`
- [x] submit 9 GPU jobs, verify all 9 embeddings exist with the right row counts
- [x] ~~score all 9 with the inner-source scorer~~ — done, but on convention (1);
      **superseded**, see §6 correction
- [x] re-embed all 9 with `large-v1` and re-score under conventions (2) and (3)
- [x] report per-dataset `avg_batch_correction` / `avg_bio_conservation`, the
      3 exclusions, the `immune_atlas` mapping rate, and the
      `perirhinal_cortex`/`lung_atlas` duplication
- [x] benchmark scPRINT-2 `ji9krimq` — **done but INVALID** (`_v3`/`_v3a`): the
      gene-index alignment repair was never applied to the v2 path, so all 20,004
      human genes were misindexed. Root-caused at the end of this section.
- [x] generalise the alignment helpers to per-organism vocabularies and call them
      from `embed_one_v2.py`; repair the `scprint2db` ontology (8 IDs, 16/16 exact)
- [x] re-embed all 9 with the alignment fixed (`_v4`), re-score under both
      conventions, and replace the `_v3`/`_v3a` tables — `avg_bio` 0.310 → **0.505**,
      gained on all 9 datasets, and 0.505 falls inside upstream's own 0.467–0.553
      range for this checkpoint
- [x] `_v4a` (all facets) — all 9 scored under **both** conventions (max disagreement
      5e-4); the facet choice is a wash on the headline (`avg_bio` mean 0.506 vs
      0.505), and `_v4a` falls *below* the measured noise floor on `human_pbmc`, so
      `_v4` stays the reported run
- [x] second noise control (`Random424`, iid gaussian) completed on all 9 — it had
      been scored on only 4, and those 4 were 4 of the 5 highest floors, so the
      4-dataset mean `avg_batch` 0.751 was selection-biased upward. On all 9 it is
      **0.660**. Nothing written down had quoted 0.751; the reported `NULL` column
      was always the 9-dataset shuffle
- [ ] decide with the user how to present `Unintegrated` (0.637 `avg_bio`, which
      ties or beats every published method in the table, BMFM's three included)

**Done 2026-09-16, corrected 2026-09-17.**

- Current results: `$WORK/scprint_large_v1_vs_published.csv` (scPRINT-1) and
  `$WORK/final_table_v4_tutorial.csv` (the combined scPRINT-1 + scPRINT-2 `_v4` +
  `Unintegrated` + `NULL` table, tutorial convention), with per-dataset scorer output
  in `$WORK/metrics_tutorial/` and `$WORK/metrics_paper/` (`*_lg_*` = `large-v1`,
  `*_v4_*` = scPRINT-2 `ji9krimq` with the gene index repaired) and embeddings in
  `$WORK/emb/` (`*_lg_scprint.h5ad`, `*_v4_scprint.h5ad`). Nothing was appended to
  the manuscript's `baseline_metrics_all.csv`.
- Superseded: `$WORK/scprint_metrics.csv` and `$WORK/metrics/` — convention (1) on
  `medium-v1.5`. Left in place, not cited, because `summarize_conventions.py` still
  reads them. `scprint_metrics.csv` is the one genuine trap in `$WORK`: its column
  names are *identical* to the manuscript's `baseline_metrics_all.csv`
  (`avg_batch_correction`, `avg_bio_conservation`, …) so it looks pasteable, while its
  `avg_bio_conservation` is on a different scale from everything else here (0.919 on
  `cell_lines`). `$WORK/README_WHICH_FILES.md` names the reportable files and the
  do-not-quote ones for anyone arriving at that directory without this document. The
  published `baseline_metrics_all.csv` was never written to, and no `original.h5ad`
  was ever opened for write.
- All 18 (9 datasets × 2 conventions) `large-v1` scorings completed; conventions
  (2) and (3) agree on all 18 dataset/embedding pairs to within 5e-4, which is the
  output precision of the (2) scorer.

Reproduction control passed: the Unintegrated embedding carried in each embedding
file was scored alongside scPRINT. Under convention (1) it reproduced the published
Unintegrated row on all 9 datasets — 17 of 18 values bit-identical, the 18th
(`pbmc_10k` `avg_batch_correction`) off by 5.1e-4 from iLISI/KBET subsampling. Under
convention (3) the `cell_lines` scGPT row is reproduced exactly (see §6). Together
these license comparing the scPRINT rows to the rest of the table.

Two prerequisites that a naive setup gets silently wrong, both fixed here:

- **Retired Ensembl IDs misalign the gene index.** bionty ships Ensembl release-114
  only; the checkpoint's `model.genes` was built on an earlier release, so IDs
  retired in between are absent from the ontology. `Collator` masks positionally
  against a sorted var axis, so a single missing ID shifts every downstream gene by
  one — 66.4% of human genes were misindexed. **A misaligned index still yields a
  finite, well-conditioned embedding of pure noise**, so "the embedding looks fine"
  does not detect this.

  Handled inside `baselines.py`, so there is nothing to remember to run:
  `_scprint_index_drift` reports the first misaligned position per organism,
  `_register_retired_genes` adds the missing IDs back, and
  `_ensure_scprint_gene_alignment` runs the check before any GPU work and repairs
  or raises (`align_ontology`, default True). A second check after preprocessing
  asserts the same invariant on the array actually handed to the Collator. Verified
  against the live ontology: 23149/23149 human, 21607/21607 mouse, no drift.

  **This is the bug that later invalidated the first scPRINT-2 runs**, because
  `embed_one_v2.py` was written standalone against the `scprint2` package and never
  routed through these helpers. If another model path is ever added, calling
  `_ensure_scprint_gene_alignment` is not optional — the failure is silent and
  produces plausible numbers. See the root-cause section at the end of §7.
- **Gene coverage varies enormously across the 9** (see §"caveats" below): the
  fraction of the model's 23,149 human slots carrying data ranges from 4.6%
  (`covid_19`) to 79.8% (`heart_atlas`).

### Superseded results (2026-09-16) — do not cite

The first run reported the table below and claimed scPRINT "sits mid-pack among the
zero-shot FMs, beating both geneformer sizes and both UCE variants and losing to
scGPT". **Both the table and that ranking claim are withdrawn.** Two independent
mistakes, neither in `baselines.py`:

1. **Wrong metric convention** — scored with `scib_metrics.Benchmarker`
   (convention (1) in §6), whose `avg_bio`/`avg_batch` are not the published
   tables' quantities. `cell_lines` `avg_bio` alone moves 0.919 → 0.528 between
   conventions, so the comparison to the other baselines was meaningless.
2. **Wrong checkpoint** — used `medium-v1.5` (18.7M params, `d_model` 256), the
   README quickstart default, against scGPT and Geneformer-316M. The paper's
   zero-shot claim is explicitly about the *large* model: "we train scPRINT at
   various scales (from 2M to 100M parameters)" and "contrary to any other method
   assessed, **our large cell model** can also achieve performances on par with many
   famous single-cell RNAseq tools ... without the need for fine-tuning".
   `large-v1` (101.1M, `d_model` 512) is the top of that range.

<details>
<summary>Withdrawn numbers (convention (1), medium-v1.5)</summary>

| dataset | avg_batch_correction | avg_bio_conservation | avg_total |
|---|---|---|---|
| cell_lines | 0.4315 | 0.9194 | 0.7242 |
| covid_19 | 0.3542 | 0.5314 | 0.4605 |
| heart_atlas | 0.3745 | 0.7028 | 0.5715 |
| human_pbmc | 0.2058 | 0.6337 | 0.4626 |
| immune_all_human | 0.4985 | 0.6605 | 0.5957 |
| immune_atlas | 0.3453 | 0.5389 | 0.4615 |
| lung_atlas | 0.3536 | 0.6152 | 0.5105 |
| pancrm | 0.2332 | 0.5215 | 0.4062 |
| pbmc_10k | 0.8488 | 0.6286 | 0.7167 |

</details>

### Corrected results (2026-09-17)

`large-v1`, `pred_embedding=["cell_type_ontology_term_id"]`, scored with
convention (3). `weighted = 0.6*avg_bio + 0.4*avg_batch`.
Source: `$WORK/scprint_large_v1_vs_published.csv`.

| dataset | avg_bio_conservation | avg_batch_correction | weighted | medium-v1.5 weighted | Unintegrated weighted |
|---|---|---|---|---|---|
| cell_lines | 0.6652 | 0.9344 | 0.7729 | 0.6990 | 0.8276 |
| covid_19 | 0.4522 | 0.8502 | 0.6114 | 0.6143 | 0.5698 |
| heart_atlas | 0.6594 | 0.9359 | 0.7700 | 0.7290 | 0.7967 |
| human_pbmc | 0.5456 | 0.8217 | 0.6561 | 0.5622 | 0.7342 |
| immune_all_human | 0.5366 | 0.8890 | 0.6775 | 0.6125 | 0.6867 |
| immune_atlas | 0.5482 | 0.8271 | 0.6598 | 0.5592 | 0.7317 |
| lung_atlas | 0.5413 | 0.9015 | 0.6854 | 0.6887 | 0.7614 |
| pancrm | 0.4985 | 0.7183 | 0.5864 | 0.5354 | 0.6924 |
| pbmc_10k | 0.7446 | 0.9764 | 0.8373 | 0.7008 | 0.8166 |
| **MEAN** | **0.5769** | **0.8727** | **0.6952** | **0.6335** | **0.7352** |

The checkpoint swap is worth +0.093 `avg_bio` and +0.061 weighted (0.634 → 0.695),
and `large-v1` wins on 7 of 9 datasets; `covid_19` and `lung_atlas` are the two
where `medium-v1.5` is marginally ahead, both within 0.004.

### Ranking against the published zero-shot tables

Means over the same 9 datasets (`dc` excluded — it ships log-normalised only, so it
is not in our set; the published means in the manuscript **include** `dc` and are
therefore not directly comparable to a 9-dataset mean). Published values parsed
programmatically from `notebooks/bmfm_rna_v2/results/zeroshot_tables.txt`:

| method | avg_bio | avg_batch | weighted |
|---|---|---|---|
| WCED MULTITASK | 0.624 | 0.907 | **0.738** |
| MLM MULTITASK | 0.630 | 0.895 | 0.736 |
| BMFM CONCAT | 0.620 | 0.908 | 0.735 |
| Unintegrated | 0.637 | 0.883 | 0.735 |
| scGPT | 0.613 | 0.880 | 0.720 |
| **scPRINT (large-v1)** | **0.577** | **0.873** | **0.695** |

**scPRINT ranks last**, 0.025 behind scGPT and ~0.04 behind the three BMFM
variants. It is not the collapse the medium checkpoint implied — it is second-best
on `pbmc_10k` (0.837, behind only scGPT's 0.859), beats scGPT on `pancrm` (0.586 vs
0.564), and beats the Unintegrated control on `covid_19` on both axes — but the
ordering does not change.

Two things to be careful about when writing the R3 response:

- **Unintegrated (0.735) ties or beats every method, including all three BMFM
  variants.** This is a property of this benchmark under this convention, not a bug,
  and the published per-dataset tables already show it (scGPT 0.813 vs Unintegrated
  0.828 on `cell_lines`). If R3 is probing baseline quality, this is exposed.
- **scPRINT-2 exists** (`cantinilab.github.io/scPRINT-2`, bioRxiv Feb 2026) and
  lists cell embedding / batch correction as a supported task. If the revision is
  claiming a comparison against the current scPRINT, that should be acknowledged.

### Usage verified against upstream (2026-09-17)

`compute_scprint` matches `scprint.tasks.cell_emb.default_benchmark`
argument-for-argument (`how="random expr"`, `max_len=4000`, `doclass=False`,
`output_expression="none"`, `Preprocessor(use_layer="counts", is_symbol=True,
force_preprocess=True, skip_validate=True, do_postp=False)`). Also checked against
the README, the FAQ and the tutorials:

- **Raw counts are correct.** README FAQ: *"scPRINT takes raw count as inputs, so
  please don't use integrated data. Just give the raw counts to scPRINT and it will
  take care of the rest."*
- **Single `cell_type` class token is correct, and is not a shortcut.** scPRINT is
  trained with *disentangled* per-facet class tokens (cell_type, disease, ethnicity,
  sex). `medium-v1.5` records exactly this one token as its own `pred_embedding`
  default; `large-v1` records all four. Averaging all four measurably *hurts* bio
  conservation (`pbmc_10k` 2k subset: 0.744 with 1 token vs 0.602 with 4), which is
  what the disentanglement design predicts.
- **`how="random expr"` / `max_len=4000`** mirror pretraining ("2200 randomly
  selected expressed genes") and the paper's note that the model "can make
  predictions on much larger sequences of genes at inference time". Truncation does
  not bind on `pbmc_10k` (~3.3k genes carry data, under the 4000 cap).
- **Gene coverage is not the explanation.** It is *anti*-correlated with the bio
  gap: `covid_19` has the lowest coverage (4.6%) and the smallest gap (0.018),
  `pbmc_10k` has 14.2% coverage and the largest gap (0.262).
- Gene index alignment verified 23149/23149 human, no drift; embeddings are
  512-dim on all 9 (confirming `large-v1`) with no dead dimensions.

Cross-validation: conventions (2) and (3) were run independently on every dataset
and agree to 3 decimals throughout (e.g. `cell_lines` bio 0.6650 vs 0.6652,
`pbmc_10k` 0.7450 vs 0.7446).

### Caveats to state in the revision

- **Gene coverage.** Ontology mapping rate / fraction of the model's 23,149 human
  slots carrying data: `cell_lines` 58.7%/76.4%, `covid_19` 97.3%/**4.6%**,
  `heart_atlas` 69.1%/79.8%, `human_pbmc` 61.9%/78.1%, `immune_all_human`
  97.5%/51.4%, `immune_atlas` **80.5%**/71.9%, `lung_atlas` 77.8%/76.1%, `pancrm`
  94.2%/62.2%, `pbmc_10k` 100.0%/**14.2%**. The two rates are independent: a
  dataset can map nearly all of its own genes and still fill few model slots
  because it was distributed pre-filtered to a small panel. `covid_19` (4.6%) and
  `pbmc_10k` (14.2%) are the datasets where scPRINT sees the least of its input
  space; these numbers are a property of the distributed files, not of the run.
- **`perirhinal_cortex` / `lung_atlas` duplication in the published table.** Across
  all 10 existing baselines these two datasets have bit-identical
  `avg_bio_conservation`, KMeans NMI, KMeans ARI, Silhouette label, cLISI and BRAS.
  Only Harmony, Unintegrated and `X_uce_33l` differ at all, and only in the 4th–7th
  decimal — the signature of the same data scored twice under two names with a
  different seed for the stochastic batch metrics. Does not affect the scPRINT rows
  (`perirhinal_cortex` is not among the 9), but it should be resolved before the
  table is republished.

### scPRINT-2 (2026-09-17)

R3's comment is about scPRINT as a baseline, and scPRINT-2 now exists, so the v1
numbers above are extended with a v2 run rather than left to look like a
comparison against a superseded model.

**Which checkpoint — and the mistake made here first.** The first pass of this run
used `small-v2.ckpt` and asserted in this document that it was the only public v2
checkpoint and that `ji9krimq.ckpt` was an undownloadable internal artefact. **Both
claims were false**, and the resulting 9-dataset benchmark measured the wrong model.

What is actually true: `jkobject/scPRINT-2` on HF does 401 and does hold test
datasets rather than weights, and `config/{medium,large,vlarge}` are training
configs. But the v1 repo `jkobject/scPRINT` has an **`other_ckpts/` subdirectory**
that a top-level listing does not surface, and `other_ckpts/ji9krimq.ckpt`
(2,180,479,695 bytes) is in it, served HTTP 200 with no authentication. That is the
checkpoint the paper's own results come from: upstream's repro notebook for the
batch-correction figure loads exactly it —

```python
# cantinilab/scPRINT-2, notebooks/scPRINT-2-repro-notebooks/batch_corr_op.ipynb, cell 12
ckpt_path = os.path.join(LOC2, "ji9krimq.ckpt")
model = scPRINT2.load_from_checkpoint(ckpt_path, precpt_gene_emb=None, gene_pos_file=None)
```

The lesson worth keeping: `scprint2 easy_setup` downloads `small-v2.ckpt` and every
README inference example uses it, so it is the obvious thing to grab and it loads
without complaint. Nothing in the tooling tells you it is not the published model.
Verify the checkpoint against the paper's own repro notebook, not against the
quickstart.

**The two checkpoints are not interchangeable.** From their `hyper_parameters` and
`state_dict`:

| | `small-v2.ckpt` | `ji9krimq.ckpt` (the paper's) |
| --- | --- | --- |
| params | 187.8M | **409.7M** |
| d_model / nhead / nlayers | 256 / 4 / 8 | **512 / 8 / 16** |
| `expr_emb_style` | `metacell` | **`continuous`** |
| epoch / global_step | 37 / 760000 | 21 / 440000 |
| gene vocab (human) | 335,182 (20,004) | 335,182 (20,004) |
| `pred_embedding` | `['cell_type_ontology_term_id']` | `['cell_type_ontology_term_id']` |

Both carry the same 9 class facets and the same 335,182-gene vocabulary, so the
vocabulary-coverage findings below hold for either. `nlayers_cell` is 6 in both.

##### Flavours, parameters and representation widths

All four checkpoints that were run at any point, counted from the `state_dict` by
module rather than taken from a model card. "Gene-indexed tables" is
`gene_encoder.embedding` plus `pos_encoder.pe`, both one row per vocabulary gene:

| flavour | file | **total** | main transformer | gene-indexed tables | other | d_model | layers | organisms | vocab rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `medium-v1.5` | 222 MB | 18.7M | 6.32M | 11.46M (61%) | 0.93M | 256 | 8 | 2 | 44,756 |
| `large-v1` | 1.24 GB | **101.1M** | 50.44M | 45.83M (45%) | 4.79M | 512 | 16 | 2 | 44,756 |
| `small-v2` | 890 MB | 187.8M | 14.75M | 171.74M (**91%**) | 1.26M | 256 | 8 | 16 | 335,182 |
| `ji9krimq` | 2.18 GB | **409.7M** | 63.87M | 343.75M (**84%**) | 2.10M | 512 | 16 | 16 | 335,182 |

Representation widths, measured as the `obsm` width actually handed to the scorer, not
inferred from `d_model`:

| run | flavour | 1 facet | all facets | facet combination |
| --- | --- | --- | --- | --- |
| `_lg` | `large-v1` | **512** | 512 | average |
| `_v2` / `_v2a` | `small-v2` | **64** | **424** | **concatenate** |
| `_v3` / `_v4` / `_v4a` | `ji9krimq` | **256** | **256** | average |
| — | `Unintegrated` | 30 | — | PCA |
| — | nulls | 424 | — | built from `_v2a`, hence `Random424` |

Two things this settles empirically:

- **`_v4a` is 256 dims, exactly like `_v4`.** Adding four more facets does not widen
  the representation, which is direct confirmation of the `compressor=no` → *average*
  branch at `cell_emb.py:240-251` rather than a reading of the source. `small-v2` is
  the flavour that concatenates (64 → 424).
- **`small-v2`'s single-facet representation is only 64 dims**, and 91% of its
  parameters are vocabulary tables. Another reason the `_v2`/`_v2a` runs are not a
  meaningful scPRINT-2 baseline beyond being the wrong checkpoint.

**Three recipe differences from v1, each of which would silently break a run.**
Taken from the upstream Figure-5 batch-correction repro notebook and cross-checked
against `cell_emb.py` and the checkpoint's own `hyper_parameters`:

1. **`expr_emb_style` is per-checkpoint and must be read off the model, never
   hardcoded.** `small-v2` is `metacell`: it conditions each cell on its k-nearest
   neighbours, so the AnnData needs a graph first — `Preprocessor(do_postp=True)`
   then `sc.pp.neighbors(adata, use_rep="X_pca")`. `ji9krimq` is `continuous` and
   needs neither. Upstream gates both on the same expression
   (`do_postp=model.expr_emb_style == "metacell"`, `default_benchmark`
   `cell_emb.py:440`; the notebook sets `metacell = model.expr_emb_style ==
   "metacell"` in cell 14), and `embed_one_v2.py` does the same. Hardcoding it
   either way silently mis-preprocesses one of the two checkpoints.
2. `gene_pos_file` and `precpt_gene_emb` are stale absolute paths in the checkpoint
   and must be passed as `None`. This is safe and is what upstream itself does
   (`cli.py:197`, notebook cell 12) — **not** a workaround that loses information:
   `on_load_checkpoint` (`model.py:818-823`) reconstructs the positional encoder
   from the checkpoint's own `pos_encoder.pe` (335182x1x256) when `__init__` left
   it `None`, and `model.py:1076` uses it in the forward pass. Verified explicitly,
   because "override the path that fails to resolve" is exactly the shape of a
   change that would quietly zero out gene identity.
3. `model.mask_zeros = True` (notebook cell 14; not a recorded hparam).

`pred_embedding=["cell_type_ontology_term_id"]` is the notebook's value **and** both
checkpoints' recorded default, so it is also what `default_benchmark` uses (it falls
back to `["all"]` only when `model.pred_embedding is None`, which is not the case
here). `max_len=3200` is the notebook's; `default_benchmark` computes 4000 for these
count depths, so this is exposed as a flag rather than assumed. Missing
FlashAttention is harmless: the `attention` hparam is `'normal'`.

**`max_len` measured, not assumed.** Run on `cell_lines`, the widest-panel dataset
in the suite (32,738 genes, 3,338 non-zeros per cell, so the only one that can
exceed 3200 at all):

| `max_len` | avg_bio | avg_batch |
| --- | --- | --- |
| 3200 (notebook) | 0.354 | 0.965 |
| 4000 (`default_benchmark`) | 0.351 | 0.964 |
| 8000 | 0.272 | 0.953 |

More context does not help and 8000 clearly hurts, so the low scores are not the
model being starved of genes. `mask_zeros=True` means only non-zeros are eligible,
which is why 3200 is already close to sufficient here and comfortably so everywhere
else (non-zeros per cell: `covid_19` 126, `pbmc_10k` 452, `immune_atlas` 1324,
`immune_all_human` 1536, `heart_atlas` 1501, `human_pbmc` 1618, `lung_atlas` 1952).

**`use_raw` checked, and inert for these files.** `scdataloader`'s `Preprocessor`
defaults to `use_raw=True` and, at `preprocess.py:180-182`, silently does
`adata.X = adata.raw.X` whenever `.raw` exists — which would override the counts
layer `embed_one_v2.py` validates via `_counts_source`, making that validation
meaningless. Verified across all 9: none has `.raw` and none has any `layers`, so
`X` is what reaches the model and the check holds. (It is *not* inert for upstream's
cellxgene inputs, where `X` is normalised and `.raw` holds the counts — there the
same line is what makes their run correct.)

**v1's "averaging dilutes the cell-type signal" finding does not transfer.** With
`model.compressor is not None` — small-v2's compressor *is* a `ModuleDict` — the v2
`Embedder` **hstacks** per-facet embeddings rather than averaging them, so
`pred_embedding=all` is a legitimate variant to measure, not a known-worse one.

**Isolation.** Separate venv (`envs/scprint2`), separate lamindb settings dir
(`.lamin_v2`) and instance (`lamindb_scprint2`), separate runner
(`embed_one_v2.py` / `run_embed_v2.sh` / `submit_v2.sh`), separate output suffix
`_v2`. The v1 numbers therefore stay reproducible. Same read-only contract on
`original.h5ad`. Two v2-specific hazards handled: the Preprocessor replaces
`obs.index` with fresh UUIDs, so the embedding is mapped back **positionally**
behind an assertion that the cell count is unchanged; and the `Embedder` streams
chunks to a relative `data/` dir, so each job gets its own scratch cwd or
concurrent datasets collide on filenames.

**What must be said to R3 regardless of the numbers:** the run uses the same
checkpoint the scPRINT-2 paper's own batch-correction figure uses
(`ji9krimq.ckpt`), zero-shot, with upstream's own `pred_embedding` — so it is not a
comparison against a crippled or superseded variant. Upstream's evaluation is
genuinely zero-shot too: `FinetuneBatchClass` is imported in that notebook but never
called.

#### Read the two axes jointly: that is what the metric pair is for

The pair is designed so that neither axis can be satisfied alone. `avg_batch` is
trivially maximised — a structureless embedding mixes batches perfectly, and the nulls
below score up to 1.000 on it. `avg_bio` cannot be gamed that way: both nulls sit at
0.16–0.17 against an analytic floor of 1/6. But the converse trap is just as real:
`avg_bio` *can* be raised by picking up structure that happens to separate cells, and
if that structure is batch, the gain shows up as a loss on `avg_batch`. So neither a
high `avg_batch` nor a rising `avg_bio` is evidence on its own. **Both moving up
together is the only signature that cannot be bought cheaply**, and it is the
criterion this document should be read against.

This is what makes the gene-index repair defensible as a fix rather than a tuning:

| | `_v3` (misaligned) | `_v4` (repaired) | Δ |
| --- | --- | --- | --- |
| `avg_bio` | 0.310 | **0.505** | **+0.195** |
| `avg_batch` | 0.741 | **0.832** | **+0.091** |

Both axes rose on **8 of the 9** datasets. The one exception is `cell_lines`, where
`avg_batch` was already 0.965 against a noise floor of 1.000 — no headroom to rise
into, so it went −0.004. On the five datasets whose `avg_batch` sat low, it gained
0.08–0.20. A misindexed embedding loses on both axes at once and a repaired one gains
on both at once; that joint movement is stronger evidence than the invariant checks
(gene counts, warning counts) recorded elsewhere in this file, because it is the
outcome measure rather than a proxy for it.

Applied to the full table, the joint criterion also settles what can and cannot be
claimed on either side. Nothing here sits on a trade-off frontier by accident:

| Pareto frontier (dominated by nothing) | `avg_bio` | `avg_batch` |
| --- | --- | --- |
| Unintegrated (PCA) | **0.637** | 0.883 |
| MLM MULTITASK | 0.630 | 0.895 |
| WCED MULTITASK | 0.624 | 0.907 |
| BMFM CONCAT | 0.620 | **0.908** |

Those four trade genuinely — ~0.017 of `avg_bio` for ~0.025 of `avg_batch`. Everything
else in the table is *dominated*, beaten on **both** axes simultaneously:

- **scPRINT-2 `ji9krimq` (0.505/0.832) is beaten on both axes by six things**: all
  three BMFM variants, scGPT, scPRINT-1 `large-v1`, and Unintegrated. It therefore
  cannot be described as favouring batch correction over bio conservation, or as
  making any trade at all. It is behind on both.
- **scPRINT-1 `large-v1` (0.577/0.873)** is likewise beaten on both by five.
- **scGPT (0.613/0.880) is beaten on both axes by Unintegrated (0.637/0.883)**, so
  plain 30-dim PCA satisfies the joint criterion better than scGPT does on these nine
  datasets. This is the open presentational question in §7, and the joint reading
  makes it sharper rather than softer: scGPT cannot be defended as trading either.

#### The noise floor, and why `avg_batch` cannot carry a claim on its own

Before any scPRINT-2 number is interpreted: `avg_batch = mean(graph_conn,
ASW_label/batch)`, and **both components are maximised by an embedding that carries
no information at all** — perfect batch mixing is what you get when nothing can be
distinguished. So the column needs a measured null, not an assumed one.

The null used is the **row-shuffled** real embedding (`make_noise_ctrl.py`): identical
per-dimension scales, identical row norms, identical marginal distribution, zero
cell-to-label correspondence. A second, independently constructed null (`Random424`,
iid gaussian row-normalised to the same sphere) is now scored on **all nine**
datasets as a check on the first. The two nulls share nothing but cell count and
scale, so where they agree, the floor is a property of the dataset rather than of how
one null was built. They agree within 0.01 on six of the nine (`pbmc_10k` 0.002,
`cell_lines` 0.003, `human_pbmc` 0.004, `immune_atlas` 0.007, `covid_19` 0.009,
`immune_all_human` 0.009) and within 0.05 on the other three (`heart_atlas` 0.028,
`lung_atlas` 0.043, `pancrm` 0.047). Gaussian noise floors slightly *higher* than a
shuffle everywhere, mean `avg_batch` 0.660 vs 0.643, so the conservative floor is
`max` of the two. Scored by the same scorer, same conventions:

| dataset | avg_bio | avg_batch | ASW_label/batch | graph_conn |
| --- | --- | --- | --- | --- |
| cell_lines | 0.167 | **0.997** | 0.994 | 1.000 |
| covid_19 | 0.147 | 0.538 | 0.986 | 0.090 |
| heart_atlas | 0.164 | 0.636 | 0.886 | 0.385 |
| human_pbmc | 0.165 | 0.715 | 0.977 | 0.453 |
| immune_all_human | 0.163 | 0.586 | 0.962 | 0.211 |
| immune_atlas | 0.159 | 0.539 | 0.956 | 0.122 |
| lung_atlas | 0.161 | 0.487 | 0.891 | 0.082 |
| pancrm | 0.159 | 0.589 | 0.959 | 0.219 |
| pbmc_10k | 0.164 | 0.703 | 0.982 | 0.423 |
| **MEAN** | **0.161** | **0.643** | **0.955** | **0.332** |

Three consequences, all of which apply to the published table and not just to
scPRINT:

- **`ASW_label/batch` is anti-informative in this range.** Noise scores 0.955 mean;
  every real method scores *lower* (Unintegrated 0.821, scPRINT-1 large-v1 0.840,
  scPRINT-2 small-v2 all-facets 0.890). On this component pure noise beats every
  method in the manuscript's table. Within a family, the run with the higher
  `ASW_label/batch` is the run closer to carrying nothing.
- **`graph_conn` is the only discriminating half** (noise 0.332 vs ~0.89-0.94 for
  real methods), and its noise value swings 1.000 → 0.082 by dataset, so it also
  sets how much room `avg_batch` has on each dataset. On `cell_lines` the null
  scores `avg_batch` 0.997 — higher than every published method, whose best is WCED
  at 0.959.
- **`avg_bio` is sound.** Floor 0.161 (shuffle) and 0.172 (gaussian) against the
  analytic 1/6 = 0.167 (NMI→0, ARI→0, ASW_label→0.5); both nulls stay inside
  0.147–0.177 on every dataset. Differences in `avg_bio` mean what they appear to
  mean.

##### How much room `avg_batch` actually has, per dataset

With both nulls scored on all nine, the floor can be subtracted rather than assumed.
Taking the conservative floor `max(Shuffled, Random424)` and the reported `_v4` run:

| dataset | floor | `_v4` `avg_batch` | headroom |
| --- | --- | --- | --- |
| cell_lines | **1.000** | 0.962 | **−0.038** |
| covid_19 | 0.547 | 0.816 | +0.269 |
| heart_atlas | 0.664 | 0.844 | +0.180 |
| human_pbmc | 0.719 | 0.825 | +0.106 |
| immune_all_human | 0.595 | 0.845 | +0.249 |
| immune_atlas | 0.546 | 0.785 | +0.239 |
| lung_atlas | 0.530 | 0.812 | +0.282 |
| pancrm | 0.636 | 0.654 | **+0.019** |
| pbmc_10k | 0.705 | 0.947 | +0.242 |
| **MEAN** | — | **0.832** | **+0.172** |

Two entries matter more than the mean:

- **`cell_lines` has no headroom at all: the gaussian null scores a perfect 1.000.**
  Every method in the manuscript's table scores *below* noise on that dataset
  (best is WCED at 0.959, `_v4` is 0.962). The `avg_batch` column carries no
  information about batch correction on `cell_lines` and should not be read as if it
  does — for any method, ours included.
- **`pancrm` clears its floor by 0.019.** Combined with its `avg_bio` 0.352 — the
  worst of the nine and below every published method — `pancrm` is the one dataset
  where scPRINT-2 is close to indistinguishable from noise on *both* axes. This is
  the same dataset flagged below as a genuine weakness the gene-index repair does not
  explain, and the floor subtraction is independent evidence for that reading.

Everywhere else the repaired run clears its floor by 0.11–0.28, so `avg_batch` 0.832
is a real measurement on seven of nine datasets even though it is the lowest of any
real method.

Keep this subordinate to the joint reading above, though. An earlier draft of this
section treated floor subtraction on `avg_batch` as the primary analysis; it is not.
Because `avg_batch` is the axis that can be maximised for free, subtracting its floor
tells you only whether a number is interpretable, never whether it is good. Its
residual value is narrow and specific: `cell_lines` `avg_batch` is uninformative for
*every* method in the table, and `pancrm` is the one dataset where scPRINT-2 is close
to noise on both axes at once. Ranking, and every claim about capability, comes from
the joint position.

Practical rule for the revision: rank on `avg_bio`, quote `avg_batch` only with the
null beside it, and do not write "method X has the best batch correction". An
earlier draft of this document did exactly that for scPRINT-2 small-v2 (0.891,
"best of any scPRINT run and third overall"); that claim is **withdrawn** — it was
reporting proximity to the noise value, not a capability. `build_final_table.py
--with-null` emits these rows into the table so the floor travels with the numbers.

Diagnostically, this is also what pointed at the wrong-checkpoint mistake: a profile
of high `avg_batch` with depressed `avg_bio` is the signature of a weak embedding,
because the batch half saturates as information is lost.

#### scPRINT-2 `small-v2` results — the WRONG checkpoint, retained as a lesson

These 9-dataset numbers are a correct measurement of `small-v2.ckpt`, which is *not*
the published scPRINT-2. They are kept because they document what the mistake cost
and because the facet-choice finding below is a property of the v2 architecture that
carries over. **Do not cite these as scPRINT-2.** Means over the same 9 datasets;
the two conventions agree to 3 decimals on every mean, and all three runs'
`Unintegrated` controls are *bit-identical* (0.637 / 0.883 / 0.735), an independent
check that the separate v2 venv, lamindb instance and preprocessing path did not
perturb the source data.

| method | avg_bio | avg_batch | weighted |
| --- | --- | --- | --- |
| WCED MULTITASK | 0.624 | 0.907 | 0.738 |
| MLM MULTITASK | 0.630 | 0.895 | 0.736 |
| BMFM CONCAT | 0.620 | 0.908 | 0.735 |
| Unintegrated | 0.637 | 0.883 | 0.735 |
| scGPT | 0.613 | 0.880 | 0.720 |
| scPRINT-1 (large-v1) | 0.577 | 0.873 | 0.695 |
| **scPRINT-2 (all facets)** | **0.540** | **0.891** | **0.680** |
| scPRINT-2 (cell_type facet) | 0.422 | 0.816 | 0.580 |

Findings:

- **The facet choice is the single largest factor for v2**, +0.118 avg_bio and
  +0.075 avg_batch from `cell_type` to `all`, and it helps on all 9 datasets in
  *both* components (avg_bio +0.03 to +0.26, largest on `heart_atlas`; avg_batch
  +0.02 to +0.16). This is the *opposite* sign to
  v1, for the structural reason in the section above (v1 averages, v2
  concatenates). Upstream's own Figure-5 notebook uses the *weaker* of the two.
- **scPRINT-2 small-v2 does not beat scPRINT-1 large-v1** on these 9 (0.680 vs
  0.695), and both are below all four published methods. The deficit is in bio
  conservation (0.540 vs 0.577). Its higher `avg_batch` (0.891 vs 0.873) is *not* a
  point in its favour — see the noise floor above; on `ASW_label/batch` it is the
  closest of the three runs to the noise value of 0.955.
- **What collapses is `ASW_label`, and it is not a pipeline artefact.** It sits at
  0.477-0.556 on *every* dataset; since the metric is `(s+1)/2`, that is a
  silhouette of ≈ 0, i.e. no Euclidean class separation, while NMI/ARI stay at
  0.632/0.476 off the kNN graph. Upstream reports the same range for their own
  checkpoint (notebook cell 20: Silhouette label 0.504-0.612, KMeans ARI 0.225-0.440
  across their four datasets — a *lower* ARI than measured here), so a flat
  `ASW_label` is characteristic of scPRINT-2 embeddings rather than evidence of a
  bug in this pipeline.
- v2's larger vocabulary helps coverage only where the source file is not
  pre-filtered: `cell_lines` fills 87.9% of the human model slots (v1 76.4%), but
  `pbmc_10k` still only 16.3% (v1 14.2%) because that file ships 3,289 genes.
- `pred_embedding=["other"]` cannot be requested on its own -- upstream's
  `_predict` raises `ValueError: 'other' is not in list`, as "other" is only valid
  inside the `["all"]` expansion -- so the per-facet contribution was not isolated.

#### scPRINT-2 `ji9krimq` results (the publication checkpoint)

> **The reportable run is `_v4`.** The first pass at this checkpoint (`_v3`/`_v3a`)
> was invalid: it fed the model gene embeddings from the wrong rows, because
> `embed_one_v2.py` never called the gene-index alignment repair that
> `compute_scprint` applies by default. All 20,004 human genes were misindexed by
> 3-8 positions. See *Root cause: the collator indexes a local gene table that does
> not match the checkpoint*, below. `_v4` is the same two configurations with the
> alignment repaired, and its numbers are the ones to cite. The `_v3` column is kept
> in the table below only so the size and direction of the defect stay on the record;
> the `scPRINT-1 large-v1`, `Unintegrated` and `NULL` columns were never affected
> (verified: the triggering warning fires in 27/32 `_v3` logs, 0/6 `large-v1` logs,
> and **0/9 `_v4` logs**, and every `_v4` log reports `20004 model genes exist`
> where `_v3` reported `19999`).

Run as `_v4` (single `cell_type_ontology_term_id` facet — upstream's config, and the
primary number) and `_v4a` (all facets — a check on the averaging effect, see below).
Configuration is identical to the `small-v2` run except for the checkpoint and the
`expr_emb_style`-driven preprocessing branch, so the comparison between the two
isolates the checkpoint.

##### Results, all 9 datasets

`avg_bio = mean(NMI, ARI, ASW_label)`, `avg_batch = mean(graph_conn,
ASW_label/batch)` — the manuscript's own definitions. Both metric conventions
(the scGPT tutorial's `scib_eval` and this repo's
`calculate_batch_integration_metrics`) were run independently and agree to 3
decimal places on every one of the 9 datasets, so only one set of figures is
quoted; the tutorial convention is the one tabulated, because it is the one that
reproduces the published scGPT column exactly. `NULL` is a row-shuffled embedding on
the same file: the noise floor.

| dataset | scPRINT-2 `_v4` bio | batch | (`_v3` bio, invalid) | scPRINT-1 large-v1 bio | batch | Unintegrated bio | batch | NULL bio | batch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cell_lines | 0.547 | 0.962 | 0.354 | 0.665 | 0.934 | 0.742 | 0.957 | 0.167 | 0.997 |
| covid_19 | 0.447 | 0.816 | 0.336 | 0.452 | 0.850 | 0.440 | 0.764 | 0.147 | 0.538 |
| heart_atlas | 0.591 | 0.844 | 0.225 | 0.659 | 0.936 | 0.722 | 0.909 | 0.164 | 0.636 |
| human_pbmc | 0.480 | 0.825 | 0.351 | 0.546 | 0.822 | 0.632 | 0.887 | 0.165 | 0.715 |
| immune_all_human | 0.482 | 0.845 | 0.320 | 0.537 | 0.889 | 0.549 | 0.893 | 0.163 | 0.586 |
| immune_atlas | 0.490 | 0.785 | 0.316 | 0.548 | 0.827 | 0.647 | 0.859 | 0.159 | 0.539 |
| lung_atlas | 0.517 | 0.812 | 0.278 | 0.541 | 0.902 | 0.666 | 0.904 | 0.161 | 0.487 |
| pancrm | 0.352 | 0.654 | 0.248 | 0.499 | 0.718 | 0.608 | 0.819 | 0.159 | 0.589 |
| pbmc_10k | 0.641 | 0.947 | 0.363 | 0.745 | 0.976 | 0.727 | 0.952 | 0.164 | 0.703 |
| **MEAN** | **0.505** | **0.832** | **0.310** | **0.577** | **0.873** | **0.637** | **0.883** | **0.161** | **0.643** |

The repair gained `avg_bio` on **all 9 datasets with no exceptions**, by +0.104
(`pancrm`) to +0.366 (`heart_atlas`), moving the mean 0.310 → 0.505. `avg_batch`
moved only where it had headroom — flat on `cell_lines` (0.965 → 0.962) and
`human_pbmc` (0.824 → 0.825), both already near saturation, and up 0.08–0.20 on the
five datasets that sat low. That asymmetry is what a real information gain looks like
on a metric that noise maximises; see the noise-floor section.

Against the four published methods recomputed over these same 9 datasets (the
paper's averages include `dc`, which we do not have, so they are not directly
comparable): `avg_bio` means are BMFM CONCAT 0.620, MLM MULTITASK 0.630, WCED
MULTITASK 0.624, scGPT 0.613. So on `avg_bio`, the full ordering is

> Unintegrated 0.637(!) > MLM 0.630 > WCED 0.624 > CONCAT 0.620 > **scGPT 0.613** >
> **scPRINT-1 0.577** > small-v2 all-facets 0.540 > **scPRINT-2 ji9krimq 0.505** >
> small-v2 1-facet 0.422 > NULL 0.161

**The headline for R3 is that scPRINT-1 is roughly competitive (0.577) and scPRINT-2
`ji9krimq` is below it (0.505), with both below the four published methods.** Three
things about that sentence have to be stated precisely, because each is easy to
overclaim in either direction:

- **0.505 agrees with upstream's own level for this checkpoint.** Upstream's scIB
  numbers span 0.467–0.553 with mean 0.491 across their five datasets (below). Our
  9-dataset 0.505 sits inside that range. This is the external check that the `_v4`
  numbers are sound and the `_v3` ones were not: the earlier ~0.18 shortfall against
  upstream is gone, and it closed by agreement with the model's authors rather than
  by our own before/after comparison.
- **large-v1 beats `ji9krimq` on all 9 datasets**, margins 0.005 (`covid_19`) to
  0.147 (`pancrm`). `ji9krimq` is 409.7M parameters against large-v1's 101.1M, but
  84% of that gap is gene-vocabulary tables that are mostly inert on human data —
  the two transformers are 13% apart at identical `d_model` 512 / 16 layers — so this
  is closer to size-matched than the totals suggest. It still carries no scaling
  claim in either direction; see the corrected discussion in §11.
- **`ji9krimq`'s `avg_batch` of 0.832 is the lowest of any real method** in the table
  (0.873–0.908 for the rest). Any reading in which scPRINT-2 is the best batch
  corrector is now contradicted by our own corrected numbers, on top of upstream's
  published iLISI 0.018 / kBET 0.162 / PCR 0.000 for immune_cell_atlas.

`pancrm` is the one dataset where the corrected number is still conspicuously weak:
0.352 against a published band of 0.450–0.647, i.e. below even scGPT's 0.450, which
is itself the weakest published method there — and `Unintegrated` scores 0.608 on the
same file. It also gained the least from the repair (+0.104). So `pancrm` is a
genuine scPRINT-2 weakness that the gene-index defect does not account for, and it
should not be presented as residual bug.

The two features of the old `_v3` column that a reader would have probed are both
resolved by the fix, which is itself evidence the diagnosis was right:

- `_v3`'s `avg_bio` was strikingly **flat**, 0.225–0.363, across datasets whose
  difficulty varies widely. `_v4` ranges 0.352–0.641, tracking dataset difficulty in
  the same order as `Unintegrated` (0.440–0.742) — the profile of a representation
  that responds to how separable the data is.
- On `heart_atlas`, `_v3`'s `avg_batch` 0.640 sat within 0.004 of the noise floor
  (0.636) — indistinguishable from a row shuffle on that column. `_v4` scores 0.844,
  clearly off the floor. `heart_atlas` was flagged in advance as the dataset with the
  most room to move and the least saturated batch metric, and it moved the most
  (+0.366 bio, +0.204 batch).

Integrity cross-check: `Unintegrated` was scored independently inside each of the
five runs (`_lg`, `_v2`, `_v2a`, `_v3`, `_v4`), from the source `obsm` of a freshly-read
`original.h5ad` each time. All five columns agree to 3dp on all 9 datasets, which
confirms no run perturbed the source data.

The recipe was diffed line-by-line against upstream's cell 17 and matches it:
`load_from_checkpoint(..., precpt_gene_emb=None, gene_pos_file=None)`,
`mask_zeros = True`, `Preprocessor(force_preprocess=True, skip_validate=True,
do_postp=metacell)`, `sc.pp.neighbors(use_rep="X_pca")` only when metacell, and
`Embedder(how="random expr", max_len=3200, num_workers=8,
pred_embedding=["cell_type_ontology_term_id"], keep_all_labels_pred=True,
doplot=False)`. Two deliberate departures, both to keep the cell set fixed so the
rows stay comparable to the published ones scored on the same file:
`min_nnz_genes=0, min_valid_genes_id=1, min_dataset_size=1` (upstream keeps the
defaults, whose `min_valid_genes_id=10_000` would reject `pbmc_10k` and `covid_19`
outright — see the gene-panel note below), and `is_symbol` auto-detected rather
than pinned to `False` (upstream's inputs are cellxgene Ensembl-indexed; ours are
mostly symbol-indexed).

`do_postp` is the only other thing that differs between the `small-v2` and
`ji9krimq` runs (True vs False, because it is gated on `metacell`), so it was
checked before attributing any difference between the two to the checkpoint:
`scdataloader/preprocess.py:370-376` writes the normalised matrix to
`adata.layers["norm"]` and adds `obsm["X_pca"]`. It does **not** overwrite `X`.
Both runs therefore hand the model the same raw counts, and the difference between
them is the checkpoint and nothing else.

##### `all facets` averages for this checkpoint, so the all-facets run is a check and not a rival

`cell_emb.py:240-251` branches on `model.compressor`: a `ModuleDict` hstacks the
per-facet embeddings, `None` sums them and divides by the count. `ji9krimq` logs
`compressor=no`, so for it all-facets **averages**, exactly like large-v1 — it does
*not* concatenate the way `small-v2` does. The earlier note that "v2 concatenates"
was true of `small-v2` and does not generalise to the v2 checkpoint that matters.
Single-facet is right for `ji9krimq` because it is the checkpoint's own recorded
`hyper_parameters["pred_embedding"]` and upstream's configuration; the all-facets run
`_v4a` was done to measure the effect rather than assert it.

Measured on all 9 datasets with the gene index repaired, **the facet choice is a wash
on the headline**:

| dataset | `_v4` bio | `_v4a` bio | Δ | `_v4` batch | `_v4a` batch | Δ |
| --- | --- | --- | --- | --- | --- | --- |
| cell_lines | 0.547 | 0.551 | +0.004 | 0.962 | 0.942 | −0.019 |
| covid_19 | 0.447 | 0.477 | +0.030 | 0.816 | 0.804 | −0.012 |
| heart_atlas | 0.591 | 0.554 | −0.037 | 0.844 | 0.902 | +0.059 |
| human_pbmc | 0.480 | 0.509 | +0.029 | 0.825 | 0.574 | **−0.251** |
| immune_all_human | 0.482 | 0.472 | −0.011 | 0.845 | 0.839 | −0.006 |
| immune_atlas | 0.490 | 0.506 | +0.015 | 0.785 | 0.829 | +0.044 |
| lung_atlas | 0.517 | 0.515 | −0.003 | 0.812 | 0.831 | +0.019 |
| pancrm | 0.352 | 0.353 | +0.002 | 0.654 | 0.665 | +0.011 |
| pbmc_10k | 0.641 | 0.620 | −0.020 | 0.947 | 0.977 | +0.029 |
| **MEAN** | **0.505** | **0.506** | **+0.001** | **0.832** | **0.818** | **−0.014** |

The `avg_bio` means differ by 0.001, with signs mixed 5 up / 4 down, so averaging the
facets neither dilutes nor enriches the cell-type signal in any systematic way — and
either way it is nowhere near the ~0.11 gap to the published methods, which is the
question the run was there to answer. `avg_batch` is likewise mixed 4 up / 5 down;
its −0.014 mean is carried entirely by one dataset, `human_pbmc` at −0.251, and
excluding that dataset the mean Δ is **+0.016**. A plausible reading of `human_pbmc`
is that the extra readouts (`assay`, `disease`, `ethnicity`, `sex`, `tissue`) are
confounded with its batch variable, so folding them in encodes batch structure; but
that is a per-dataset conjecture, not a general mechanism, since `heart_atlas` and
`immune_atlas` move the *other* way by +0.059 and +0.044.

Both runs are now scored under **both** conventions, and the two independent scorers
agree to within **0.0005** on every one of the 18 `_v4a` comparisons (`_v4a` means:
tutorial 0.5063/0.8180, paper 0.5062/0.8181; `_v4`: 0.5054/0.8322 and 0.5052/0.8322).
So the `_v4` vs `_v4a` conclusion does not rest on one scorer.

That also settles what `human_pbmc` is doing, and it is worth more than the mean. Its
noise floor there is 0.719 (shuffle 0.715, gaussian 0.719 — the two nulls agree to
0.004 on that dataset). `_v4` scores 0.825, clearly above it. **`_v4a` scores 0.574,
which is 0.146 *below* the floor**, and the paper convention independently returns
0.574 as well. An embedding can only mix batches *worse* than pure noise by carrying
structure that actively tracks batch, so on `human_pbmc` the all-facets embedding is
encoding batch as signal rather than failing to remove it. `_v4a` is below its floor
on two datasets (`human_pbmc` −0.146 and `cell_lines` −0.057, the latter unavoidable
since that floor is 1.000); `_v4` is below only on `cell_lines`. That is a cleaner
reason to report `_v4` than "the mean is a wash", and it does not depend on the
per-dataset conjecture above.

Two corrections this table forces, recorded because both were stated in this file
earlier and both were wrong:

- The claim that all-facets averaging **dilutes** the cell-type signal was based on
  `_v3a` losing 0.072 on `cell_lines` (0.354 → 0.282). With the gene index repaired
  the same comparison *gains* 0.004. That dilution effect was an artefact of the
  misalignment: averaging facets computed from wrong embedding rows compounds the
  error, averaging correctly-indexed ones does not.
- A three-dataset reading of the repaired numbers looked like a clean "bio up, batch
  down" trade with a tidy confounding story behind it. It inverted on the fourth
  dataset (`heart_atlas`). Four points were not enough to name a mechanism, and the
  nine-dataset answer is simply that the effect is small and dataset-dependent.

##### The embedding is 256-dim although `d_model=512`, and that is correct

The checkpoint is two-width: the gene side is 512 (`gene_encoder.0.embeddings.weight`
335182x512, `pos_encoder.pe` 335182x1x512, `expr_encoder.encoder.0.weight` 512x1)
while the cell side is 256 (`class_encoder.embedding.weight` 10x256,
`cell_transformer.blocks.0.mixer.out_proj.weight` 256x256). Since
`output_cell_embs = cell_embs` when the compressor is None
(`model.py:1216-1218`), the per-facet cell embedding is the cell-transformer width,
256. So a 256-dim output from a `d_model=512` checkpoint is architectural, not a
truncation — worth recording because "the embedding is half the model width" reads
like a bug and would otherwise get chased twice.

##### Upstream's own numbers for this checkpoint corroborate the level

This is the load-bearing external check, because it is upstream's code, upstream's
checkpoint, and a full-panel dataset. Notebook cell 17's scIB output for
`cellxgene_census/immune_cell_atlas`, embedding key `scprint_emb`:

| metric | ji9krimq (upstream's own run) |
| --- | --- |
| KMeans NMI | 0.596 |
| KMeans ARI | 0.341 |
| Silhouette label | 0.535 |
| Isolated labels | 0.497 |
| cLISI | 0.991 |
| Graph connectivity | 0.856 |
| iLISI | 0.018 |
| kBET | 0.162 |
| PCR comparison | 0.000 |
| Bio conservation (their aggregate) | 0.592 |
| Batch correction (their aggregate) | 0.342 |
| Total | 0.492 |

(That table is cell 17's *stored* output. `immune_cell_atlas` is commented out in
cell 13's current state — "issue with scib (too large?)" — so the notebook as
shipped would not regenerate it; the output is from an execution when it was
enabled, and cell 17's own stdout confirms it with `doing
cellxgene_census/immune_cell_atlas`.)

Cell 20 hard-codes the same panel for the four datasets the notebook still runs, so
upstream's numbers are available for **five** datasets in total. On the manuscript's
`avg_bio = mean(NMI, ARI, ASW_label)`:

| upstream dataset | NMI | ARI | ASW_label | **avg_bio** |
| --- | --- | --- | --- | --- |
| mouse_pancreas_atlas | 0.656 | 0.440 | 0.563 | **0.553** |
| immune_cell_atlas | 0.596 | 0.341 | 0.535 | **0.491** |
| dkd | 0.517 | 0.354 | 0.548 | **0.473** |
| hypomap | 0.570 | 0.225 | 0.612 | **0.469** |
| gtex_v9 | 0.617 | 0.281 | 0.504 | **0.467** |
| MEAN | | | | **0.491** |

**This is where the honest reading changes, and it is worth being blunt about it.**
Upstream's range is 0.467–0.553. Ours is 0.225–0.363. The two ranges **do not
overlap** — our best dataset scores below their worst. So upstream's numbers
establish two things, and only two: (i) this checkpoint is *not* a strong performer
on this metric family even in its authors' hands, well below the ~0.70 the published
methods and `Unintegrated` reach; and (ii) there is nevertheless a systematic ~0.18
gap between their runs and ours that these numbers do **not** explain.

An earlier version of this section stopped at (i) and treated ~0.5 as corroborating
our ~0.31. That was too generous to our own pipeline: 0.31 is not 0.49.

The same applies to the label head. Cell 19 records upstream's own zero-shot
`ref_cls` — the identical argmax-over-`CL:`-columns readout used in the diagnostic
below — at accuracy 0.605 / macro F1 0.459 on `dkd` and 0.569 / 0.356 on `gtex_v9`.
Macro F1 in the 0.36–0.46 range is not reachable by majority-class guessing, so
their head genuinely discriminates. Ours returns AMI 0.102 on `pbmc_10k`. That is a
second, independent instance of the same gap.

The candidate explanations considered at the time were: the cell-quality gates
(upstream keeps `min_nnz_genes=200, min_valid_genes_id=10_000`, we set them to
0/1/1 to hold the cell set fixed — and `covid_19` averages 126 non-zero genes per
cell, `pbmc_10k` 452, so upstream's own gate would discard much of what we feed the
model); panel width and Ensembl-vs-symbol indexing; and the NMI/ARI clustering
convention (upstream's scIB uses KMeans at `n_clusters = n_labels`, our conventions
use Leiden at optimised resolution). All three would have had to be *large* to
account for 0.18, and none of them is the cause. **The cause has since been found
and it is none of these** — see the root-cause section below.

The experiment that discriminates between "our pipeline is faithful" and "our
pipeline is the problem" is to run our code on their data
(`embed_upstream_ds.py` / `run_embed_up.sh`: `dkd`, 39,176 cells x 35,469 genes,
the only one of the four small enough to be convenient at 403 MB). **It has now
run, and it settles the attribution against us.** With upstream's own
`Preprocessor` defaults restored and `is_symbol=False` pinned, on upstream's own
`dkd`, with upstream's own checkpoint, our pipeline reproduces the failure: 17
distinct cell-type calls over 39,176 cells, 23,699 of them `CL:4023056`
(*vascular leptomeningeal cell (Mmus)*), AMI 0.0374 / ARI -0.0164 — where
upstream reports accuracy 0.605 / macro F1 0.459 on that exact dataset and
checkpoint. The preprocessing itself was clean (`39176 -> 39176` cells, 100%
kept, 85,720 genes) and the embedding was non-degenerate (39,176 unique rows, no
dead dims). Same data, same recipe, same checkpoint: **the failure travels with
our environment, not with the data.** So the ~0.18 gap is not a property of the
manuscript's datasets, and the numbers above cannot be attributed to the model.

**And it cuts against the batch-correction reading directly.** Upstream's richer
batch panel for the same embedding is iLISI 0.018, kBET 0.162, PCR 0.000, batch
aggregate 0.342 — i.e. by every batch metric that is *not* saturated by
structurelessness, scPRINT-2 barely mixes batches at all. That is independent
confirmation of the noise-floor finding above: a high `mean(graph_conn,
ASW_label/batch)` for this embedding is metric saturation, not batch correction.
Any sentence of the form "scPRINT-2 has strong batch correction" is contradicted by
upstream's own table.

##### The zero-shot label head does not merely underperform — it scores exactly zero

A low `ASW_label` on its own cannot distinguish "the model is being fed badly" from
"this checkpoint's cell-type facet is weakly structured", so the model's *own*
cell-type call was read out as a diagnostic (`embed_one_v2.py --save-preds`, off by
default so the benchmark files keep one schema). `keep_all_labels_pred=True` leaves
one logit column per ontology term in `obs`; the call is the argmax over the `CL:`
block, which is the idiom `model.py:1326` documents
(`output["cls_output_cell_type_ontology_term_id"].argmax(-1)`).

The column naming was verified against the tensor rather than assumed, because a
block misalignment would produce exactly the symptom below and would *not* be
caught by AMI/ARI (those are invariant to relabelling, but not to reading the wrong
block). `model.py:2534-2545` builds `self.pred` as `torch.cat([... for clsname in
self.classes], dim=1)`, and `cell_emb.py:262-266` names the columns with
`for cl in model.classes` — the same iteration order, so the `CL:` block is the
cell-type block.

Run on `pbmc_10k`, chosen because 10x PBMC is as in-distribution as a
cellxgene-trained model gets, so an OOD excuse cannot apply:

| true label | mast cell `CL:0000097` | `CL:1000695` | `CL:0000525` | `CL:0002629` | other 5 |
| --- | --- | --- | --- | --- | --- |
| CD4 T cells | 3391 | 439 | 520 | 75 | 571 |
| CD8 T cells | 1088 | 101 | 127 | 13 | 119 |
| CD14+ Monocytes | 923 | 401 | 319 | 520 | 64 |
| B cells | 571 | 575 | 268 | 181 | 26 |
| NK cells | 207 | 136 | 18 | 41 | 55 |

Only **9 of ~500** cell-type terms are ever the argmax; 6,509 of 11,990 cells
(54%) are called *mast cell*, which is essentially absent from peripheral blood;
and the top call is spread across the true types roughly in proportion to their
abundance (67% of CD4 T, 75% of CD8 T, 41% of monocytes). Against the true labels
the calls score **AMI 0.102, ARI 0.110** — barely above chance. Top logits sit in a
narrow band (-1.88 to -0.49), which is itself not the profile of a confident head.
`cell_lines` behaves the same way (7 distinct calls, and `CL:4023056` takes 2,838
of 293t and 2,896 of jurkat — near-identical across two utterly different cell
lines).

**AMI/ARI understate this, and the correct metric is worse.** The head's 500
cell-type outputs are the *leaves* of the CL hierarchy: `label_decoders
['cell_type_ontology_term_id']` has 1032 entries, `labels_hierarchy` has 531, and
the partition is clean — every ancestor key is >= 500 and every descendant index is
< 500. So indices 0-499 are what the head can emit and 500-1030 are ancestor-only.
Every coarse label our sceval data uses is ancestor-only (T cell 526, endothelial
534, erythrocyte 563, macrophage 564, B cell 565, DC 575, neuron 584, monocyte 588,
NK 596, CD4 T 597, CD8 T 598, leukocyte 611), so a *hierarchical* scorer — the one
upstream uses, which credits a leaf call whose ancestors include the true term — is
the right instrument. Scored that way on `immune_all_human`: **accuracy 0.0000, 0
of 33,506 cells, all 16 types at 0.0%.** (An intermediate note in this file briefly
concluded from the leaf/ancestor structure that the AMI/ARI readout was simply the
wrong metric and the finding was an artifact. That retraction was itself wrong: the
hierarchical score is zero, so the finding is stronger than the original framing,
not weaker.)

The readout was verified rather than assumed at every step, because a
misalignment there would produce exactly this symptom. `head_out == label_counts`
for all nine facets and the widths sum to 1113, which is the width of `self.pred`,
so column naming from `range(label_counts[cl])` is exact. `model.py:2483-2490`
builds `ind = {"other": 0}` then `self.classes.index(i) + 1`, and
`model.py:1220-1227` reads `output["input_cell_embs"][:, i + 1, :]`, against a
`class_encoder = CategoryValueEncoder(len(self.classes) + 1, ...)` — the `+1`s
agree, so facet extraction is correct. Upstream's own readout (notebook cell 17) is
character-for-character the same argmax over `CL:`-prefixed columns.

##### Root cause: the collator indexes a *local* gene table that does not match the checkpoint

The gene index handed to scPRINT-2 is not an identifier — it is a **row position**
in a gene table loaded from our local lamindb instance at runtime. That table does
not reproduce the checkpoint's gene list, so nearly every human gene is read from
the wrong row of the frozen gene-embedding matrix. The chain, all in installed
library code:

1. `scprint2/tasks/cell_emb.py:111-118` constructs `Collator(organisms=...,
   valid_genes=model.genes, ...)` with **no `genedf` argument**.
2. `scdataloader/collator.py:117` therefore falls back to `genedf =
   load_genes(self.organisms)` — the local bionty/lamindb table.
3. `collator.py:133-136` computes `tot = genedf[genedf.index.isin(valid_genes)]`
   and `start_idx[org] = np.where(tot.organism == organism)[0][0]`, after a bare
   `print("Some valid genes are not in the genedf!!!")` on line 130 — the only
   signal that anything is wrong. Nothing raises.
4. `collator.py:290` emits `gene_locs.append(loc + self.start_idx[organism_id])`,
   where `loc` is a position within the *local* table.
5. `model.py:859-869` would rebuild `hp["genes"]` from the local table and
   reconcile the two, but the branch is guarded by `if type(...) is list` and
   `hp["genes"]` is a **dict**, so it is skipped and `model.genes` stays the
   checkpoint's own list.

`load_genes` sorts on `ensembl_gene_id` and drops a hardcoded `DROP` set, so the
ordering is deterministic given the gene *set* — which means any missing gene
silently shifts every position after it.

The measured misalignment. The checkpoint's 16 gene lists total **335,182**,
exactly the rows of `gene_encoder.0.embeddings.weight` (335182x512) and
`pos_encoder.pe` (335182x1x512). The local table has 552,495 rows, of which `tot`
= **335,174** — **8 genes short**:

| organism | ckpt_n | local_n | ckpt block offset | local block offset | intra-block position agreement |
| --- | --- | --- | --- | --- | --- |
| NCBITaxon:10090 (mouse) | 21550 | 21547 | 0 | 0 | **55.12%** |
| NCBITaxon:9606 (human) | 20004 | 19999 | 246612 | 246609 | **38.74%** |
| the other 14 | identical | identical | — | shifted 3-8 | 100.00% |

The missing genes are `ENSMUSG00000040715`, `ENSMUSG00000040838`,
`ENSMUSG00000116048` (mouse, at intra-block positions 11876 / 11921 / 21382) and
`ENSG00000139656`, `ENSG00000148362`, `ENSG00000169789`, `ENSG00000169807`,
`ENSG00000290183` (human, at 7747 / 8953 / 12434 / 12436 / 19959).

Human is the case that matters here, and it is misaligned in **both** ways at once.
The three missing mouse genes shift the whole human block by -3, so *every* human
gene is read from the wrong embedding row; on top of that, the intra-block shift is
0 up to position 7747 and then -1..-5 for the remaining 12,252 genes. Net: all
20,004 human genes are off, by 3 for the first 39% and by 4-8 after. The model
receives real expression values attached to systematically wrong gene identities —
which is precisely the input that would produce a low-rank, cell-type-blind
embedding and a label head that fires on unrelated tissues.

**The v1 control makes this an attribution, not a hypothesis.** The
`load_genes` warning fires in **27 of 32** v3-era LSF logs — including the `dkd`
upstream-config run — and in **0 of 6** checked `large-v1` logs. Same environment,
same `original.h5ad`, same `Preprocessor`, same scorers: the checkpoint whose gene
set our local table fully covers scores `avg_bio` **0.745** on `pbmc_10k` (the
second-highest of any method in the table); the one whose gene set it does not
cover scores **0.31**. That is the difference between the two runs.

It is also visible in the returned embedding itself. After centering, `_v3`
embeddings carry `||mean||^2/total` of 0.993-0.9995 — almost all the variance is a
single constant offset — and effective rank 3.4-9.2. Per dataset (PC1 fraction /
effective rank): `cell_lines` 0.421/9.2, `covid_19` 0.640/4.9, `heart_atlas`
0.740/3.4, `human_pbmc` 0.409/8.8, `immune_all_human` 0.466/6.4, `immune_atlas`
0.557/6.1, `lung_atlas` 0.532/6.1, `pancrm` 0.669/4.2, `pbmc_10k` 0.735/3.4.
On the one file measured for all three runs, `cell_lines`, `large-v1` reaches
effective rank **19.5** (`||mean||^2/total` 0.981, PC1 0.345) against `_v3`'s 9.2.

**Effective rank is a proxy and it must not be leaned on.** Measured after the fix,
`cell_lines` `_v4` comes out at effective rank **5.5** (PC1 0.517) — *lower* than
the misaligned `_v3`'s 9.2. A scrambled input can easily spread variance over more
directions than a coherent one, so this is not evidence against the fix; it is
evidence that the spectrum does not settle the question either way. The `avg_bio`
scores do, and only they should be quoted. (Note also that the `large-v1` figure is
per-dataset: 19.5 on `cell_lines` is not comparable to a rank measured on a
different file.)

This single defect accounts for every symptom coherently: the collapsed effective
rank, `avg_bio` pinned flat at 0.225-0.363 regardless of panel width or sequencing
depth, hierarchical label-head accuracy of exactly zero, cell-type calls drawn from
unrelated tissues, and the reproduction of all of it on upstream's own dataset.

##### Consequence for the revision, and the fix

**The `_v3` and `_v3a` rows above must not be reported as scPRINT-2 performance.**
They measure our environment's gene table, not the model. The ~0.18 gap to
upstream's 0.491 is explained and is ours. Everything that was ruled out along the
way stays ruled out (correct checkpoint, column naming, facet extraction,
leaf/ancestor structure, genuine raw counts on all nine files, `max_len`,
`use_raw`, `do_postp`, embedding degeneracy), which is what leaves the gene table
as the answer rather than one candidate among many.

**The most uncomfortable part of this finding is that the bug was already known,
already fixed, and already tested — in the scPRINT-1 path only.** §7 above records
it under "Retired Ensembl IDs misalign the gene index", with three helpers in
`baselines.py` (`_scprint_index_drift`, `_register_retired_genes`,
`_ensure_scprint_gene_alignment`) that `compute_scprint` calls by default
(`align_ontology=True`) before any GPU work, and five tests in
`test_scprint_baseline.py` covering it. `embed_one.py` reaches that code because it
loads `baselines.py` by path; **`embed_one_v2.py` was written standalone for the
`scprint2` package and never called it.** That is the whole mistake. It also
explains the 27/32-versus-0/6 warning split exactly: the v1 runs were repaired, the
v2 runs never were. The mechanism is not new, only its second victim is.

The fix, applied:

1. `baselines.py` — the three helpers now accept the checkpoint's gene vocabulary
   either as a flat sequence (scPRINT-1) or as a per-organism mapping (scPRINT-2),
   via a new `_genes_by_organism`. The mapping form is *required* at 16 organisms:
   the old prefix attribution would file chicken's `ENSGALG*` genes under human,
   because `"ENSGALG".startswith("ENSG")`. All 26 existing tests still pass.
2. `embed_one_v2.py` — calls `_ensure_scprint_gene_alignment(model._genes,
   model.organisms)` before `model.to(device)`, with `--align-ontology
   fix|check|off`. It passes the per-organism dict and *all 16* organisms, because
   the three missing mouse genes displace all 20,004 human genes by themselves.
3. Ran the repair against `anonymous/scprint2db`: 8 gene IDs registered,
   `_ensure_scprint_gene_alignment` re-verified zero drift on all 16 organisms, and
   `tot` is now **335,182 rows — exactly the embedding table**. The Collator's own
   arithmetic was then reproduced independently, comparing both the intra-block gene
   order and `start_idx` against the checkpoint: **16/16 organisms exact on both**
   (before the repair, 2 organisms were misordered and 15 block offsets were
   wrong). Registration is
   additive and idempotent, and the placeholder rows are inert: no real dataset
   carries a retired ID, `Preprocessor` zero-pads them, and `how="random expr"`
   samples only from `np.where(expr > 0)`, so a zero column can never be selected.
   They change gene *indexing* only, never model input.

**The fix is confirmed in the metric, not merely in the invariants.** All nine
datasets were re-embedded as `_v4` and re-scored: `avg_bio` rose on every one of the
nine, mean 0.310 → **0.505**, and the largest single gain (+0.366) landed on
`heart_atlas`, the dataset flagged in advance as having the most room and the least
saturated batch metric. 0.505 falls inside upstream's own 0.467–0.553 range for this
checkpoint, so the ~0.18 gap this section was written to explain is closed. Full
table under *Results, all 9 datasets*, above.

One methodological caution, recorded because it cuts against the fix and was nearly
leaned on as evidence for it: **the effective-rank proxy moved the wrong way.** On
`cell_lines`, `_v3` scored eff_rank 9.2 and `_v4` scored 5.5 — lower — while `avg_bio`
rose 0.354 → 0.547 on the same file. `large-v1` scores 19.5 there. The spectrum
settles nothing in either direction and only `avg_bio` should be quoted.

Two claims that must **not** be revived by this: the withdrawn "scPRINT-2 has the
best batch correction" claim stays withdrawn — upstream's own iLISI 0.018 / kBET
0.162 / PCR 0.000 is independent of our bug, and the corrected `_v4` numbers now
agree with it, since `avg_batch` 0.832 is the *lowest* of any real method in the
table — and the superseded `small-v2` `_v2`/`_v2a` runs still must not be cited as
scPRINT-2.

Caveats that survive regardless: upstream's evaluation of this checkpoint is zero-shot
(`FinetuneBatchClass` is imported but never called), and it scores the single-facet
256-dim embedding via `embedding_key="scprint_emb"`, so `_v4` is the like-for-like
configuration.

On size-matching, an earlier version of this paragraph said `ji9krimq` (409.7M) is
"roughly 4x large-v1 (101.1M)" and that the comparison therefore carries no scaling
claim. **That was too generous to `ji9krimq` and is corrected here.** Counted by
module (see §"Flavours, parameters and representation widths"), 84% of that 4.05x gap
is gene-indexed lookup tables, not capacity:

- both are `d_model` 512 with 16 layers;
- main transformer 56.76M (`ji9krimq`) vs 50.44M (`large-v1`) — **13% apart**, or 27%
  counting `ji9krimq`'s extra 7.11M `cell_transformer`;
- the rest of the gap is the gene table growing 7.5x, 44,756 → 335,182 rows, to cover
  16 organisms instead of 2. **All nine datasets here are human**, so the large
  majority of those rows are never indexed in these runs.

So the capacity actually brought to bear on this benchmark is much closer to matched
than the totals imply, and `large-v1` still outscores `ji9krimq` on all nine datasets.
State the narrow version: total parameters differ 4x, transformers differ 13%, and the
difference is mostly a multi-organism vocabulary that is inert on human data. The one
asymmetry running the other way is representation width — `ji9krimq` emits 256 dims
against `large-v1`'s 512 — but width is not the binding constraint on this benchmark,
since 30-dim PCA `Unintegrated` has the highest `avg_bio` in the table.

Gene-panel note, now a *secondary* limit behind the gene-table defect but still
worth keeping for the re-run: upstream
evaluates on cellxgene datasets carrying full transcriptomes, and its own
`Preprocessor` default (`min_valid_genes_id=10_000`) refuses anything smaller. Our
9 datasets range from 1,200 genes (`covid_19`) and 3,289 (`pbmc_10k`) to 33,694
(`human_pbmc`); five are above 15k and four are not. With `max_len=3200`, the
narrow-panel datasets are outside the regime upstream ever measures, so per-dataset
scPRINT-2 numbers on those should be read as "this is what the published recipe
yields on the manuscript's datasets", not as a claim about the model's ceiling.
`cell_lines` (32,738 genes) is the control for this: it is full-panel and one of the
easiest datasets in the suite, so it is the row where a panel-width excuse does not
apply.

## 8. Untracked files to leave alone

Unrelated to this work, do not commit:
`bmfm_targets/evaluation/benchmark_configs/data_module/bmmc_xsite.yaml`,
`bmfm_targets/tokenization/resources/scconcept_panels.py`,
`bmfm_targets/tokenization/resources/scconcept_panels/`.
