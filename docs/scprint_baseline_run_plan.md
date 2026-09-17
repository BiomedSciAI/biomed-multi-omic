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

**Done 2026-09-16, corrected 2026-09-17.**

- Current results: `$WORK/scprint_large_v1_vs_published.csv`, with per-dataset
  scorer output in `$WORK/metrics_tutorial/` and `$WORK/metrics_paper/`
  (`*_lg_*` = `large-v1`) and embeddings in `$WORK/emb/` (`*_lg_scprint.h5ad`).
- Superseded: `$WORK/scprint_metrics.csv` and `$WORK/metrics/` — convention (1) on
  `medium-v1.5`. Left in place, not cited. The published
  `baseline_metrics_all.csv` was never written to, and no `original.h5ad` was ever
  opened for write.
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

## 8. Untracked files to leave alone

Unrelated to this work, do not commit:
`bmfm_targets/evaluation/benchmark_configs/data_module/bmmc_xsite.yaml`,
`bmfm_targets/tokenization/resources/scconcept_panels.py`,
`bmfm_targets/tokenization/resources/scconcept_panels/`.
