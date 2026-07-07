# wced dataloader profiling — instructions for a fresh session

**Status:** untracked scratch doc. Do not commit.
**Goal:** find why the `scconcept_ablation_wced` arm is ~7× slower per epoch than the
vanilla wced run, and whether it's a fixable (output-neutral) data-pipeline inefficiency.

---

## RESOLVED (2026-07-04) — root cause found & fixed

**Verdict: FIXABLE & OUTPUT-NEUTRAL.** It is *not* the WCED masker (the doc's prime
suspect #1 was wrong — the masker tokenize+label-scatter is <5% of dataloader CPU).

**Root cause:** `dropout_random` (`sample_transforms.py:483`) is O(n²). `dropout_indices`
was built as a **list**, and the reconstruction comprehension does `i not in dropout_indices`
for every element — a linear scan per element. With `expose_zeros: all` and no
`max_mfi_data_len` set (so `truncate_sequence` never runs before dropout in the collator's
input `transform_inputs`, `multifield_collator.py:278`), this runs over the full **~19,071
genes** × 2 fields × 64 cells. The 1024 cap only happens later at HF-tokenizer truncation.

**Evidence (py-spy on live worker, LSF 7991, host b05-02, 2999 samples):**
- `dropout_random:483` = **82.2%** of dataloader-worker CPU. All 3 sampled workers caught there.
- Everything else is noise: HF `_batch_encode_plus` 2.4%, `select_*_labels` ~2%, `mask_inputs`
  0.3%, `_batchwise_pad_zero` ~0.5%, `get_genes_expressed_in_batch` 0.2%.
- Standalone micro-bench @ 19,071 genes: current list impl = **60.9 CPU-s / 64-cell batch**;
  set-based fix = **0.223 s** → **273× faster**, output **bitwise-identical** (verified with
  a fixed RNG seed: same random draw order, membership result unchanged).

**DEEPER root cause (why vanilla wasn't slow):** vanilla never ran dropout at all. Its config
used `sequence_dropout_ratio: 0.2` — **not a real parameter** (the real name is
`sequence_dropout_factor`). It fell into `DataModule.__init__(**kwargs)` (`data_module.py:49`,
"to enable stashing hydra config settings") and was silently ignored. The ablation configs use
the correct key `sequence_dropout_factor: 0.2`, so dropout_random actually runs. So the 7x is:
the ablation turned on an augmentation vanilla never had, and it happens to be O(n²) at 19k genes.
(Confirmed: `grep sequence_dropout_ratio` → 0 hits in code; `has sequence_dropout_factor: False`
in the vanilla ClearML config `2dbd5a10023b4a61be7321bf91c63b9c`.)

Note the arm-specific severity: `wced` has `expose_zeros: all` → dropout over ~19k genes (fatal).
`both`/`contrastive` have `expose_zeros: null` → dropout over only expressed genes (~1-3k),
sub-second — which is why `both` (7992) completed fine.

**Fixes applied:**
1. `sample_transforms.py:477-482`: `dropout_indices = { ... }` (set, not list). O(1) membership →
   O(n). Output bitwise-identical (verified, fixed RNG seed). Latent-bug guard for any future
   expose_zeros=all + dropout run.
2. `run/scconcept_ablation_wced.yaml`: removed `sequence_dropout_factor: 0.2` to match vanilla
   (which had no dropout). This alone eliminates the 7x — dropout never runs. Left a comment to
   re-add if the augmentation is actually wanted. **`scconcept_ablation_both.yaml` NOT changed**
   (dropout is cheap there, run already completed, genuine augmentation-design question) — but
   consider removing for arm consistency.

**Heads-up for the running job:** 7991's workers already imported the old code AND the old config,
so it won't speed up without a **kill + restart** (editable install → both fixes are live for any
new run).

Original investigation notes below (kept for reference).

---

## The problem (one paragraph)

The wced ablation (`expose_zeros: all`, WCED reconstruction of all ~19,071 genes) is
**data-loading bound**: the running job (LSF 7991, host b05-02) does ~**908 steps/h** at
batch 64 (~58k cells/h) with **~28 CPU cores pegged and the GPU idle** (3.13M CPU-s over
~31 h wall). The vanilla wced-only run (`llama_medium_wced_h5ad_nolabels`, ClearML task
`2dbd5a10023b4a61be7321bf91c63b9c`, **`AnnCollectionDataModule`**) did 11 epochs in **11.6 h**
(~400k cells/h). Same data, same `limit_genes: protein_coding`, same `expose_zeros: all`,
same `WCEDMasker`. So the gap is the **new `DataModule` vs `AnnCollectionDataModule`** data path.
Per step the workers burn ~110 CPU-s; we need to know exactly which call eats it.

## Already ruled out (by reading + micro-benchmark)

- **Densify + `get_genes_expressed_in_batch`**: cheap in isolation (~50 ms + ~75 ms per
  64-cell batch). Not the ~110 CPU-s/step. (`base_rna_dataset.py:309` densifies each cell to a
  19,071-elem Python list via `binned_data[idx].toarray().tolist()[0]`; AnnCollection does the
  *same* at `anncollection_dataset.py:224,327`.)
- **`np.isin` in batch-wise pad-zero**: ~12 ms/cell over ~18k gene strings. Not it.
- **Branch patches**: the only branch-unique commit touching the batch-wise pad-zero code is
  the collation-crash fix (`110731f`, benign). `913d4cf` ("Fix tokenizer padding bug in
  `_replace_tokenized_ids_with_continuous_values`") is O(seq_len) both before/after AND it
  *fixed a crash* (shape mismatch at max_length>64 with expose_zeros padding) — so pre-patch
  this config didn't even run. Not the regression.
- **Note:** earlier micro-benchmarks were confounded by ~60–90 s `bmfm_targets` import time;
  ignore any timing that imported the package inside the timed region.

## Leading hypotheses (test these)

1. **WCED label build tokenizes all ~19k genes/cell.** `WCEDMasker.mask_inputs`
   (`masking/strategy.py:225-237`) runs a **second** full `transform_inputs` over the
   *pre-transformed* MFI (`batch["mfi"]`, all ~19k genes, `max_length: None`,
   `pad_zero: None`) and then HF-tokenizes every cell's ~19k gene strings
   (`is_split_into_words=True, truncation=False, padding="longest"`) → a `[64 × ~19k]` tensor,
   plus a per-sample × per-label-set scatter loop (`LABEL_SET_FUNCTIONS`, 3 sets). This is the
   prime suspect for the ~110 CPU-s. **But it is shared with vanilla** — so if this dominates,
   the question becomes why AnnCollection's path is cheaper (different collator? cached gene
   token ids? vectorized label build?).
2. **Double transform pass**: `transform_inputs` runs in the collator (`multifield_collator.py:278`,
   input, capped 1024) *and* in the masker (`strategy.py:225`, labels, uncapped ~19k). Confirm
   whether the uncapped pass re-does expensive work.
3. **Densify-to-Python-list roundtrips**: each transform re-does `np.array(19071-list)`; ~211 ms
   for 4 roundtrips × 64 cells. If the chain has many transforms, this adds up.
4. **Is the input actually capped at 1024?** Confirm the tokenized *input* seq dim is ~1024, not
   ~19k (if pad-zero isn't limiting, everything downstream explodes).

## Method 1 — py-spy on the running worker (fastest, most accurate; do this first)

The job is running; sample it live, no rebuild.
```bash
# on the login node, find the exec host
bjobs -o "jobid stat exec_host" 7991
# ssh to the exec host (e.g. zu-a100-b05-02), find the python dataloader workers
ssh <exec_host> 'pgrep -af bmfm-targets-run | head'
# if py-spy is available (pip install py-spy --user if not):
py-spy top --pid <a_dataloader_worker_pid>          # live hot functions
py-spy dump --pid <worker_pid>                        # stack snapshot
# or a flamegraph over 30s:
py-spy record -o /tmp/wced_worker.svg --pid <worker_pid> --duration 30
```
Look for whether time is in: the HF tokenizer (`lookup_field_tokenizer`), `transform_inputs`,
`get_genes_expressed_in_batch`, `_replace_tokenized_ids_with_continuous_values`, `toarray/tolist`,
or the `LABEL_SET_FUNCTIONS` scatter.

## Method 2 — offline cProfile of the real dataloader (robust, self-contained)

Runs on CPU (no GPU). One-time cost: reads a 7.5 GB h5ad (use `dev.h5ad`, smaller, and/or
`limit_dataset_samples`). Build the DataModule exactly as the run does via Hydra, then profile a
few batches. Skeleton:

```python
# /proj/bmfm/users/dmichael/envs/biomedmultiomic/bin/python this_script.py
import cProfile, pstats, io, time
from hydra import initialize_config_dir, compose
from bmfm_targets.tasks.task_utils import main_config   # same builder bmfm-targets-run uses

with initialize_config_dir(config_dir="/u/dmichael/biomed-multi-omic/run", version_base="1.2"):
    cfg = compose(config_name="scconcept_ablation_wced", overrides=[
        # keep it small/fast for profiling:
        "data_module.num_workers=0",                 # profile in-process so cProfile sees it
        "data_module.dataset_kwargs.processed_data_source.train=/proj/bmfm/omics/data/scRNA/cellxgene/2025-01-30/cellxgene_random_1pct_h5ad/dev.h5ad",
    ])
# NOTE: main_config expects the SCBertMainHydraConfigSchema; adapt to how
# bmfm_targets/tasks/scbert/scbert_main.py:main builds cfg_obj = main_config(cfg).
cfg_obj = main_config(cfg)
dm = cfg_obj.data_module          # this is a functools.partial; call with tokenizer+fields
# Reproduce how main_run instantiates it — inspect bmfm_targets/tasks/task_utils.py:main_run
# to see the exact tokenizer/fields wiring, OR just grab dm from a trainer build_only path.
dm.setup("fit")
dl = dm.train_dataloader()

it = iter(dl)
next(it)                          # warm up (first batch pays h5ad/index setup)
pr = cProfile.Profile(); pr.enable()
t = time.time()
for _ in range(5):
    next(it)
print("5 batches:", time.time() - t, "s")
pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(30)
print(s.getvalue())
```
If reproducing `main_config`/`main_run` wiring is fiddly, the more reliable variant is to add a
`build_only` short-circuit in `bmfm_targets/tasks/task_utils.py:main_run` (return
`trainer, model, datamodule` before `trainer.fit`), call it, then profile `datamodule.train_dataloader()`.
`num_workers=0` is essential so cProfile sees the transform time in-process.

## Method 3 — Lightning profiler via a short GPU run (fallback)

`bsub` a tiny run (few steps) with a profiler and no ClearML, then read the printed summary
(it breaks down `run_training_batch` vs `get_train_batch`/dataloader-next):
```bash
bsub -J wced_prof -o /u/dmichael/.lsf/cluster/%J.out -e /u/dmichael/.lsf/cluster/%J.err \
  -R 'select[ngpus>=1] span[hosts=1]' -M 128G -gpu 'num=1:mode=exclusive_process' \
  "/proj/bmfm/users/dmichael/envs/biomedmultiomic/bin/bmfm-targets-run \
     -cd /u/dmichael/biomed-multi-omic/run -cn scconcept_ablation_wced \
     task.max_steps=30 task.num_sanity_val_steps=0 ~track_clearml"
```
Check whether `TrainingTaskConfig`/`TrainerConfig` exposes a `profiler` field
(`grep -rn profiler bmfm_targets/config/`); if so set it to `simple`. Otherwise add
`profiler="simple"` where the `L.Trainer(...)` is constructed in `main_run`.

## Compare against AnnCollection (the fast path)

To see *why* vanilla is faster, profile the same 5-batch harness but pointed at
`AnnCollectionDataModule` (vanilla config pulled to `/tmp/vanilla_wced.yaml` via
`clearml.Task.get_task("2dbd5a10023b4a61be7321bf91c63b9c").get_configuration_object("OmegaConf")`).
Diff the two `tottime` profiles — the delta functions are the fix targets.

## What "done" looks like

- A `tottime` ranking (or py-spy hot list) showing the function(s) that consume the ~110 CPU-s/step.
- Verdict: **fixable & output-neutral** (e.g., precompute gene→id map so the label build doesn't
  HF-tokenize ~19k strings/cell; keep numpy instead of Python-list roundtrips; skip the redundant
  transform pass) **vs. inherent** to WCED-all-genes (then reuse the vanilla run or run wced on
  `AnnCollectionDataModule`).

## Key file:line map

- `bmfm_targets/datasets/base_rna_dataset.py:302-309,412` — expose_zeros densify path
- `bmfm_targets/tokenization/multifield_collator.py:278,287` — input transform + `batch["mfi"]` save
- `bmfm_targets/training/masking/strategy.py:169-199,225-253` — WCED label build (2nd transform + tokenize-all-genes)
- `bmfm_targets/training/sample_transforms.py:720-765` (batch-wise pad-zero), `:1195-1210` (transform_inputs apply + `get_genes_expressed_in_batch`)
- `bmfm_targets/tokenization/multifield_tokenizer.py:464-487` — `_replace_tokenized_ids_with_continuous_values` (patch 913d4cf)
- `bmfm_targets/datasets/anncollection/anncollection_dataset.py` — the fast (vanilla) path to diff against

## Env / data

- Python: `/proj/bmfm/users/dmichael/envs/biomedmultiomic/bin/python` (editable install of this repo → code edits are live)
- Config dir: `/u/dmichael/biomed-multi-omic/run`, config: `scconcept_ablation_wced`
- Data: `/proj/bmfm/omics/data/scRNA/cellxgene/2025-01-30/cellxgene_random_1pct_h5ad/{train,dev,test}.h5ad` (train = 7.5 GB)
- Running job to sample/kill: LSF `7991` (wced). Contrastive `7987` and both `7992` already completed.
- Full session context: `docs/scconcept_ablation_session_notes.md` (Session 3) and memory `project_scconcept.md`.
