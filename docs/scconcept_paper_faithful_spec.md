# scConcept paper-faithful refactor — implementation spec

**Status:** design spec for a fresh implementation session. Untracked; do not commit until reviewed.
**Author context:** written after (a) profiling the wced dataloader, (b) reading the scConcept
paper Methods (biorxiv 2025.10.14.682419), (c) auditing `_PairedViewCollator` and the WCED masker.
**Goal:** re-implement the contrastive arm to be faithful to the paper, calibrate the three
ablation arms (contrastive / wced / both) so they differ *only* in heads+losses, and do it by
**reusing the existing `sample_transforms.py` building blocks organically** — expanding the
package's expressivity, not grafting a parallel pipeline.

---

## 0. The one idea that makes this clean

**A gene panel is a token list. Applying a panel to a cell is exactly
`limit_fields_to_token_list` (a.k.a. `limit_input_tokens`).**

Everything below is a consequence:
- A *view* = the cell subset to a panel = `limit_input_tokens=panel` (+ rank-encode + truncate).
- Two disjoint views = the same cell subset to two disjoint panels.
- "Nonzero-by-omission" (paper's rule) = with `expose_zeros: all`, the **input** transform drops
  zeros (nonzero-only view); `panel ∩ cell` becomes `panel ∩ nonzero(cell)`.
- "Reconstruct the held-out view" (the WCED target we agreed on) = point the **WCED masker's
  independent label transform** at the *other* panel via its existing `limit_input_tokens` hook
  (`strategy.py:189`); it keeps the real zeros (no zero-drop), so the target has a genuine
  zero/nonzero mix. This differential zero handling is the whole trick (§2.2).

So the contrastive input path stays paper-faithful (nonzero only, no padding), while WCED
reconstructs a bounded held-out panel *including its zeros* — which is precisely the "spatial
imputation" task the paper evaluates downstream, promoted to a pretraining objective.

---

## 1. What the paper actually does (ground truth)

From the Methods (quotes condensed):
- **Single-dataset mini-batch.** "construct each mini-batch using cells drawn exclusively from a
  single dataset." B=512 in pretraining. → we already have this:
  `ConditionHomogeneousBatchSampler` keyed on `dataset_id`.
- **Batch-level panels, not per-cell.** "at each training step … we randomly create two
  non-overlapping distinct gene panels … and subset **every cell in the minibatch** based on both
  panels." One `(panel1, panel2)` pair per step, applied to all cells.
- **Panel modes (randomly switched):** (a) random panel — a random number of genes from the
  dataset's gene pool, size stochastic "from the full transcriptome to a minimum threshold of 400
  genes"; (b) one of 14 pre-designed spatial panels with "random dropout ranging from zero up to
  50%."
- **Encoding: nonzero only, local rank encoding.** "genes of nonzero expression are ranked
  according to their raw expression levels"; max length **L=1000** top nonzero genes. No zero
  padding — the paper treats zero-padding (VAE/PCA) as a weakness.
- **The dual-meaning-of-absence** (a gene missing from a view = not-in-panel OR zero-in-cell) is
  handled by the **same-view (intra-panel) negative term** in the InfoNCE loss, *not* by padding.
- **No reconstruction/decoder** anywhere. The WCED arm is *our* ablation addition; paper-fidelity
  constrains only the contrastive mechanics.

### Fidelity gaps in the current impl (to fix)
1. `_PairedViewCollator._make_views` splits **each cell independently** (per-cell rng permutation)
   → cells in "view A" share no common panel → the same-view negative term is contrasting cells
   with *different* panels (semantics broken).
2. Panel size is `min_frac/max_frac` of *each cell's expressed* genes, not absolute counts from a
   shared dataset pool.
3. Views are built from the cell's expressed genes only, so there is no notion of a panel gene the
   cell doesn't express (which is central to the paper's "informed panel" robustness story).

---

## 2. Target architecture

### 2.1 Data flow (all three arms share this pipeline; only heads+losses differ)

```
Dataset (expose_zeros: all)             # real zeros present in the mfi
   │  mfi = {genes: [...all in-var genes...], expressions: [...incl 0.0...]}
   ▼
ConditionHomogeneousBatchSampler        # one dataset per batch (paper-faithful negatives)
   │
   ▼
PanelSplitCollator  (refactor of _PairedViewCollator)
   │  once per batch: sample (panel1, panel2) disjoint token sets  ← PanelSampler
   │    pool = get_genes_expressed_in_batch; sizes log-uniform, floor 400 (clamped to pool)
   │  label source per row = the FULL cell (all genes, expose_zeros: all) — NOT panel-limited
   │  INPUT transform per view: limit_input_tokens=panelX → DROP ZEROS → sort desc by expr → truncate L
   │  MASKER label transform (separate sample_transforms_kwargs): whole cell, KEEP zeros, unfiltered
   ▼
tokens: block layout [A_0..A_{N-1}, B_0..B_{N-1}]   # unchanged from today
   │
   ├── ContrastiveHead ([CLS])  → InfoNCE (cross-view + same-view)     [contrastive, both]  (no labels)
   └── WCED masker              → input_genes + non_input_genes objectives (all_genes weight 0) [wced, both]
```

**The WCED objective is just `input_genes` + `non_input_genes` (equal weight), `all_genes` = 0
(metric-only).** The masker splits its label set into genes-in-the-input vs genes-not-in-the-input.
The **label source is the whole cell, unfiltered** (everything the dataset emits under
`expose_zeros: all`), so `non_input_genes` = the entire held-out transcriptome (all ~19k minus the
input panel), incl. its real zeros. This is the actual whole-cell reconstruction task — automatic in
every arm, no "reconstruct the opposite panel" machinery, and **the label universe is NOT bounded**
(only the *input* is panel-limited). This is the standing WCED loss shape for all runs.

### 2.2 Zero treatment — let all zeros in, cut them only at the input stage

`expose_zeros` is a **dataset-level** flag; keep it **`all`** so the real zeros are in the mfi. The
masker's **separate `sample_transforms_kwargs`** (`strategy.py:191`) is the seam that makes this
trivial:
- **Input transform:** DROP zeros → paper-faithful nonzero-only view (then panel-limit, rank-encode,
  truncate to L). Reuse the existing expressed-only path: `pad_zero_expression_strategy` with
  `expressed_sample_ratio: 1.0` (`_batchwise_pad_zero_expressed_genes` `use_expressed_only` branch,
  `:734-739`) already yields nonzero-only truncated to `max_length`. No new transform needed.
- **Masker label transform:** whole cell, unfiltered, zeros kept → the label carries the cell's real
  zeros, so `is_zero_bce` is meaningful. The masker's `input_genes` vs `non_input_genes` split yields
  seen-vs-held-out automatically (§2.1); loss = `input_genes` + `non_input_genes` equal, `all_genes`
  = 0 (metric). `non_input_genes` = the whole held-out transcriptome; **do not bound the label** (the
  label build/tokenize over ~19k is inherent WCED cost — the vanilla wced run did exactly this).
- **Contrastive arm doesn't label at all**, so its labeling stage is moot — nothing to configure.

Why this is correct *and* simplest:
- The label zeros are the **true measured zeros** from the data — no need to reconstruct or
  negative-sample them. Genes not in the dataset var never enter the mfi → they stay `-100` =
  correctly "unmeasured" (the zero-vs-missing distinction is handled for free by `expose_zeros: all`).
- `is_zero_bce` becomes meaningful and identical in structure for wced and both; `expose_zeros` is
  no longer an ablation axis; wced and both run the **same WCED task** (`input_genes` +
  `non_input_genes`, held-out = non_input).
- **No `expand_to_panel`, no negative sampling, no measured-universe plumbing, no ZINB, no special
  view-B reconstruction.** Dropped.

Performance note: `expose_zeros: all` densifies ~19k genes/cell, but that was already ruled out as
the bottleneck (~50 ms/64-cell batch); the real cost was `dropout_random` O(n²), now fixed. Panel-
limiting keeps the input transform bounded, and bounding the masker's label universe bounds the
label build + its HF-tokenize (removes today's ~19k-gene tokenize). Re-profile after building.
Optional later add-ons (not now): depth/library-size weighting of the zero loss (reuse RDA
`[S]/[T]`), or a ZINB head.

---

## 3. Concrete code changes (mapped to existing building blocks)

Reuse first; add only what's genuinely new. File refs are current line numbers.

### 3.1 Fix `limit_fields_to_token_list` performance — `sample_transforms.py:633`
`[i for i in mfi[field] if i in token_list]` is O(n·|token_list|) when `token_list` is a list
(same class of bug as the `dropout_random` O(n²) we just fixed). **Require/convert `token_list` to
a `set`.** This function becomes the per-view panel-application hot path, so it must be O(n).

### 3.2 Make `limit_input_tokens` dynamic (per-batch panel), threaded like `expressed_genes_in_batch`
Today `limit_input_tokens` is a static config list baked into `compose_transforms` at build time
(`:971`). Generalize the existing **batch-context threading** already used for
`expressed_genes_in_batch`:
- `transform_inputs` (`:1197-1212`) already computes `expressed_genes_in_batch` once per batch and
  passes it as a kwarg into each per-cell `combined_func`. Add an analogous optional
  `input_token_panel` kwarg to the same channel.
- `limit_fields_to_token_list` consumes `input_token_panel` from kwargs when present (dynamic),
  falling back to the static `token_list` (config) otherwise. One unified mechanism, no new
  side-channel.

### 3.3 Drop zeros at the input stage only — reuse existing pad-zero
No new transform. With `expose_zeros: all` the mfi carries real zeros; the **input** transform drops
them (paper-faithful nonzero view) while the **masker label** transform leaves them in (§2.2). Reuse
`pad_zero_expression_strategy` with `expressed_sample_ratio: 1.0` — the `use_expressed_only` branch
(`:734-739`) already returns nonzero-only truncated to `max_length`. The masker's separate
`sample_transforms_kwargs` simply omit that step.

### 3.4 New: `PanelSampler` — `bmfm_targets/training/panels.py` (new small module)
Per-batch sampler producing two disjoint panels (token sets) from a gene pool.
- **Pool = `get_genes_expressed_in_batch`** (`:1149`) — the batch is single-dataset, so this is the
  batch's measured/expressed universe. (DECIDED.)
- Mode: **`random` for v1** — panel *size* (gene count) sampled **log-uniformly, floor 400**,
  clamped to pool size (mirrors `_PairedViewCollator._sample_panel_size` log-uniform, `:74`; biases
  toward smaller/sparser panels like real targeted assays). (DECIDED.) `predesigned` (14 spatial
  panels + 0–50% dropout) deferred to a later resource registry — see §8.
- Disjoint split: permute pool, `panel1 = pool[:k]`, `panel2 = pool[k:k+m]`. `overlap_prob` optional.
- **Per-worker RNG** via `worker_info.seed`, copied from `_PairedViewCollator.rng` (`:41-63`) — do
  not build RNG in `__init__` (pickled → correlated workers; this bug was already fixed once).

### 3.5 Refactor `_PairedViewCollator` → `PanelSplitCollator` — `paired_view_collator.py`
- Sample `(panel1, panel2)` **once per `__call__`** (per batch), not per cell.
- **Do NOT pre-subset the mfi.** Pass the **full cell** to the base `MultiFieldCollator` and thread
  the per-batch panel as `input_token_panel` (§3.2) so the panel limits only the **input** transform.
  The masker's label source (`batch["mfi"]`, pre-transform) therefore stays the whole cell → whole-
  cell `non_input` labels (§2.1). Run twice for the two views (A→panel1, B→panel2); keep the block
  layout `[A…, B…]` so `ContrastiveObjective` (`both_batch=True`) and the token-0 WCED decode are
  unchanged. (This is why panel-limiting must be a threaded *input transform*, not mfi subsetting.)
- For the **wced-only arm** (no contrastive head): single-view mode, input = panel1 rows only
  (`n_views=1`). The WCED masker's `input_genes`/`non_input_genes` split then supplies the
  seen-vs-held-out objectives — no second view needed.
- Delete per-cell `_make_views` (`:92-142`); its feature-dropout can be reintroduced as the
  standard `dropout_random` on the input view if desired (now O(n), already fixed).

### 3.6 WCED masker — config only, no structural change
- Loss config (all WCED-bearing arms): `input_genes` + `non_input_genes` equal weight, `all_genes`
  weight 0 (metric). The masker already computes these three label sets — this is the payoff of the
  existing design (`select_input_labels`/`select_non_input_labels`, `strategy.py:9-41`).
- The label transform keeps real zeros (no zero-drop step) and is **unfiltered** (whole cell):
  `input_genes` = seen panel1 nonzeros, `non_input_genes` = everything else in the cell (panel1
  zeros + the entire held-out transcriptome). Genuine zero/nonzero mix (§2.2).
- **No label bounding.** `mask_inputs` (`:225-237`) tokenizes the whole cell (~19k) for the label —
  inherent WCED cost the vanilla run already paid; leave it. (Optional later micro-opt: cache
  gene→id so the label build skips re-tokenizing, but not required.)

### 3.7 Rank encoding — reuse, no new code
Input views: `sequence_order: "sorted"` → `sort_by_field(field="expressions", reverse=True)`
(`:1012-1013`, `:192`) = local rank encoding, descending by expression. Then `truncate_sequence`
(`:431`) to L=1000. Confirm the CVE (continuous_value_encoder) path is compatible with sorted
order (it should be — ordering is independent of value encoding).

---

## 4. Performance guardrails (the user explicitly asked)

- **`expose_zeros: all` is fine now** (densify ~50 ms/64-cell batch, already ruled out; the 7× was
  `dropout_random` O(n²), fixed). Do NOT reintroduce an O(n²) over the ~19k densified genes.
- **Input is panel-bounded** (≤ L=1000). **The WCED label is NOT** — it's the whole cell (~19k), the
  inherent WCED cost the vanilla run already paid at 400k cells/h. Panel-limiting must be a threaded
  *input transform*, never mfi pre-subsetting (else the label loses the held-out genes; §3.5).
- **Set membership everywhere.** `limit_fields_to_token_list` (§3.1), and any panel/expressed-gene
  lookups. Audit for the `x in <list>` antipattern (we've hit it twice now: `dropout_random`,
  `_batchwise_pad_zero`'s `np.isin(..., list(...))`).
- **Compute batch-level state once per batch** (panels, `expressed_genes_in_batch`), thread as
  kwargs — never per cell.
- **WCED label tensor** is `[B × vocab_len]` (~19071) at `strategy.py:218-222`. With panel-scoped
  targets only `panel_out` positions are filled; consider whether a dense full-vocab tensor is
  worth it at B=128 (≈2.4M floats/set × 3 sets) or whether to scope the label tensor to the panel.
  Keep dense for compatibility initially; note as an optimization.
- Re-profile with py-spy after implementation (harness: see `scconcept_wced_profiling_instructions.md`).

---

## 5. Hydra config plan (only expose what's genuinely different)

`run/` already has config groups (`data_module/`, `model/`, `task/`, `trainer/`, `fields/`,
`tokenizer/`, `label_columns/`) and the composition idiom (`defaults:` list + `_self_`, e.g.
`mlm_train_config.yaml`). The scconcept configs are the only flat monoliths — convert them.

Proposed decomposition:
- **Shared base** (identical across all three arms today, verified): `seed`, `tokenizer`
  (protein_coding), `model` (llama 12L/384/flex), `task` (TrainingTaskConfig), `trainer` optimizer
  block, `label_columns: []`, clearml `project_name`, and the data_module commons (num_workers 16,
  max_length 1024, sequence_order sorted, limit_genes protein_coding, log_normalize, the
  ConditionHomogeneousBatchSampler + PanelSampler settings). Put in group option files:
  `model/llama_scconcept.yaml`, `task/scconcept.yaml`, `trainer/scconcept_base.yaml`,
  `data_module/scconcept_base.yaml` (the shared panel-split pipeline), `panel_sampler/default.yaml`.
- **Per-arm (the only genuine differences):**
  - `fields/scconcept_{contrastive,wced,both}.yaml` — which `decode_modes` (contrastive / wced /
    both).
  - losses (inline in each top config or `trainer/scconcept_{arm}.yaml`).
  - collator mode: `n_views` (2 for contrastive/both, 1 for wced) + which heads active.
  - `track_clearml.task_name`.
- Each top config `scconcept_ablation_{arm}.yaml` becomes a thin `defaults:` list + a few `_self_`
  overrides (losses, task_name, decode_modes). Target: a reader can see the entire ablation axis at
  a glance.

Fix incidental drift while here: `wced` was missing `num_sanity_val_steps: 2`; unify the data
access (`train.h5ad` vs `processed.h5ad`+`split_column` — same cells, pick one); `batch_size` vs
`${n_cells_per_batch}`.

---

## 6. Calibrated ablation matrix (the end state)

| arm | input views | heads | WCED loss | batch sampler |
|---|---|---|---|---|
| contrastive | A(panel1), B(panel2), nonzero, rank-enc | ContrastiveHead(CLS) | — (no labels) | ConditionHomogeneous (single-dataset) |
| wced | A(panel1) only, nonzero, rank-enc | WCED | input_genes(w) + non_input_genes(w) + all_genes(0) | **plain shuffle** (see note) |
| both | A(panel1), B(panel2) | Contrastive + WCED | input_genes(w) + non_input_genes(w) + all_genes(0) | ConditionHomogeneous (single-dataset) |

All arms: `expose_zeros: all` (zeros dropped at input, kept in label), shared panel sampler,
identical model/optimizer. Differences reduce to heads + losses + n_views + batch sampler. WCED loss
is always `input_genes` + `non_input_genes` equal-weight with `all_genes` = 0 (metric-only). The
label source is the **whole cell, unfiltered**, so `non_input_genes` = the entire held-out
transcriptome (all ~19k minus the input panel) — produced automatically by the masker; the label is
never panel-bounded (only the input is).

> **NOTE (decided #4):** the `wced` arm uses **plain shuffling**, not the single-dataset
> `ConditionHomogeneousBatchSampler` (which exists for contrastive negatives). This is a minor
> batch-composition difference between `wced` and `contrastive`/`both` — accepted as good enough;
> flagged here so it's a *declared* difference, not silent drift.

---

## 7. Validation plan (before any long run)

1. **Unit: panel split.** Two panels disjoint; each view = `panel ∩ nonzero(cell)`; batch-level
   (all cells in a view share the panel). Assert on a toy batch.
2. **Unit: WCED target.** For a known cell+panel, label over `panel_out` has correct zeros/nonzeros;
   `is_zero_bce` target is a real mix (not all-nonzero).
3. **Perf smoke.** 5-batch cProfile (harness in `scconcept_wced_profiling_instructions.md`, Method
   2): confirm no function is O(n²), no ~19k densify, dataloader CPU/step within target. py-spy the
   live worker after launch.
4. **Numerical parity where expected.** Contrastive recall on paired views should be ≥ the current
   impl once panels are batch-level (same-view negatives now valid). Compare to job 7987 (recall@1
   0.70).
5. **is_zero_bce sanity.** In wced+both, `is_zero_bce` should now be a real, comparable number
   (not ~2e-5). This is the headline calibration check.

---

## 8. Risks / open questions

**DECIDED (2026-07-05):** panel pool = `get_genes_expressed_in_batch` (#1); random panels only for
v1, log-uniform size floor-400 (#2/#3); `wced` uses plain shuffle, not the homogeneous sampler,
declared in §6 (#4); WCED loss = `input_genes` + `non_input_genes` equal, `all_genes` = 0 — held-out
falls out of the masker's existing split, no view-B machinery (#5); Hydra refactor **before**
running (#6); `sequence_dropout_factor` removed from `both.yaml` (#7, done).

Also DECIDED: **WCED label = the whole cell, unfiltered** (all genes from `expose_zeros: all`);
`non_input_genes` = the entire held-out transcriptome. The label is never panel-bounded — only the
input is. (Resolves the earlier "label universe" question.)

**Still open:**
- **Predesigned panels (14 spatial):** deferred to a resource registry; add later for spatial-transfer.
- **Dense `[B × vocab]` label tensor** memory at larger B (see §4) — keep dense v1, optimize if needed.

---

## 9. Suggested implementation order

1. §3.1 set-fix + §3.2 dynamic `limit_input_tokens` threading (small, unlocks everything).
2. §3.4 `PanelSampler` + unit test (§7.1).
3. §3.5 `PanelSplitCollator` (contrastive path first; verify recall parity §7.4).
4. §3.3 input zero-drop (expressed-only) + §3.6 masker WCED loss = input_genes+non_input_genes
   (wced path; verify §7.2, §7.5).
5. `both` arm (superset).
6. **§5 Hydra decomposition — do this BEFORE any run (#6).** Refactoring the configs into a shared
   base + per-arm groups mechanically prevents cross-arm drift except in declared places, so it
   gates the launch rather than following it.
7. §7.3 perf profile; then launch calibrated ablations.

Key files: `bmfm_targets/training/sample_transforms.py`, `.../masking/strategy.py`,
`bmfm_targets/datasets/paired_view_collator.py`, `bmfm_targets/training/data_module.py`
(`ContrastiveDataModule` `:1319`), `run/scconcept_ablation_*.yaml`. Env + data + profiling harness:
see `docs/scconcept_wced_profiling_instructions.md`. Prior findings: memory `project_scconcept`.
