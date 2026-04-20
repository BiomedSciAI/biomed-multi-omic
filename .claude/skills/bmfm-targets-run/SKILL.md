---
name: bmfm-targets-run
description: Generate, modify, debug, and run YAML configs and `bmfm-targets-run` CLI commands for the biomed-multi-omic (BMFM) package — scRNA (bmfm-rna) and DNA (bmfm-dna) foundation models. Use this skill whenever the user is working in the BiomedSciAI/biomed-multi-omic repo, mentions `bmfm-targets-run`, wants to run prediction / fine-tuning / pretraining / interpretation with BMFM checkpoints (e.g. `ibm-research/biomed.rna.bert.*`, `ibm-research/biomed.rna.llama.*`, `ibm-research/biomed.dna.modernbert.*`), asks about Hydra configs under `run/`, needs to translate a published checkpoint config into a fine-tuning config, picks a loss for a new downstream task, or is debugging a failing Hydra/yaml config in this codebase. Trigger even if the user doesn't say "yaml" or "config" — e.g. "I have an h5ad and I want embeddings" or "help me finetune on celltype" are core use cases for this skill.
---

# bmfm-targets-run

The biomed-multi-omic package is enormously flexible — it's a single Hydra-configured training loop that handles RNA + DNA foundation models across pretraining, fine-tuning, prediction, and interpretability. That flexibility is also why new users get stuck: the yaml surface is large, and which knobs matter depends on which checkpoint you're starting from and what problem you're solving.

This skill's job is to help the user move from **"I have data and a goal"** to **"a working command / yaml that does what I intended, with the right loss for my problem and the right overrides for my checkpoint"** — without drowning them in config. It does that by reading the checkpoint's provenance (the yaml it was trained from) rather than prescribing a recipe, by actively inspecting the user's data before picking labels or losses, and by always ending with a dry-run.

## Scope

Six task classes — all go through the same `bmfm-targets-run` entry point:

1. **predict** (`-cn predict`) — zero-shot embeddings + label predictions from a BMFM checkpoint on an h5ad
2. **finetune** (`-cn finetune`) — supervised RNA fine-tuning on a label column
3. **pretrain** (`-cn mlm_train_config` / `-cn rda_mlm` / `-cn multitask_train_config`) — MLM / RDA / WCED / multitask pretraining
4. **dna_predict** (`-cn dna_predict`) — DNA embeddings from a csv of sequences
5. **dna_finetune** (`-cn dna_finetune_train_and_test_config`) — DNA supervised fine-tuning
6. **interpret** — Captum-based attribution using `InterpretTaskConfig` (see `references/interpret.md`)

Out of scope for this skill: systematic benchmarking across datasets (`bmfm_targets/evaluation/benchmark_configs/`) and dataset conversion (h5ad→litdata, parquet→litdata). If the user asks for those, tell them the skill doesn't cover them and point at the relevant directories.

## Environment detection

Before doing anything else, figure out which environment you're in. This affects everything downstream.

- **Check for `run/` in the current working directory.** If it exists, the user has the repo cloned. Use the repo's own configs as starting points and invoke with `-cd run -cn <config_name>`. Read `run/checkpoints/<ckpt>.yaml` directly for provenance.
- **If `run/` doesn't exist**, the user pip-installed. Fall back to the skill's bundled templates in `examples/`. Invoke with `--config-dir <path-to-skill>/examples -cn <template_name>`. Don't pretend `run/` exists.
- **Check for a virtualenv.** If the CWD has a `.venv/bin/bmfm-targets-run`, prefer that binary. Otherwise use whatever `bmfm-targets-run` resolves to on PATH.

## Core mental model

The checkpoint is the frame. Everything else is overrides.

1. **Identify the task class** (the six above) from what the user describes.
2. **If the user names a checkpoint, read its provenance** — `run/checkpoints/<ckpt_id>.yaml` now exists for every published v1 and v2 checkpoint (llama included, as of April 2026). The provenance yaml tells you what data_module, fields, tokenizer, losses, and max_length the checkpoint was trained with. Don't hardcode its overrides — read them from the provenance. See `references/checkpoints.md` for a per-checkpoint summary of what's load-bearing.
3. **Override only what the user's situation requires.** Paths and working_dir always. `max_length` if the user needs more or less context (the pretraining value is the ceiling, not a requirement). `rda_transform` if the user wants a different RDA policy. Losses if the downstream task demands something different from what the checkpoint was trained with (see `references/losses.md`).
4. **Propose what the checkpoint used as the default loss.** Only reach for alternatives when the problem genuinely differs — e.g. user has a regression target (MSE), severe class imbalance (focal), hierarchical labels (HCE). Explain the reasoning when you deviate.

## Workflow

Follow this sequence. Don't skip the inspection step — it's what separates this skill from a recipe book.

### 1. Understand the user's goal

Pin down, in this order: **what task class** (predict / finetune / pretrain / dna_{predict,finetune} / interpret), **what data they have** (h5ad path / csv directory / existing checkpoint), **what checkpoint** (HuggingFace id or local path, if any), **what they want out** (embeddings, predictions, fine-tuned model, attribution maps). If any of these are ambiguous, ask — don't guess.

### 2. Actively inspect the data

This is the step users struggle to do themselves. Before you write any yaml:

- **For h5ad input**: read the file with `scanpy`, print `adata.shape`, `adata.obs.columns`, `adata.obs.dtypes`, and for each candidate label column print `nunique()` + head. Propose label column(s) to the user for confirmation.
- **For a csv directory (DNA)**: read `train.csv` head + dtypes, identify which columns are the sequence and which are labels, and whether labels are float (regression) or categorical (classification).
- **Check for a split column** (`split_random`, `split_stratified_<label>`, or a user-named one). If absent in h5ad, propose creating one (see `references/data_prep.md`).

See `references/data_prep.md` for the exact snippets to run. Always confirm inferences with the user before writing the yaml.

### 3. Choose the output shape

The skill produces one of two artifacts:

- **CLI one-liner** — for `predict` and `dna_predict`. Small, stateless, matches the repo README style. Put all overrides inline as `key=value` or `++key=value`.
- **Yaml file** — for `finetune`, `pretrain`, `dna_finetune`, and `interpret`. Enough knobs that a one-liner becomes unreadable, and the user will want to rerun it. Write it under `configs/` in the repo root (or user-specified path), use plain `.yaml`, and produce a matching CLI one-liner that runs it with `-cd <dir> -cn <name>` or `--config-dir <dir> --config-name <name>`.

### 4. Write the config

- Start from the closest existing template: `run/<task_class>.yaml` for the overall shape; `run/checkpoints/<ckpt>.yaml` for everything the checkpoint requires. Don't re-derive fields that the checkpoint provenance already sets correctly.
- **When `checkpoint=` is set, do NOT emit `tokenizer:`, `fields:`, or `model:` blocks in the user's yaml.** The framework loads all three from the checkpoint and whatever you put in the yaml is ignored (predict / interpret) or a footgun waiting to happen (finetune, where a mismatched field spec silently breaks head shapes). Read the provenance to *understand* what the checkpoint expects so you can set the right `data_module` overrides, but don't copy those blocks into the user's config. The only time these blocks should appear is when the user is pretraining from scratch (no `checkpoint=`) or deliberately extending the schema (rare).
- **Label-name clash with pretrained heads.** A clash happens **if and only if** the user's `label_column_name` is *identical* to a name the checkpoint already has a head for (e.g. the v2 47m WCED has `cell_type_ontology_term_id`, `tissue`, `tissue_general`, `donor_id`, `sex`; the v1 BERT WCED multitask has `cell_type`, `tissue`, `donor_id`). When names match, Lightning tries to load pretrained head weights into the user's fresh head: if `n_unique_values` matches the pretrained head it loads silently with the wrong label-index meaning, and if it doesn't it fails with a shape mismatch. When the names differ, a fresh head is added and there is nothing to worry about. Rule: check the checkpoint's provenance yaml; if the user's column matches a name there, rename it (e.g. `cell_type` → `my_cell_type`); otherwise move on.
- **At predict/test time**, `label_columns` works differently: the checkpoint's label_columns is authoritative and the YAML's is ignored (with a warning). Pass an explicit empty list (`label_columns: []`) to get embedding-only mode. Don't try to "use a different label column" at predict time; finetune first or run embedding-only.
- Use `++` for fields that might not exist in the base config (the Hydra "add or override" operator). Use plain `key=value` for fields that definitely exist. Getting this wrong is a common Hydra error — see `references/overrides.md`.
- For losses: read the user's problem type and the checkpoint's own losses, then propose. See `references/losses.md`.
- For custom pretraining field/loss/tokenizer triples: see `references/fields_tokenizer.md`. This is deep lore; most users shouldn't need it.
- **Split strategy matters more than people think.** Large pretrained RNA models learn donor / batch structure; a random split at finetune is too easy and generalization numbers will flatter the model. If the user has a `donor_id` or `batch` / `assay` / `dataset_id` column with ≥4 distinct values, propose a held-out-donor split (or held-out-batch) over `split_random`. If the grouping column has only 2–3 values it's not really splittable — note that to the user rather than pretending otherwise. See `references/data_prep.md` for snippets.
- **Pooling method for inference and finetune heads.** The `pooling_method` knob on `trainer` controls how the sequence is reduced to a vector for downstream heads. Options: `"first_token"` (default, always valid), `"pooling_layer"` (only valid when the pooler was trained — i.e. sequence-classification or multitask checkpoints, NOT pure MLM), `"mean_pooling"`, an integer CLS index (e.g. `1`, `2` for the v2 47m's per-task CLS tokens), or a list like `[1, 2]` to concat multiple CLS tokens (e.g. S+T for CellXGene-style representations). For RDA-style `predict`, `first_token` (or `0`) is usually right. For finetuning off a multi-CLS v2 checkpoint, either pick the `decode_from` that matches the pretrained head closest to your task, or fall back to `first_token`. See `references/overrides.md`.

### 5. Dry-run validate (always)

Before telling the user "run this", invoke the Level-1 validator:

```bash
bash .claude/skills/bmfm-targets-run/scripts/dry_run.sh <args you were going to give the user>
```

It wraps `bmfm-targets-run ... --cfg job --resolve` to render the fully-resolved config without starting training. If it errors, fix the config before handing it over. If the user explicitly asks for a smoke test (Level 2), add `task.max_epochs=1 ++task.max_steps=5 data_module.num_workers=0 task.accelerator=cpu task.precision=32` to actually run a few steps.

### 6. Hand off

Give the user: the config file path (if any), the exact command line, a one-sentence note on what the command will produce and where, and any follow-up they'll likely want (e.g. `label_dict.json` will appear in `working_dir/`).

## Default loss selection (short version)

The long version is in `references/losses.md`. The short version:

- **Classification (multi-class)**: `cross_entropy`. Switch to `focal` (γ=2) if classes are imbalanced.
- **Binary classification**: `cross_entropy` with 2 classes is usually fine; `bce_with_logits` if the label is inherently binary float.
- **Regression**: `mse`. Set `ignore_zero: true` if you're predicting expression-like values where most cells have zero.
- **Multi-label (K independent binary targets)**: `bce_with_logits`.
- **Cell Ontology cell type labels specifically**: `hce` (hierarchical cross-entropy). See the HCE section below — in practice this only applies to `cell_type_ontology_term_id` from CellXGene. For any other ontology-backed label you do **not** have the ontology wired in and must fall back to `cross_entropy` or `focal`.
- **Multitask fine-tuning mirroring the checkpoint**: copy the checkpoint's loss block verbatim, then adapt the `label_column_name`s.
- **Adversarial de-biasing (e.g. remove donor effects)**: set `gradient_reversal_coefficient` on the unwanted label's `LabelColumnInfo`; add a matching entry to `trainer.losses`. The v1 BERT WCED multitask checkpoint (coef 3.0, donor_id) and the v2 32m Llama MLM multitask (coef 1.0, donor_id) are the canonical references. Note: the v2 47m WCED llama does **not** use GRL — check the specific checkpoint's provenance.
- **WCED**: requires BOTH `WCEDMasker` as the data_module's masking_strategy AND losses with `field_name: expressions` + `wced_target:`. The `wced_target` value is the user's choice: `input_genes`, `non_input_genes`, or `all_genes` — each is a different reconstruction objective. See `references/fields_tokenizer.md`.
- **Any label loss requires a matching `LabelColumnInfo`** in `label_columns`. The loss binds an objective; the `LabelColumnInfo` defines the head shape (regression vs classification, n_classes, decode_from, GRL coef). Without the label_columns entry, instantiation fails.

When you deviate from the checkpoint's defaults, explain why in one line.

## HCE is narrower than it looks

HCE (Hierarchical Cross-Entropy) requires a registered ontology with roll-up edges so the objective can charge partial credit for predicting an ancestor of the correct term. In the current codebase the only ontology wired in is `celltypeont` (Cell Ontology), used exclusively for the `cell_type_ontology_term_id` column from CellXGene. That's the only label for which `hce` is a real option today.

Practical rules:

- If the user is fine-tuning on CellXGene `cell_type_ontology_term_id` (the pretraining label for the v2 multitask checkpoints), `hce` is the natural match and lets the pretrained head init transfer cleanly. Set `label_ontology: celltypeont` on the `LabelColumnInfo`.
- For any other label — including "cell type" columns with non-Cell-Ontology terms, or cell-type annotations the user made themselves — `hce` is not available. Use `cross_entropy` (or `focal` if imbalanced). Don't pretend otherwise; `hce` without a matching registered ontology will fail at instantiation.
- If the user genuinely has a hierarchical label backed by a different ontology, registering it is out of scope for this skill — point them at `bmfm_targets/training/losses/objectives/hce.py` and the ontology loading code.

## A note on input data units

Every published RNA checkpoint was pretrained on **raw integer counts** that were then normalized internally (either by `log_normalize_transform` at CPM10k+log1p, or by an RDA mode that does its own log-normalization). Raw counts is the expected input for any downstream use. If the user's h5ad is already log-normalized, they can usually still fine-tune on a classification head (off-distribution but it converges), and maybe get usable embeddings via `predict`, but they should NOT run reconstruction / WCED / MLM / zero-shot prediction heads — the output distributions won't match. Ask them to re-obtain raw counts, or pick a different checkpoint. For RDA checkpoints specifically, refuse to proceed on log-normalized input — RDA operates on count-level values and will produce garbage. See `references/checkpoints.md` for the full policy.

## Working directory and artifact policy

- If the user provides `working_dir`, use it.
- Otherwise default to `./outputs/<task_class>_<YYYYMMDD-HHMMSS>/` in the current repo, and tell the user where you're putting it.
- `/tmp` is acceptable for one-off `predict` runs where the artifacts are disposable, but default to a persistent dir otherwise.
- `predict` produces `embeddings.csv` and `predictions.csv` in `working_dir/`; `finetune` produces `labels.json` plus the usual Lightning checkpoints; `interpret` produces attribution maps.

## Debugging existing yaml

When the user pastes a failing config:

1. Read the full Hydra error. The first stack line tells you where instantiation failed; the second tells you which field. `Error instantiating 'bmfm_targets.config.TrainerConfig': missing 1 required positional argument: losses` means `trainer.losses` is absent or empty.
2. Run the dry-run script on their config before proposing fixes — it'll surface all resolution errors, not just the first.
3. Common failures: `_partial_` missing on a data_module that takes kwargs later, `${oc.env:...}` referencing an unset env var, `++` needed because the field isn't in the base schema, `-cd run` forgotten so Hydra can't find the named config.

See `references/overrides.md` for the full gotcha list.

## When to go deep

Don't read every reference file every time. Load on demand:

- `references/checkpoints.md` — **before writing any config that names a checkpoint.** Has the v1 + v2 roster and how to infer overrides from provenance.
- `references/losses.md` — when the user's problem type isn't obviously covered by the checkpoint's losses.
- `references/overrides.md` — when a yaml fails with a Hydra error, or when the user asks about `++` vs `.` vs defaults-list override semantics.
- `references/data_prep.md` — when the user has an h5ad / csv that isn't split-ready, or when you need to inspect what label columns exist.
- `references/interpret.md` — for any interpret task.
- `references/fields_tokenizer.md` — for custom pretraining: new tokenizer vocabs, new field/loss combinations, scale_adapt vs mlp_with_special_token_embedding, `decode_modes`.

## Known limits

- The skill does not currently cover `benchmark_configs/` or dataset conversion utilities (`h5ad2litdata`, `parquet2litdata`). Users asking for those should be pointed at `bmfm_targets/evaluation/benchmark_configs/README.md` and `bmfm_targets/datasets/data_conversion/`.
- The skill assumes the current working directory is the repo root (or a subdirectory of it) when `run/` is present. If the user is elsewhere, tell them to `cd` or pass `--config-dir <abs path to run/>`.
