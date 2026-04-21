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

Out of scope: `benchmark_configs/` and dataset conversion (h5ad→litdata, parquet→litdata). Point users at `bmfm_targets/evaluation/benchmark_configs/README.md` and `bmfm_targets/datasets/data_conversion/`.

## Environment detection

Before doing anything else, figure out which environment you're in.

- **Check for `run/` in the current working directory.** If it exists, the user has the repo cloned. Use the repo's own configs as starting points and invoke with `-cd run -cn <config_name>`. Read `run/checkpoints/<ckpt>.yaml` directly for provenance.
- **If `run/` doesn't exist**, the user pip-installed. Fall back to the skill's bundled templates in `examples/`. Invoke with `--config-dir <path-to-skill>/examples -cn <template_name>`. Don't pretend `run/` exists.
- **Check for a virtualenv.** If the CWD has a `.venv/bin/bmfm-targets-run`, prefer that binary. Otherwise use whatever `bmfm-targets-run` resolves to on PATH.
- **MPS (Apple Silicon) is fully supported** — use `task.accelerator=mps` for all models. BERT models work natively. LLaMA v2 models require the `flex_attention` CPU/MPS guard in `bmfm_targets/models/common/llama/llama_layers.py` (see feedback memory). On a single MPS device, **run predict jobs sequentially** — parallel jobs thrash shared GPU memory. Chain with `&&`.

## Core mental model

The checkpoint is the frame. Everything else is overrides.

1. **Identify the task class** from what the user describes.
2. **If the user names a checkpoint, read its provenance** — `run/checkpoints/<ckpt_id>.yaml` exists for every published v1 and v2 checkpoint. The provenance yaml tells you what data_module, fields, tokenizer, losses, and max_length the checkpoint was trained with. Don't hardcode overrides — read them from provenance. See `references/checkpoints.md` for a per-checkpoint summary.
3. **Override only what the user's situation requires.** Paths and working_dir always. `max_length` if needed. Losses if the downstream task differs from pretraining.
4. **Propose what the checkpoint used as the default loss.** Only reach for alternatives when the problem genuinely differs — regression (MSE), imbalance (focal), hierarchical ontology labels (HCE). Explain the reasoning when you deviate.

## Workflow

Follow this sequence. Don't skip the inspection step — it's what separates this skill from a recipe book.

### 1. Understand the user's goal

Pin down task class, data path, checkpoint id, and desired output. Ask if any are ambiguous — don't guess.

### 2. Actively inspect the data

Before writing any yaml:

- **For h5ad input**: read with `scanpy`, print `adata.shape`, `adata.obs.columns`, `adata.obs.dtypes`, and for each candidate label column print `nunique()` + head. Propose label column(s) for confirmation.
- **For a csv directory (DNA)**: read `train.csv` head + dtypes, identify sequence and label columns, check if labels are float (regression) or categorical.
- **Check for a split column.** If absent in h5ad, propose creating one (see `references/data_prep.md`).

Always confirm inferences with the user before writing the yaml.

### 3. Choose the output shape

- **CLI one-liner** — for `predict` and `dna_predict`.
- **Yaml file** — for `finetune`, `pretrain`, `dna_finetune`, and `interpret`. Write it under `configs/` and produce a matching one-liner with `-cd <dir> -cn <name>`.

### 4. Write the config

- Start from the closest template: `run/<task_class>.yaml` for shape; `run/checkpoints/<ckpt>.yaml` for what the checkpoint requires.
- **When `checkpoint=` is set, do NOT emit `tokenizer:`, `fields:`, or `model:` blocks.** The framework loads all three from the checkpoint. Read provenance to understand what the checkpoint expects so you can set the right `data_module` overrides, but don't copy those blocks into the user's config.
- **Label-name clash with pretrained heads.** A clash happens **if and only if** the user's `label_column_name` is identical to a name the checkpoint already has a head for (v2 47m WCED: `cell_type_ontology_term_id`, `tissue`, `tissue_general`, `donor_id`, `sex`; v1 BERT WCED multitask: `cell_type`, `tissue`, `donor_id`). Mismatched shapes fail loudly; matched shapes load silently with wrong label-index meaning. When names differ, a fresh head is added — nothing to worry about. Rule: check provenance; if the user's column matches a name there, rename it.
- **At predict/test time**, `label_columns` in the YAML is ignored (the checkpoint's is authoritative). Pass `label_columns: []` for embedding-only mode.
- Use `++` for fields that might not exist in the base config. Use plain `key=value` for fields that definitely exist. For pooling, always use `trainer.pooling_method=<value>` — no `++`.
- **Split strategy matters.** Large pretrained RNA models learn donor / batch structure; a random split at finetune is too easy. If the user has `donor_id` / `batch` / `dataset_id` with ≥4 distinct values, propose a held-out-donor split. See `references/data_prep.md`.
- **Pooling method — always choose explicitly, never rely on checkpoint defaults.** See section below.

### 5. Dry-run validate (always)

```bash
bash .claude/skills/bmfm-targets-run/scripts/dry_run.sh <args>
```

Renders the fully-resolved config without starting training. Fix errors before handing off. For a smoke test (Level 2), add `task.max_epochs=1 ++task.max_steps=5 data_module.num_workers=0 task.accelerator=cpu task.precision=32`.

### 6. Hand off

Config file path (if any), exact command line, one sentence on what it produces and where, likely follow-up (e.g. `label_dict.json` in `working_dir/`).

## Pooling method

**The pooling choice is a biological question, not just a model property.** Each CLS token in a multitask checkpoint encodes a different biological axis. A task-specific CLS (e.g. LLaMA-47m `CLS[1]`, trained for cell-type classification) concentrates cell-type signal and suppresses everything else — disease state, batch effects, developmental gradients. If you pool from `CLS[1]` and AD vs. control separation disappears, that's the intended tradeoff, not a bug. **Match the CLS to the downstream biological question, not to whatever gives the best cell-type ARI.**

Always emit `++trainer.pooling_method=<choice>` explicitly. Never omit it; never rely on the stored checkpoint default (it reflects pretraining setup, not inference intent).

Options:
- `"first_token"` — raw CLS at position 0. Safe fallback for any checkpoint.
- `"pooling_layer"` — CLS → Linear → tanh. **Only valid when this head was trained** (multitask checkpoints). Using it on an untrained pooler produces garbage silently.
- `"mean_pooling"` — mean over non-padding positions. Rarely used in published recipes.
- integer (e.g. `1`) — hidden state at a specific CLS index, matching the `decode_from` routing the checkpoint was trained with.
- list of ints (e.g. `[1, 2]`) — concatenation of multiple CLS positions.

Per-checkpoint choices (full table in `references/checkpoints.md`):

| Checkpoint | Use |
|---|---|
| `bert.110m.mlm.rda.v1` | `first_token` — pooler untrained |
| `bert.110m.mlm.multitask.v1` | `pooling_layer` — pooler trained |
| `bert.110m.wced.v1` | `first_token` — pooler untrained |
| `bert.110m.wced.multitask.v1` | `first_token` — avoids Linear projection; use `pooling_layer` only if you want the projected representation |
| `llama.32m.mlm.multitask.v1` | `pooling_layer` — single CLS, pooler trained |
| `llama.47m.wced.multitask.v1` | `1` (cell type), `2` (tissue), `[1,2]` (combined), `first_token` as last resort |

## Default loss selection

Full details in `references/losses.md`. Quick guide:

- **Multi-class classification**: `cross_entropy`. Use `focal` (γ=2) for imbalanced classes.
- **Binary classification**: `cross_entropy` with 2 classes; `bce_with_logits` for inherently binary float labels.
- **Regression**: `mse`. Set `ignore_zero: true` for expression-like values.
- **Multi-label (K independent binary targets)**: `bce_with_logits`.
- **`cell_type_ontology_term_id` from CellXGene specifically**: `hce` (hierarchical cross-entropy, `label_ontology: celltypeont`). This is the **only** label for which `hce` is valid today — for any other cell-type column, use `cross_entropy`.
- **Adversarial de-biasing, WCED, multitask**: see `references/losses.md`.
- **Any label loss requires a matching `LabelColumnInfo`** in `label_columns`. Without it, instantiation fails.

When you deviate from the checkpoint's defaults, explain why in one line.

## Input data units

Every published RNA checkpoint was pretrained on **raw integer counts** normalized internally to CPM10k + log1p. Raw counts is the expected input. If the user's h5ad is already log-normalized:
- Fine-tuning on a classification head: usually works (off-distribution but converges).
- `predict` embeddings: may be usable if prior normalization was close to CPM10k+log1p.
- Reconstruction / WCED / MLM heads: do **not** proceed — decoder output distributions won't match.
- RDA checkpoints: refuse entirely — RDA operates at count level and will produce garbage on log-normalized input.

See `references/checkpoints.md` for the full policy.

## Working directory and artifact policy

- Use the user's `working_dir` if provided; otherwise default to `./outputs/<task_class>_<YYYYMMDD-HHMMSS>/`.
- `/tmp` is acceptable for disposable `predict` runs; prefer a persistent dir otherwise.
- `predict` produces `embeddings.csv` and `predictions.csv`; `finetune` produces `labels.json` + Lightning checkpoints; `interpret` produces attribution maps.
- **Disk space warning for `predict`.** On a large h5ad (10k+ cells), `predictions.csv` (logits + probabilities) can reach 2–3 GB per checkpoint. Warn the user to check available space (`df -h .`) before running. If only embeddings are needed, delete `predictions.csv` after each run — `embeddings.csv` is typically <100 MB. Note: `task.output_predictions=false` does **not** suppress logit output in the current codebase; manual deletion is the only workaround.

## Multi-checkpoint embedding comparison

When comparing embeddings across several checkpoints on the same h5ad:

1. **Use a named `working_dir` per checkpoint** (not timestamped). E.g. `working_dir=/tmp/bmfm_bert_first_token/`. Makes downstream loading deterministic.
2. **Run sequentially on MPS/CPU.** Chain with `&&`.
3. **Record the pooling method** in the dir name or a README. Silent defaults produce results that are uninterpretable later.
4. After all runs: load `embeddings.csv` from each dir, concatenate with a `method` column, compute shared metrics.

## Debugging existing yaml

1. Read the full Hydra error. The first stack frame says where instantiation failed; the second says which field.
2. Run the dry-run script before proposing fixes — it surfaces all resolution errors, not just the first.
3. Common failures: `_partial_` missing on a data_module; `${oc.env:...}` referencing an unset var; `++` needed for a field not in the base schema; `-cd run` forgotten.
4. **`AttributeError: 'dict' object has no attribute 'merge_from_checkpoint'`** — `trainer:` block is missing from the base yaml. Fix: ensure the base yaml has a `trainer:` stub (all shipped yamls under `run/` should have one). Then use `trainer.pooling_method=X` without `++`.

See `references/overrides.md` for the full gotcha list.

## When to go deep

Load on demand:

- `references/checkpoints.md` — before writing any config that names a checkpoint.
- `references/losses.md` — for adversarial de-biasing, WCED, multitask, or anything not in the quick guide above.
- `references/overrides.md` — when a yaml fails with a Hydra error.
- `references/data_prep.md` — when the h5ad isn't split-ready or you need to inspect label columns.
- `references/interpret.md` — for any interpret task.
- `references/fields_tokenizer.md` — for custom pretraining: new vocabs, new field/loss combinations, `decode_modes`.
