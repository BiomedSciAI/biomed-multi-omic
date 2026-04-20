# Hydra overrides, CLI flags, and common errors

## CLI anatomy

```
bmfm-targets-run [-cd <config_dir>] -cn <config_name> [key=value ...] [++key=value ...] [~key]
```

- `-cd` / `--config-dir`: where Hydra looks for configs. **If invoking from the biomed-multi-omic repo root, use `-cd run`**; otherwise pass `--config-dir <abs path>/run`. The entry point's default `config_path` is the directory of `scbert_main.py`, which is NOT where users want to look.
- `-cn` / `--config-name`: the `.yaml` filename (no extension) to load as the top config. E.g. `predict`, `finetune`, `rda_mlm`, `mlm_train_config`, `multitask_train_config`, `dna_predict`, `dna_finetune_train_and_test_config`.
- `key=value`: override a field that already exists in the resolved config. Fails if the field is not present.
- `++key=value`: add-or-override. Use this when the field may not be in the base config (e.g. `++data_module.rda_transform=auto_align` against a config that doesn't already set it).
- `~key`: remove a field entirely. Useful for `~model` in predict scenarios where you want everything loaded from the checkpoint instead.

## The three flavors of override

| Syntax | Behavior | When to use |
| --- | --- | --- |
| `key=value` | Replace existing field; error if missing | Known fields in the base config |
| `++key=value` | Add if missing, replace if present | Optional fields (e.g. `rda_transform`) |
| `+key=value` | Add if missing, error if present | Rare; avoid |
| `~key` | Remove field | Ask model to reload structure from checkpoint |

## Dotted paths

Hydra uses dots for nested fields: `data_module.max_length=4096`, `task.accelerator=cpu`, `trainer.learning_rate=1e-5`. Lists are indexed: `label_columns[0].label_column_name=my_col`. For lists, overriding via index can be fragile — prefer editing the yaml.

## Interpolation (`${...}`)

Values can reference other config fields: `working_dir: ${oc.env:BMFM_TARGETS_TRAINING_DIR}/my_run` — this reads the env var `BMFM_TARGETS_TRAINING_DIR`. If the env var is unset and there's no default, it errors at resolution time, not config-load time. The dry-run (`--cfg job --resolve`) catches this.

Common pitfalls:

- `${var}` without the `oc.env:` prefix references another config field, not an env var. `${working_dir}` means "look up `working_dir` elsewhere in the config".
- Circular interpolation silently causes resolution failure. Don't set `a: ${b}` and `b: ${a}`.

## `-cd run` vs `--config-dir`

`-cd run` is a relative path, evaluated from the current working directory. It only works if you're running from the repo root (or any directory that has a `run/` subdirectory). From a different CWD, pass an absolute path:

```bash
bmfm-targets-run --config-dir /abs/path/to/biomed-multi-omic/run -cn predict ...
```

If the user pip-installed and doesn't have `run/`, the skill falls back to its bundled `examples/`:

```bash
bmfm-targets-run --config-dir /abs/path/to/skill/examples -cn predict_template ...
```

## Config composition (defaults list)

Top-level configs compose from smaller configs via a `defaults:` block at the top:

```yaml
defaults:
  - data_module: base_seq_cls
  - task: train_and_test
  - trainer: default
  - _self_
```

This pulls `run/data_module/base_seq_cls.yaml`, `run/task/train_and_test.yaml`, `run/trainer/default.yaml`, merges them, and then merges `_self_` (the current file) on top. To override a default at the CLI, use `data_module=some_other_group`.

## Common errors and fixes

### `Error instantiating '...': missing 1 required positional argument: losses`

The dataclass has a required field that wasn't set. Usually means `trainer.losses` is absent. Add a minimal `trainer.losses` block:

```yaml
trainer:
  losses:
    - label_column_name: my_label
      name: cross_entropy
```

### `Key '...' is not in struct`

You used `key=value` for a field that doesn't exist in the base config. Switch to `++key=value`.

### `InterpolationKeyError: Interpolation key 'X' not found`

A `${X}` reference can't resolve. Either the field `X` isn't in the config, or for `${oc.env:X}` the env var isn't set. Check the full error — it tells you where the reference came from.

### `Could not find 'predict'`

Hydra can't find the named config. Either you forgot `-cd run`, your CWD is wrong, or the config name has a typo. `ls run/*.yaml` to check.

### `ValidationError: Invalid type assigned`

The yaml's value doesn't match the dataclass's type. For example `max_length: "4096"` (string) vs `max_length: 4096` (int). OmegaConf is strict.

### `_partial_` confusion

Many BMFM data_modules and models use `_partial_: true`, which means "instantiate later with these kwargs, don't fully construct now". If you see errors about missing required args to a data_module, it's likely because `_partial_` got dropped somewhere. Restore it.

### Dry-run catches all of these

```bash
bmfm-targets-run -cd run -cn <your_config> --cfg job --resolve <your overrides>
```

This renders the fully-resolved config without running training. Every one of the above errors shows up here.

## Knobs that matter

A short list of the most-asked-about overrides:

- `data_module.max_length` — token sequence length. Pretraining ceiling = checkpoint's training value (typically 1024 or 4096 for v1, 8192 for v2). Can be smaller at inference/fine-tuning. If you need longer, you'll need to re-pretrain.
- `data_module.rda_transform` — only applies when using RDA-trained checkpoints or when pretraining with RDA. Five accepted values, each a distinct policy:
  - `"downsample"` — random downsample to match pretraining counts. Data augmentation flavor; what the v1 RDA checkpoint was trained with.
  - `"poisson_downsample"` — Poisson-noise downsample. Stronger augmentation; pairs with `renoise:` in some pretraining recipes.
  - `"auto_align"` — at collate time, pick the max-counts cell in the dataset and align every sample's target-reads T to that value. Recommended for **inference / zero-shot prediction**, not for training (per the docstring at `bmfm_targets/training/data_module.py:199`). Requires the dataset to implement `.max_counts()` (CellXGene SOMA and litdata paths do).
  - `"equal"` — set source reads S equal to target reads T; prepares inputs in RDA style without upsampling. Useful for encoder-only feature extraction when you don't want augmentation.
  - integer (e.g. `10000`) — hard-code T to a specific value. Reproducible across runs and datasets.
  - `null` / omitted — no RDA transform.
  All RDA modes log-normalize internally, so set `data_module.log_normalize_transform: false` when any RDA mode is on (the framework warns and disables the standalone log_normalize automatically, but the explicit setting is cleaner).
- `data_module.log_normalize_transform` — `true` if the checkpoint expects log-normalized input (this is the norm for every non-RDA published checkpoint); `false` for RDA checkpoints (RDA normalizes internally) or when the h5ad is *already* log-normalized upstream. Raw-count input data with this set to `false` and no RDA will give garbage — see `references/checkpoints.md` for the full story.
- `data_module.limit_genes` — `null` (all genes), `protein_coding` (~19k genes; recommended for matching pretraining corpora), or `tokenizer` (genes known to the tokenizer).
- `data_module.num_workers` — 0 for debugging / CPU, 8+ for GPU training.
- `task.accelerator` — `gpu` (default) or `cpu` (for CI / debugging).
- `task.precision` — `16-mixed` (default for GPU) or `32` (for CPU / debugging).
- `task.max_epochs` / `task.max_steps` — set `max_steps=5` for smoke tests.
- `task.val_check_interval` — `null` disables mid-epoch validation; integer = steps; float = fraction of epoch.
- `working_dir`, `input_file`, `input_directory`, `checkpoint` — the externally-facing paths. Set these from the user's values.
- `batch_column_name` / `counts_column_name` — **optional**, only used by the batch-integration evaluation callback (see `bmfm_targets/training/callbacks.py`). Never add these unprompted. When running `predict` against a dataset that has batch structure the user cares about, offer the callback explicitly: "your data looks like it has batches — want me to add batch-integration metrics? That needs a batch column (e.g. `batch`, `donor_id`) and a total-counts column (usually added automatically by scanpy QC)". Only emit these keys if the user opts in.
- `trainer.pooling_method` — how the sequence is pooled into a vector for downstream heads (embeddings, classification). Signature: `str | int | list[int | str]`. Accepted values:
  - `"first_token"` (default, always valid) — use the hidden state at position 0. Correct choice for any MLM-only checkpoint (its pooler is untrained).
  - `"pooling_layer"` — use the transformer's `[CLS] → Linear → tanh` pooler. Only valid when the pooler was trained: sequence-classification / multitask checkpoints. Will silently produce garbage on a pure MLM checkpoint.
  - `"mean_pooling"` — mean over token positions (ignoring padding).
  - integer (e.g. `1`) — use hidden state at a specific CLS index; matches `decode_from` on a `LabelColumnInfo`. Useful for the v2 47m WCED checkpoint which has 5 CLS tokens.
  - list of ints / strs (e.g. `[1, 2]`) — concatenate hidden states at several positions. Pattern used in the v2 recipes to combine S+T CLS tokens into one embedding.
  For `predict` against an RDA checkpoint, `first_token` or `0` is the usual choice. For finetune against a multi-CLS v2 checkpoint, pick the `decode_from` of whichever pretrained head is closest to your task; if unsure, `first_token` is always a safe fallback.

## Multi-task Hydra runs (sequential)

The `scbert_main.py` entry supports `task` being a **list** of task configs; the loop runs each in sequence, reusing the trainer. Useful for "train, then test" or "finetune, then predict" pipelines. See `run/prediction_train_and_test_config.yaml` and `run/transform_and_prediction_train_config.yaml` for examples. This skill rarely needs to generate these from scratch — if the user wants that pattern, start from one of those templates.
