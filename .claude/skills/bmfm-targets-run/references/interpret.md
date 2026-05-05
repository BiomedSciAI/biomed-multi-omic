# Interpret: Captum-based attribution

Interpret runs Captum Layer Integrated Gradients over a trained checkpoint to produce per-gene importance scores for each sample, per label of interest. It uses `InterpretTaskConfig` (not `TrainingTaskConfig` or `PredictTaskConfig`) — same entry point, different task type.

## Minimum viable config

When `checkpoint=` is set the model, tokenizer, and fields are all rehydrated from the checkpoint. Don't put `model:`, `tokenizer:`, or `fields:` blocks in the interpret yaml — they'll either be ignored or silently conflict with what the checkpoint actually needs.

Adapt `examples/interpret_from_checkpoint.yaml`:

```yaml
defaults:
  - _self_

data_module:
  _target_: bmfm_targets.datasets.base_rna_expression.BaseRNAExpressionDataModule
  _partial_: true
  max_length: 2048         # must be <= the checkpoint's training max_length
  limit_dataset_samples: 200  # attribution is slow; cap the sample count
  batch_size: 1            # attribution is forced to bs=1 anyway
  num_workers: 4
  dataset_kwargs:
    processed_data_source: ${input_file}
    split_column_name: ${split_column_name}

label_columns:
  - _target_: bmfm_targets.config.LabelColumnInfo
    label_column_name: ${label_column_name}
    is_regression_label: false

task:
  _target_: bmfm_targets.config.InterpretTaskConfig
  default_root_dir: ${working_dir}
  accelerator: gpu
  precision: 16-mixed
  tf32_mode: medium
  attribute_kwargs:
    n_steps: 50            # Integrated Gradients steps; 20–100 typical
  attribute_filter:
    ${label_column_name}: [<class_a>, <class_b>]   # which labels to attribute toward
  checkpoint: ${checkpoint}

input_file: null
working_dir: /tmp/interpret_run
checkpoint: null
label_column_name: null
split_column_name: split_random
```

## `attribute_filter` is load-bearing

Without `attribute_filter`, the attribution module runs Integrated Gradients for **every** class in the label dict — on a 200-class cell-type model that is ~200× slower than necessary and the resulting output is unreadable. The user almost always wants attribution toward a handful of classes (e.g. "which genes make this cell look like a T cell?").

`attribute_filter` is a dict keyed by label column name, where each value is a list of label values to attribute toward:

```yaml
task:
  attribute_filter:
    cell_type: ["T cell", "B cell", "NK cell"]
    # unlisted columns fall back to "all classes"
```

At runtime the code merges your `attribute_filter` with the full class set; only the listed classes get per-sample IG. Always propose a small list to the user, even if they say "all of them" — usually they actually mean 3–5.

## Post-processing: `get_mean_attributions`

Raw output is one attribution vector per (sample, label-column, label) triple. Users almost always want the aggregate: "which genes are consistently important for class X across all cells of class X?". Use `bmfm_targets.evaluation.interpret.get_mean_attributions` for that:

```python
from bmfm_targets.evaluation.interpret import get_mean_attributions, join_sample_attributions
import pandas as pd

# After the run completes, load the per-sample attributions (format depends on run output)
# — see bmfm_targets/evaluation/interpret.py for join_sample_attributions usage.
attribution_df = join_sample_attributions(...)    # genes × samples

mean_attr = get_mean_attributions(attribution_df, alpha=0.05, significance_attr=0.1)
# mean_attr has columns: attribution, p_value, -log2p, highlight
# highlight = True where p < Bonferroni(alpha) and |attribution| > significance_attr
top = mean_attr.query("highlight").sort_values("attribution", ascending=False).head(30)
```

This is the canonical "top genes" view. The `highlight` column filters to genes that are both statistically significant (one-sample t-test against 0 with Bonferroni correction) and have a meaningful effect size.

Point the user at this function rather than letting them eyeball raw attributions — the p-value / effect-size combination is what makes the list interpretable.

## What gets produced

Under `working_dir/` the run writes per-sample `SampleAttribution` records (gene × score tuples, per label-column-per-label). Exact file layout depends on the task version — inspect `bmfm_targets/evaluation/interpret.py` if the user needs the schema.

## Gotchas

- **Don't hardcode model / tokenizer / fields.** With `checkpoint=` set, all three are loaded from the checkpoint. Putting them in the yaml is at best noise, at worst a silent mismatch.
- **`max_length` must be ≤ the checkpoint's training value** (the positional embedding table isn't extendable).
- **Labels must match** the pretrained head's label vocab, OR the checkpoint must have been fine-tuned on the user's label set. Attributing toward an unseen class returns garbage.
- **It's slow.** Budget roughly `n_samples × n_filter_classes × n_steps` forward/backward passes. A few hundred samples × 3–5 classes × 50 steps is fine on a single GPU; full-dataset × all-classes is not.
- **GPU memory.** Batch is forced to 1, but `n_steps=50` holds 50 forward graphs. Drop `n_steps` to 20 if you OOM.
- **`.ckpt` file on disk vs HuggingFace id.** Interpret needs the full Lightning checkpoint; if the user only has an HF id, they need to either download + extract to `.ckpt` or fine-tune first and point at the resulting `.ckpt`.

## When the user says "show me which genes matter"

Flow:

1. Confirm they have a local `.ckpt` file (not just a HuggingFace id).
2. Confirm they have a data subset (100–500 samples is typical).
3. Confirm the **specific classes** to attribute toward — push back if they say "all of them"; pick 3–5.
4. Write the config from `examples/interpret_from_checkpoint.yaml` — no model/tokenizer/fields blocks, with `attribute_filter` set.
5. Dry-run, then run. Warn about runtime.
6. Hand them the `get_mean_attributions` snippet for post-processing; tell them to look at the `highlight` column for the interpretable top-gene list.
