# Loss selection

Loss choice is where users get the most confused — the loss system is composition-based (`DataSource` × `Objective`), with many combinations legal. Most users shouldn't touch it. This file explains how to pick, when to deviate, and why.

## The governing principle

**The checkpoint's own losses are your default.** Not because they're always right for your downstream task, but because they encode the shape of the model's output heads. A loss block that doesn't match the model's heads will either fail instantiation or silently train garbage.

For fine-tuning, you usually replace the pretraining losses with task-appropriate losses — that's fine, the heads for new `LabelColumnInfo` entries are created as needed. But for `predict` and `interpret` (no loss involved) and for pretraining-continuation, the checkpoint's own block is the starting point.

## Every label loss requires a matching `LabelColumnInfo`

A `trainer.losses` entry with `label_column_name: foo` by itself does nothing — it binds an objective, but the classification head is built from the matching `LabelColumnInfo` entry in `label_columns`. If there is no `label_columns` entry with `label_column_name: foo`, no head is created and instantiation fails. When you add a fine-tuning loss on a new column, add the `LabelColumnInfo` first and define its shape there (`is_regression_label`, `n_unique_values`, `classifier_depth`, `decode_from`, `gradient_reversal_coefficient`, `silent_label_values`). The loss entry only picks the objective (cross_entropy, focal, mse, …) and optional weight.

This is the same direction for `field_name:` losses on expressions/genes — the head is created from the field's `decode_modes`; the loss picks an objective for a head that already exists.

## Problem type → default

This table is what `run/finetune.yaml` already documents in a comment; expanded here with reasoning.

| Problem | Default `name:` | When to use |
| --- | --- | --- |
| Multi-class classification (balanced) | `cross_entropy` | The baseline. Always safe. |
| Multi-class classification (imbalanced) | `focal` with `focal_gamma: 2` | When one class has >5× more examples than another, or when rare-class recall matters. The v1 multitask checkpoints use this on tissue / donor_id. |
| Binary classification (2-class categorical) | `cross_entropy` | Fine — it degenerates to 2-class softmax. |
| Binary classification (single float label) | `bce_with_logits` | When the label is already a float in [0, 1] — e.g. a probability target. |
| K independent binary labels (multi-label) | `bce_with_logits` | One head outputs K logits; each is an independent binary decision. |
| Regression (single target) | `mse` | Standard MSE. |
| Regression (zero-inflated, e.g. expression) | `mse` with `ignore_zero: true` + paired `is_zero_bce` | Decompose into "is it zero?" (BCE) and "given it's nonzero, how much?" (MSE on nonzeros). This is what WCED does for expression reconstruction. |
| CellXGene cell type ontology labels | `hce` (Hierarchical Cross-Entropy) | **Only** for `cell_type_ontology_term_id` backed by Cell Ontology (`label_ontology: celltypeont`). This is effectively the single HCE use case wired into the codebase today — see the HCE section in `SKILL.md`. For all other "cell type" columns or custom hierarchies, use `cross_entropy` or `focal`. |
| Adversarial de-biasing (remove nuisance signal) | same loss as above + `gradient_reversal_coefficient` on the `LabelColumnInfo` | The classic example: train to remove donor/batch effects. See the section below. |

## Objective catalog (from `bmfm_targets.training.losses.objectives`)

Available `name:` values in `trainer.losses`:

- `cross_entropy` — `CrossEntropyObjective`. Standard classification.
- `focal` — `FocalObjective`. Takes `focal_gamma` (default 2).
- `hce` — `HCEObjective`. Hierarchical cross-entropy; requires an ontology label with roll-up.
- `mse` — `MSEObjective`. Takes `ignore_zero: true|false`.
- `mae` — `MAEObjective`. Less common; use if you specifically want L1.
- `bce_with_logits` — `BCEWithLogitsObjective`. For binary / multi-label float targets.
- `is_zero_bce` — `IsZeroBCEObjective`. Binary "is this expression value zero?" — pairs with `mse` + `ignore_zero: true` for zero-inflated regression.
- `is_zero_focal` — `IsZeroFocalObjective`. Focal variant of is_zero_bce.
- `token_value` — `TokenValueObjective`. Specialized for scRNA expression tokens; see `fields_tokenizer.md`.

The `DataSource` side is implicit — it's picked from whether the loss entry has `label_column_name:` (→ `LabelSource`), `field_name:` (→ `FieldSource`), or `field_name:` + `wced_target:` (→ `WCEDFieldSource`). You don't configure this directly.

## Gradient reversal (adversarial de-biasing)

Used to strip out nuisance information from the model's representations — e.g. making the model work well for cell-type classification while being actively *bad* at predicting donor_id.

**How it works in BMFM:** set `gradient_reversal_coefficient` on the `LabelColumnInfo` you want to *remove*. Add a matching `trainer.losses` entry with that label's name and an appropriate loss (usually `focal` or `cross_entropy`). During training the gradient flowing back from that loss head gets flipped, pushing the encoder to make the representation *less* predictive of that label.

Canonical reference: `run/checkpoints/biomed.rna.bert.110m.wced.multitask.v1.yaml`, where `donor_id` has `gradient_reversal_coefficient: 3.0` and a focal loss with `weight: 0.05`. The small loss weight + high GRL coefficient is deliberate — you want the signal to flow back aggressively but not dominate the primary loss.

For fine-tuning: same pattern. Add the nuisance column as a `LabelColumnInfo` with a GRL coefficient, add a matching loss entry, and keep the weight small relative to your primary task loss.

## Loss weights

When you have multiple losses (multitask finetuning, or WCED-style pretraining), explicit `weight:` values matter. Guidance:

- Unweighted entries default to weight 1.0.
- The primary task loss should dominate; auxiliary / nuisance losses should typically be 0.05 – 0.2.
- For WCED pretraining, the expression reconstruction losses (`mse` + `is_zero_bce`) each carry weight 1.0 by convention — see the wced.v1 and wced.multitask.v1 yamls. When you have both `input_genes` and `non_input_genes` WCED targets, all four entries usually have weight 1.0.

## When to deviate from the checkpoint

Deviate only when the downstream task genuinely requires it:

- The checkpoint pretrains on expressions (MSE + is_zero_bce); you fine-tune on a classification label → you add a `cross_entropy` entry for your new label column and drop the expression losses (or keep them as auxiliary at low weight if you want to retain reconstruction capability — rare).
- The checkpoint uses `focal` for an imbalanced column; your data isn't imbalanced → still fine to use `cross_entropy`, but you'll inherit the checkpoint's head init; `focal` is a fine-tuning default in the package for multitask head reuse.
- The checkpoint's loss block has multitask heads for `tissue`, `donor_id`, etc.; your data doesn't have those columns → drop those entries. Don't leave them pointing at missing columns; Hydra will fail at runtime, not at config-load.

**Always note the deviation in a comment** at the top of the generated yaml so the user understands what changed from the checkpoint's provenance and why.

## What NOT to do

- Don't mix a regression target with `cross_entropy` thinking it'll "just work" — it won't; the head shape is wrong.
- Don't add a WCED loss (`wced_target: ...`) to a checkpoint whose expressions field doesn't have `decode_modes.wced` set. The head doesn't exist.
- Don't add WCED losses without installing `WCEDMasker` as the data_module's `masking_strategy` (and vice versa) **when you are pretraining with WCED**. In a pretraining recipe WCED needs both — the masker builds the `input_genes` / `non_input_genes` / `all_genes` label tensors and the losses consume them. One without the other is a silent no-op or an instantiation failure. **This coupling only applies during WCED pretraining.** Once a WCED checkpoint is trained, it can be used like any other encoder — finetune or predict with it without WCED losses and without `WCEDMasker`; the pretrained head is discarded / ignored in those modes. See `references/fields_tokenizer.md`.
- Don't stack `mse` + `is_zero_bce` with `ignore_zero: false` on both — you'll double-count zeros.
- Don't use `focal` on a balanced binary problem; it costs you calibration for no gain.
