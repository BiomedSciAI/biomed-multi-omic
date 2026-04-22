# Fields, tokenizers, and deep pretraining lore

Most users will not touch any of this. It's here for two audiences: users doing genuinely custom pretraining, and Claude when it needs to translate a published checkpoint's fields block into a fine-tuning config (because these fields must be copied **exactly** to avoid silent misloads).

## The field/tokenizer/loss triangle

A BMFM config has three coupled pieces:

- **`fields:`** — list of `FieldInfo` objects. Each field is a data stream into the model (genes, expressions, sometimes `label_expressions`). Each field declares how it's tokenized and whether it gets masked.
- **`tokenizer:`** — a `TokenizerConfig` naming the vocab (`all_genes`, `gene2vec`, `protein_coding`, `snp`, etc.).
- **`trainer.losses:`** — what objective computes gradient against which field output head.

These three have to be consistent. A loss entry with `field_name: expressions` only works if the expressions field has a `decode_modes` that exposes the right head.

## Expression field encoders

The expressions field is where most of the complexity lives. The encoder choice tells the model how to map a real-valued expression count into a token representation.

- **`mlp_with_special_token_embedding`** — small MLP on the scalar value + a learned zero token. Used by v1 WCED and v1 MLM multitask.
- **`scale_adapt`** — scale-adaptive sinusoidal encoding with `n_sin_basis` components. Used by v1 RDA and all v2 (llama) multitask checkpoints. Handles wide value ranges more robustly.
- **`continuous_value_encoder`** — the base `tokenization_strategy`; the `encoder_kwargs.kind` specifies which flavor.

Key `encoder_kwargs`:

- `zero_as_special_token: true` — treats 0-expression as a dedicated token, not as "value zero". Recommended.
- `n_sin_basis`, `basis_scale`, `trainable` — scale_adapt-specific; copy from the checkpoint's training yaml exactly.

## Decode modes

`decode_modes` on a field tells the model what output heads to attach:

- `token_scores` — per-token classification over the field's vocab. Used for MLM-style masked token prediction on the `genes` field.
- `regression` — scalar regression head per token.
- `is_zero` — binary head predicting whether the value is zero.
- `wced: {vocab_field: genes, logit_outputs: [mse, is_zero_bce]}` — the WCED decoder. Attaches a shallow decoder that projects from CLS to vocab-sized MSE + is-zero BCE heads. The `vocab_field` tells the WCED decoder which field's vocab to target.

A field can have multiple decode modes simultaneously. Example from MLM+RDA:

```yaml
- _target_: bmfm_targets.config.FieldInfo
  field_name: expressions
  is_masked: true
  tokenization_strategy: continuous_value_encoder
  encoder_kwargs:
    kind: scale_adapt
    n_sin_basis: 48
    basis_scale: 1.5
    trainable: true
    zero_as_special_token: true
  decode_modes:
    - regression
    - is_zero
```

This exposes both a regression and an is_zero head; `trainer.losses` then has `mse` + `is_zero_bce` entries, each with `field_name: expressions`.

## WCED-specific: `wced_target`

When `decode_modes.wced` is set, the WCED decoder produces predictions for **all** genes in the vocab from the CLS embedding, regardless of which genes were in the input. The `wced_target` on the loss entry selects which subset to supervise:

- `wced_target: input_genes` — only genes that were in the cell's input tokens. Smaller signal, higher fidelity.
- `wced_target: non_input_genes` — genes NOT in the input (i.e. reconstruct-what-you-didn't-see). The meat of WCED.
- No `wced_target:` — all genes.

The v1 WCED multitask yaml has four loss entries for this — `mse` + `is_zero_bce` for both `input_genes` and `non_input_genes` — with equal weight. The non_input_genes pair is the autoencoder-like "reconstruct cells from a partial view" objective.

## Multiple CLS tokens: `prepend_tokens` and `decode_from`

The `[CLS]` token at position 0 is the default "summary" embedding most models decode classification heads from. For multitask models it can be useful to give each task its own dedicated CLS position — this prevents the heads from fighting over a single summary and lets the model specialize different positions to different tasks. BMFM supports this via two coupled settings: `prepend_tokens` on the tokenizer, and `decode_from` on each `LabelColumnInfo`.

**`tokenizer.prepend_tokens`** — an ordered list of special tokens prepended to every sequence. If unset, the default is just `[CLS]` at position 0. When set, every sequence starts with exactly these tokens in this order; positions 0, 1, 2, … correspond to list indices 0, 1, 2, …. The tokens must already exist in the tokenizer's special-token vocab.

**`LabelColumnInfo.decode_from`** — an integer position index into `prepend_tokens`. The classification head for that label reads from the hidden state at that position. If `decode_from` is `None` (default), the head reads from the model's pooled output.

Example from `run/checkpoints/biomed.rna.llama.47m.wced.multitask.v1.yaml`:

```yaml
tokenizer:
  identifier: protein_coding
  prepend_tokens: ['[CLS]', '[CLS_0]', '[CLS_1]', '[CLS_2]', '[CLS_3]']
label_columns:
  - label_column_name: cell_type_ontology_term_id
    decode_from: 1          # reads [CLS_0]
  - label_column_name: tissue
    decode_from: 2          # reads [CLS_1]
  - label_column_name: tissue_general
    decode_from: 2          # shares [CLS_1] with tissue
  - label_column_name: donor_id
    decode_from: 3          # reads [CLS_2]
  - label_column_name: sex
    decode_from: 4          # reads [CLS_3]
```

Position 0 (`[CLS]`) is reserved for the WCED decoder here.

### Implications for downstream users

1. **Leave the tokenizer alone when fine-tuning from a multi-CLS checkpoint.** The checkpoint's pretrained hidden states at positions 1..N encode task-specific summaries; swapping to a plain `[CLS]`-only tokenizer destroys this. If the user does not explicitly set `tokenizer:` in their finetune yaml, the framework loads the tokenizer shipped with the checkpoint — which is what you want.

2. **Choose `decode_from` intelligently when adding a new fine-tune head.** You have three reasonable options, in order of preference:
   - Re-use the CLS position whose pretraining task is closest to yours (e.g. for a new cell-type classifier on the WCED checkpoint, `decode_from: 1` — the CLS that already learned cell-type summaries).
   - Use a "fresh" CLS position not bound to any pretraining task (if the checkpoint has spare prepend_tokens), to avoid interference.
   - Omit `decode_from` entirely and use the pooled output — simplest, but ignores the per-task CLS specialization the checkpoint invested in.
   Prompt the user explicitly to make this call rather than guessing.

3. **If `prepend_tokens` is unset on a checkpoint (e.g. the 32m MLM multitask),** there is only `[CLS]` at position 0 and no `decode_from` on the label columns — all heads share the pooled output. Leave it that way.

## WCED masking requires both a WCEDMasker AND matching WCED losses

WCED (Whole Cell Expression Decoder) pretraining is not enabled by a single switch. It requires both:

1. A `WCEDMasker` installed as the data_module's masking strategy:
   ```yaml
   data_module:
     masking_strategy:
       _target_: bmfm_targets.training.masking.strategy.WCEDMasker
       _partial_: true
   ```
   The masker constructs three label sets per batch — `input` (genes present in the input tokens), `non_input` (genes absent from the input), and `all` — which are the prediction targets.

2. Loss entries with `field_name: expressions` and `wced_target:` on the expressions field's WCED decode head:
   ```yaml
   trainer:
     losses:
       - field_name: expressions
         name: mse
         ignore_zero: true
         wced_target: non_input_genes   # or input_genes, or all_genes
       - field_name: expressions
         name: is_zero_bce
         wced_target: non_input_genes
   ```
   The expressions field must have `decode_modes.wced: {vocab_field: genes, logit_outputs: [mse, is_zero_bce]}` set so the WCED decoder head exists to be supervised.

Omit either side and WCED does nothing: masker without losses trains no decoder; losses without masker fail because the `input/non_input/all` label tensors aren't populated.

**The choice of `wced_target` (`input_genes`, `non_input_genes`, `all_genes`) is the user's.** They encode different objectives:

- `input_genes` — "reproduce what I told you." Easier signal, less useful alone.
- `non_input_genes` — "reconstruct what I hid from you." The cell-level autoencoder flavor; this is the interesting WCED signal.
- `all_genes` — "reconstruct the whole cell." Combines both; this is what the v2 47m llama uses with weight 1.0.

The v1 BERT WCED multitask yaml uses both `input_genes` and `non_input_genes` at weight 1.0 and omits `all_genes`. The v2 llama WCED uses `all_genes` at weight 1.0 and keeps `input_genes` / `non_input_genes` at weight 0.0 (head compatibility). Either pattern is legitimate; match the checkpoint's when extending.

## A label loss REQUIRES the label in `label_columns`

A frequent failure mode: user adds a `trainer.losses` entry like
```yaml
- label_column_name: my_new_label
  name: cross_entropy
```
but forgets to add a matching `LabelColumnInfo` in `label_columns`. The framework won't silently infer it — instantiation fails because no classification head is created for `my_new_label`. **Every loss with `label_column_name:` must have a `LabelColumnInfo` entry with the same name**, and that entry drives the head shape (`n_unique_values`, `is_regression_label`, `classifier_depth`, `decode_from`, `gradient_reversal_coefficient`). Add the `LabelColumnInfo` first; the loss entry just binds an objective to it.

## Tokenizer identifiers

Set `tokenizer.identifier` to one of:

- `all_genes` — ~60k genes, the default for most RNA pretraining.
- `all_genes_v2` — newer version of `all_genes` with different gene coverage.
- `protein_coding` — ~19k protein-coding genes. Used for finetuning when `data_module.limit_genes: protein_coding` is set.
- `gene2vec` — gene embeddings from the gene2vec paper.
- `snp` — DNA SNP tokenizer.

Vocab files live under `bmfm_targets/tokenization/<identifier>_vocab/`. The identifier must match what the checkpoint was pretrained with — using a different vocab means the model's embedding table is indexed wrong and you'll get garbage outputs.

## Gradient reversal (`gradient_reversal_coefficient`)

Set on a `LabelColumnInfo`:

```yaml
label_columns:
  - _target_: bmfm_targets.config.LabelColumnInfo
    label_column_name: donor_id
    gradient_reversal_coefficient: 3.0
```

This flips the sign of gradients flowing back from this label's loss head during training — pushing the encoder to make representations *less* predictive of `donor_id`. Pair with a matching `trainer.losses` entry (usually `focal` or `cross_entropy`) with a small weight.

The `silent_label_values:` field (e.g. `- unknown`) tells the framework to not compute the loss when the label equals one of these — useful when the label is missing or uninformative for a subset of samples.

## Custom pretraining: when you really need to touch this

If the user is pretraining from scratch (no checkpoint), the fields/tokenizer block is a design choice. Rough guidance:

- **Standard MLM pretraining**: `genes` field with `is_masked: true, decode_modes: [token_scores]` + `expressions` field with `is_masked: true, decode_modes: [regression, is_zero]`. Losses: `cross_entropy` on genes, `mse` + `is_zero_bce` on expressions. Starting point: `run/mlm_train_config.yaml`.
- **WCED pretraining**: `genes` field unmasked + `expressions` field with `decode_modes.wced`, `WCEDMasker` as masking strategy. Losses: 4-entry block (mse × 2 + is_zero_bce × 2) on input_genes / non_input_genes. Starting point: `run/checkpoints/biomed.rna.bert.110m.wced.v1.yaml`.
- **RDA**: add `rda_transform` to the data_module, use `scale_adapt` encoder, set `log_normalize_transform: false` (RDA works on raw counts). Starting point: `run/rda_mlm.yaml` or `run/checkpoints/biomed.rna.bert.110m.mlm.rda.v1.yaml`.
- **Multitask**: layer classification heads on top of the reconstruction losses via `label_columns` + matching `trainer.losses` entries with label names. Starting point: `run/multitask_train_config.yaml`.

**Don't design a custom pretraining config from scratch.** Copy a checkpoint's provenance yaml and mutate it. The coupling between fields / tokenizer / masker / losses is load-bearing; silent misspecification will train a model that converges but has poor representations.

## `expose_zeros` + `pad_zero_expression_strategy` — never one without the other

For pretraining (and especially WCED), cells need some non-expressed genes in the input so the model learns to discriminate "truly zero" from "not observed". Set that via `data_module.dataset_kwargs.expose_zeros: true`.

But zero-expressed genes outnumber expressed ones by 10–100×. If you turn on `expose_zeros` and stop there, the input sequence is mostly zeros and the model learns to predict zero everywhere. You MUST also configure `pad_zero_expression_strategy` so the framework subsamples zeros down to a useful ratio:

```yaml
data_module:
  dataset_kwargs:
    expose_zeros: true
    pad_zero_expression_strategy: batch_wise   # the v2 default
    interleave_zero_ratio: 0.9                 # fraction of zeros vs nonzeros per sample
    expressed_sample_ratio: 0.2                # fraction of the *expressed* genes to keep
```

`batch_wise` is what the v2 Llama pretraining recipes use; other strategies exist (read `bmfm_targets/datasets/base_rna_expression/` for the full list). If you're unsure, copy the values from the v2 checkpoint yamls — they're tuned. **Do NOT emit `expose_zeros: true` without a matching strategy in any yaml you hand the user.**

## When this file is NOT needed

- Fine-tuning from an existing checkpoint: copy the fields/tokenizer block from the checkpoint's provenance yaml; don't re-derive.
- Prediction: the checkpoint's own config is loaded by the framework; you only override data_module paths.
- DNA tasks: DNA fields are simpler (a single `dna_sequence` field); see `run/dna_predict.yaml` for the pattern.
