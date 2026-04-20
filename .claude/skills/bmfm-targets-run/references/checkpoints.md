# Checkpoint provenance

Every published BMFM checkpoint has a training config that fully specifies what data, what objective, what tokenizer, and what model shape it was trained with. **The training config IS the provenance.** When the user names a checkpoint, read its training config before writing any override.

## V1 checkpoints (BERT-110M, July 2025 preprint)

Training configs live under `run/checkpoints/` in this repo. If the user has the repo cloned, read the file directly — it's the full truth.

| HuggingFace id | Training config | Shape | Data | Key overrides |
| --- | --- | --- | --- | --- |
| `ibm-research/biomed.rna.bert.110m.mlm.rda.v1` | `run/checkpoints/biomed.rna.bert.110m.mlm.rda.v1.yaml` | BERT 12L/12H/768, 110M | CellXGene 1% equal-downsample, raw counts | `data_module.log_normalize_transform=false`, `data_module.max_length=4096`, RDA via `++data_module.rda_transform=auto_align` (or `downsample` / an integer) |
| `ibm-research/biomed.rna.bert.110m.mlm.multitask.v1` | `run/checkpoints/biomed.rna.bert.110m.mlm.multitask.v1.yaml` | BERT 12L/12H/768, 110M | CellXGene 1% equal-downsample, log-normalized | `data_module.max_length=4096` |
| `ibm-research/biomed.rna.bert.110m.wced.v1` | `run/checkpoints/biomed.rna.bert.110m.wced.v1.yaml` | BERT 12L/12H/768, 110M | CellXGene 1% random, log-normalized, WCED masker | defaults OK |
| `ibm-research/biomed.rna.bert.110m.wced.multitask.v1` | `run/checkpoints/biomed.rna.bert.110m.wced.multitask.v1.yaml` | BERT 12L/12H/768, 110M | CellXGene 1% random, log-normalized, WCED + multitask (HCE for cell_type, focal for tissue/tissue_general/donor_id with GRL) | defaults OK |

**DNA (from the July 2025 preprint):**

| HuggingFace id | Architecture | Notes |
| --- | --- | --- |
| `ibm-research/biomed.dna.ref.modernbert.113m.v1` | ModernBERT 113M | MLM on reference human genome (GRCh38) |
| `ibm-research/biomed.dna.snp.modernbert.113m.v1` | ModernBERT 113M | MLM on SNP-imputed genome using GWAS catalog + ClinVar variants |

## V2 checkpoints (Llama, March 2026 preprint)

As of April 2026 the v2 provenance yamls are now checked in under `run/checkpoints/`. Read them directly — they are authoritative. The preprint (`arxiv 2506.14861` v2, Table `tab:hyperparams_detailed`) covers the same material with less detail.

| HuggingFace id | Training config | Shape | Objective | Key overrides |
| --- | --- | --- | --- | --- |
| `ibm-research/biomed.rna.llama.32m.mlm.multitask.v1` | `run/checkpoints/biomed.rna.llama.32m.mlm.multitask.v1.yaml` | Llama 12L/12H/384, ~32M | MLM Multitask | `change_ratio: 0.15`, `mask_ratio: 0.95`, HCE on cell_type, focal on tissue/tissue_general, **GRL on donor_id (coef 1.0, loss weight 0.05)** |
| `ibm-research/biomed.rna.llama.47m.wced.multitask.v1` | `run/checkpoints/biomed.rna.llama.47m.wced.multitask.v1.yaml` | Llama 12L/12H/384, ~47M | WCED Multitask | `sequence_dropout_ratio: 0.2`, `WCEDMasker` + WCED losses on `all_genes`, **multiple CLS tokens (`prepend_tokens: [CLS, CLS_0..CLS_3]`) with per-label `decode_from` routing** |

**Shared v2 training setup** (read from both yamls):

- Dataset: CellXGene Nexus Index, 10% random split, ~5M cells
- `max_length: 8192` (ceiling, not a requirement at inference)
- `precision: 16-mixed`, `tf32_mode: medium`
- `learning_rate: 1e-4`, `weight_decay: 0.10`, `warmup_steps: 1000`
- Per-GPU `batch_size: 8`, `accumulate_grad_batches: 2`, 4× A100 80GB
- `tokenizer.identifier: protein_coding`, `limit_genes: protein_coding`
- `log_normalize_transform: true` (data is raw counts in the underlying dataset; both checkpoints normalize internally). RDA is not used.
- Expression encoder: `scale_adapt` with `n_sin_basis: 48`, `basis_scale: 0.01`, `trainable: false`, `zero_as_special_token: true` (not `mlp_with_special_token_embedding` like v1 WCED)
- Zero-expression strategy: `batch_wise`, `interleave_zero_ratio: 0.9`, `expressed_sample_ratio: 0.2`

**MLM-specific (32m):** decode_modes `[regression, is_zero]` on expressions; single `[CLS]`; expressions losses are `mse` (ignore_zero) + `is_zero_bce` (weight 1.5). Donor_id is adversarial via `gradient_reversal_coefficient: 1.0`.

**WCED-specific (47m):** decode_modes `{wced: {vocab_field: genes, logit_outputs: [mse, is_zero_bce]}}` on expressions; five prepend tokens and label-to-CLS routing via `decode_from` (see `references/fields_tokenizer.md` for how `decode_from` works); WCED losses target `all_genes` (active, weight 1.0) plus `input_genes` and `non_input_genes` (zeroed weights, present for head compatibility). HCE for cell_type_ontology_term_id, focal for tissue/tissue_general/donor_id/sex; **donor_id has no GRL here** (different from the 32m).

## Pooling method by checkpoint

`trainer.pooling_method` selects how the encoder output becomes a vector for downstream heads. The right choice depends on what the checkpoint was trained with — see `references/overrides.md` for the full signature. Quick map:

| Checkpoint | Recommended pooling | Reason |
| --- | --- | --- |
| `biomed.rna.bert.110m.mlm.rda.v1` | `"first_token"` (or `0`) | MLM-only pretraining; pooler head is untrained. |
| `biomed.rna.bert.110m.mlm.multitask.v1` | `"pooling_layer"` OK, `"first_token"` safe | Multitask head trained; pooler saw label supervision. |
| `biomed.rna.bert.110m.wced.*` | `"pooling_layer"` (multitask variant) or `"first_token"` | WCED variant has a trained pooler via the multitask heads. |
| `biomed.rna.llama.32m.mlm.multitask.v1` | `"first_token"` or `"pooling_layer"` | Single CLS; multitask pretraining. |
| `biomed.rna.llama.47m.wced.multitask.v1` | integer index matching the task's pretrained `decode_from`, or `[1, 2]` for S+T concat, fallback `"first_token"` | Five CLS tokens are routed per-label via `decode_from`; for a finetune head, pick the CLS that was closest to your task during pretraining. |

When in doubt, use `"first_token"` — it's always valid and behaves predictably. `"pooling_layer"` on a pure MLM checkpoint will silently return an untrained projection.

## How to use provenance

When the user says "I want to run `biomed.rna.bert.110m.wced.multitask.v1` for cell-type finetuning on my data":

1. **Read the checkpoint's yaml** (`run/checkpoints/biomed.rna.bert.110m.wced.multitask.v1.yaml`). Note: `max_length: 1024`, `log_normalize_transform: true`, `limit_genes: protein_coding`, expressions field uses `mlp_with_special_token_embedding` encoder, losses include WCED decoder outputs + multitask head losses.
2. **Start from `run/finetune.yaml`** — it defines the finetuning skeleton.
3. **Carry forward from the checkpoint**: `max_length`, `log_normalize_transform`, `limit_genes`, expressions field encoder. These are load-bearing — changing them silently will break loading or silently produce wrong outputs.
4. **Replace the pretraining losses with the finetuning loss(es).** The checkpoint's multitask loss block is for pretraining; for fine-tuning on one new column use `cross_entropy` (or whichever is right — see `losses.md`).
5. **Override the paths**: `input_file`, `working_dir`, `label_column_name`, `split_column_name`.
6. **Document what came from the checkpoint vs. what's new** in a yaml comment at the top, so the user can audit.

## What "provenance" does NOT tell you

- The exact Hydra defaults list that was active (this can change between runs). Always regenerate defaults from the current `run/finetune.yaml` or `run/predict.yaml` base.
- Whether the user's data is in the same units as the pretraining corpus.

### Input data: raw counts is the expected format

**Every published BMFM RNA checkpoint was pretrained on raw integer counts**, normalized to a library size of 10,000 followed by `log1p` — either by `log_normalize_transform: true` at the data_module level, or internally by an RDA transform. Raw counts is the only fully-supported input format. When in doubt, assume the user needs raw counts and warn them if their h5ad looks normalized (non-integer values, max(X) ≪ 1000).

What happens when the user's h5ad is already normalized (float values, typical max ~6–10):

- **Fine-tuning on a classification head**: usually works, but is silently off-distribution. The encoder was trained on log1p(CPM10k), not on whatever normalization the user applied; representations may be worse but training will converge.
- **Representation extraction (zero-shot embeddings via `predict`)**: may work if the prior normalization was `log1p(CPM10k)` or very close. Otherwise representations are untrustworthy.
- **Zero-shot prediction of the pretraining labels, or any reconstruction / WCED / MLM head output**: do NOT proceed. The decoder heads expect log1p(CPM10k) distribution and will produce garbage. Tell the user explicitly to either re-normalize from raw counts or pick a different checkpoint.
- **RDA checkpoints** (`biomed.rna.bert.110m.mlm.rda.v1`): require raw integer counts. RDA operates at the count level (downsample / align total reads) before log-normalizing internally. If the user has log-normalized data, strongly warn them NOT to proceed and either re-obtain raw counts or switch to a non-RDA checkpoint.
- The batch size / GPU config — these are hardware-specific and should be set from the user's machine, not copied from the pretraining run.

## CI as a minimum-viable-overrides cheat sheet

`.github/workflows/test-cli.yml` contains the minimum override set that makes each checkpoint actually run on CPU with a tiny h5ad. These are the tested-in-CI recipes; when in doubt, check that file for the exact override string.

One subtle bit from CI: **all v1 predict invocations use `data_module.log_normalize_transform=false` except the ones where it's already false** — this is a v1 test-path quirk (the pbmc3k sample is raw counts), not a universal rule. For user data that is log-normalized, omit this override.
