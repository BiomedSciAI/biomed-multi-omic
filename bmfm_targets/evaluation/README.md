# Evaluation tools

Analysis code that works on predictions, embeddings and metric tables.

| Module | Use |
|---|---|
| `gene_learnability` | Per-gene detection and quantification skill; which gene families differ from random gene sets |
| `gene_learnability_plots` | Plots of those skills, over genes, families and training |
| `expression_baselines` | Cross-fitted baselines for the expression level of expressed genes, and model error relative to them |
| `model_comparison` | Ranks, paired tests and z-scores of models over benchmark blocks |
| `ontology` | Zero-shot cell type predictions on the cell ontology |
| `clearml_utils` | Collect metrics from ClearML projects |
| `predictions` | Prediction tables; macro F1 of a constant prediction, to spot a collapsed head |

`bmfm_targets.datasets.cellxgene.cell_overlap` counts evaluation cells that are in a
CELLxGENE pretraining split.

## Per-gene error tables

Set `trainer.batch_prediction_behavior` in a run with a field loss on `expressions`:

- `dump_token_level_errors` writes one per-gene CSV per validation. Pass it to
  `gene_learnability.gene_skill_table`.
- `dump` writes every per-cell prediction. These files are large (about 13 GB for
  114,000 cells and all genes).

## Comparing with baselines

```python
from bmfm_targets.evaluation import expression_baselines as eb

x = eb.check_expression_matrix(x)  # cells x genes, log1p(CP10K), expressed entries
folds, groups = eb.donor_grouped_folds(obs["dataset_id"], obs["donor_id"])
baseline = eb.predict_cell_type_mean(x, folds, obs["cell_type"])
per_gene, summary = eb.compare_to_baseline(x, model_pred, baseline, groups)
```

`model_pred` holds one value per stored entry, in the order of `x.data`. Build `x` and
`model_pred` from the same sorted CSC structure.

The co-expression baselines are unregularized fits with about 50 predictors. Genes with
few expressing cells can have very large errors, and these weigh on the pooled squared
error ratio. Prefer the median ratio there.
