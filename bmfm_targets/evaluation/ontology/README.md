# Ontology analysis of zero-shot cell type predictions

Decodes the leaf probabilities of an ontology-trained cell type head on the Cell
Ontology, scores the predictions, and draws three figures (paper Figure 3).

| Output | Figure 3 panel |
|---|---|
| `sankey_t{threshold}.html` | A: true to predicted labels, colored by ontology relationship |
| `probability_graph_{label}.png` | B: mean node probabilities of the cells of one label |
| `greedy_flow.html`, `greedy_flow_{nodes,links}.csv` | C: most probable child at each step from the root |
| `ontology_metrics.csv`, `ontology_predictions.csv` | metrics and per-cell labels |

The published Figure 3 was composed by hand from these panels. Panel C uses the
`GreedyFlowConfig` defaults.

## Setup

```bash
pip install -e ".[ontology]"   # plotly and the graphviz python binding
apt install graphviz           # the `dot` executable, or: brew install graphviz
```

The Cell Ontology graph is not shipped. Build it as described in
[`datasets/label_ontology/README.md`](../../datasets/label_ontology/README.md), or set
`ontology_path`. A missing dependency raises an error before prediction starts. To run
without a plot, set its config to `null`.

## Run

With the benchmark configs, as a prediction task:

```bash
bmfm-targets-run -cd bmfm_targets/evaluation/benchmark_configs -cn config_ontology \
  task=predict_ontology data_module=immune_atlas ~model ~fields \
  checkpoint_path=<checkpoint> checkpoint_name=<name>
```

Outputs go to `ontology/` in the run's log directory.

From saved probabilities, without a GPU:

```python
import pandas as pd
from bmfm_targets.evaluation.ontology import OntologyConfig, run_ontology_analysis

run_ontology_analysis(
    OntologyConfig(ground_truth_column="CellType"),
    obs,  # cell metadata indexed by cell name
    pd.read_csv("probabilities.csv", index_col=0),
    output_dir="ontology",
)
```

## Alignment

With `alignment` set (see `AlignmentConfig`), the run also tests whether cell types that
are close in the ontology have similar vectors. Vectors are mean embeddings of true or
predicted labels, or the classifier weights of a checkpoint. Ontology similarity is
personalized PageRank. `alignment_metrics.csv` has correlations, bootstrap intervals,
one-sided permutation p-values and neighbor agreement; `alignment_pairs_*.csv` and
`alignment_types_*.csv` have the values per pair and per type. `alpha_sweep` repeats the
correlation over PageRank dampings. Labels outside the ontology are left out.

## Definitions

- A node's probability is the sum over the leaves below it.
- Level is the shortest distance to a leaf. A cell takes the most probable node of the
  lowest level at which some node reaches the threshold.
- The top-down decoder (`topdown: true`) follows the most probable child from the root
  while its probability reaches the threshold.
- `hierarchical_accuracy` counts exact matches and predictions that are descendants of
  the true label. Cells whose label is not in the ontology are left out.
