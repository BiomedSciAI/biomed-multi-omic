# Cell Ontology

To download Cell Ontology use [cell_ontology/download_cell_ontology.sh](cell_ontology/download_cell_ontology.sh) to obtain file `cl-basic.obo`.

## Building Cell Ontology Graph for CELLxGENE

CELLxGENE stores cell type ids in the `cell_type_ontology_term_id` field. In the [cell_ontology/build_cell_ontology_graph.py](cell_ontology/build_cell_ontology_graph.py),

1. we test if all CELLxGENE cell types in the ontology,
2. remove all graph leaves if they are not in CELLxGENE,
3. save ontology in [GraphML](https://en.wikipedia.org/wiki/GraphML#:~:text=GraphML%20is%20an%20XML%2Dbased,for%20exchanging%20graph%20structure%20data.) format.

## Ontology Use

Built ontology should be stored in the current folder with the prefix `_`, e.g., `_cellxgene`. LabelInfo has a field `label_ontology`. To use ontology that is stored in `_cellxgene`, use the string without prefix, `cellxgene` in the `label_ontology` field. Ontology folder should store

1. `ontology.graphml`- file generated in the previous step.
2. `metadata.yaml`
   ```yaml
    node_id: cell_type_id
    node_description: cell_type
    unknown_id: unknown
    ```
Labels are stored at the `cell_id` attribute of the graph nodes. Unknown label id is `unknown_id`.


## Hierarchical Cross Entropy

Ontology is represented as a DAG. In the Hierarchical Cross Entropy loss, we employ labels of all $K$ leaves of the graph.
For a label in dataset entry, we find all graph leaves that have the label as an ancestor and calculate the Cross Entropy. Predicted probability of the label $l_i$ is the sum of probabilities of the leaves,

$$
p(l) = \frac{\sum_{i=1}^K w_i \, e^{x_i}}{\sum_{i=1}^K e^{z_i}},
$$
where $x_i$ is logits that corresponds to logits of label leaves, $w_i \in \{0,1\}^K $ is a mask to select leaves. In the next step, we maximize log-likelihood,

$$
\mathcal{L}_{\text{HCE}} = -\log \Bigg(\frac{\sum_{i=1}^K w_i \, e^{x_i}}{\sum_{i=1}^K e^{z_i}}\Bigg) = -\log \Bigg(\sum_{i=1}^K w_i \, e^{x_i}\Bigg) + \log \Bigg({\sum_{i=1}^K e^{z_i}}\Bigg)
$$

Therefore, the second term in the Cross Entropy loss is [LogSumExp](https://en.wikipedia.org/wiki/LogSumExp) with implementation in PyTorch. We had to implement only weighted LogSumExp in the first term.

### Implementation details

The edge-case is when all $w_i$ are zeros when the label is unknown. In this case $\mathcal{L}_{\text{HCE}}$ should be equal to zeros. To account for this, we introduce additional value in the vector of $w_i$, a flag that indicates if all entries are not zeros, and multiply the loss by this flag.


## Metrics calculation

The predicted label is determined as the maximum of the logits. Unlike standard cell type classification, here we work with a **set of target labels** (the leaves of the ontology).

To compute accuracy, we check whether the predicted label belongs to the set of target labels. However, the definition of other metrics such as precision and recall is less straightforward. To address this, we define **non-deterministic metrics** by mapping sets of target labels to a single target label according to the following rules:

1. If the predicted label is contained in the set of target labels, assign the target label to be the predicted label.
2. Otherwise, select a random label from the set of target labels.
