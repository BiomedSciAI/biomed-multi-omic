"""Configuration of the ontology analysis. A plot set to None is not drawn."""

from dataclasses import dataclass, field


@dataclass
class SankeyConfig:
    """True-to-predicted label flow, drawn once per threshold."""

    n_true_labels: int = 15
    n_pred_labels: int = 15


@dataclass
class ProbabilityGraphConfig:
    """Mean node probabilities of the most frequent true labels, drawn with graphviz."""

    top_n_labels: int = 5
    prob_threshold: float = 0.035
    dpi: int = 300


@dataclass
class GreedyFlowConfig:
    """Root-to-leaf flow that follows the most probable child at each step."""

    root_node: str = "cell"
    min_flow_fraction: float = 0.001
    max_nodes_per_layer: int | None = 30


@dataclass
class AlignmentConfig:
    """
    Agreement of type similarities with the ontology, see `alignment.py`.

    Attributes
    ----------
        alpha: PageRank damping.
        knn_k: neighbors compared per type.
        bootstrap_reps: 0 skips the confidence intervals.
        perm_reps: 0 skips the permutation test.
        use_ground_truth: use mean embeddings of the true labels.
        use_predictions: use mean embeddings of the predicted labels at each threshold.
        weight_checkpoint_path: checkpoint whose classifier weights are used as vectors.
    """

    alpha: float = 0.85
    knn_k: int = 10
    bootstrap_reps: int = 200
    perm_reps: int = 200
    ci_level: float = 0.95
    seed: int = 42
    use_ground_truth: bool = True
    use_predictions: bool = False
    weight_checkpoint_path: str | None = None
    weight_key: str | None = None


@dataclass
class OntologyConfig:
    """
    Settings for `run_ontology_analysis`.

    Attributes
    ----------
        ground_truth_column: obs column with the true labels, as ontology ids or names.
        model_label_column: label column of the model head that predicts ontology ids.
        ontology_path: gzipped GraphML file. None uses the packaged cell ontology
            location, see `bmfm_targets/datasets/label_ontology/README.md`.
        thresholds: probability thresholds of the predictions.
        topdown: also predict from the root down, see `OntologyAnalysis.predict_topdown`.
    """

    ground_truth_column: str
    model_label_column: str = "cell_type_ontology_term_id"
    ontology_path: str | None = None
    node_id_key: str = "cell_type_id"
    node_name_key: str = "cell_type_name"
    thresholds: list[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    )
    topdown: bool = False
    max_samples_asw: int = 50_000
    sankey: SankeyConfig | None = field(default_factory=SankeyConfig)
    probability_graph: ProbabilityGraphConfig | None = field(
        default_factory=ProbabilityGraphConfig
    )
    greedy_flow: GreedyFlowConfig | None = field(default_factory=GreedyFlowConfig)
    alignment: AlignmentConfig | None = None
