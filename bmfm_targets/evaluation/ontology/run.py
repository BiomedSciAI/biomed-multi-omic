"""Ontology analysis of a table of predicted probabilities, written to disk."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import alignment, plots
from .analysis import OntologyAnalysis
from .configs import OntologyConfig
from .graph import OntologyGraph

logger = logging.getLogger(__name__)


@dataclass
class OntologyResults:
    """Metrics, per-cell predictions and the figure files that were written."""

    metrics: pd.DataFrame
    predictions: pd.DataFrame
    sankeys: dict[float, object] = field(default_factory=dict)
    greedy_flow: object | None = None
    probability_graphs: dict[str, Path] = field(default_factory=dict)
    alignment: pd.DataFrame | None = None


def check_config_dependencies(config: OntologyConfig) -> None:
    """Raise if the plots enabled in the config cannot be drawn here."""
    plots.check_plot_dependencies(
        needs_plotly=bool(config.sankey or config.greedy_flow),
        needs_graphviz=bool(config.probability_graph),
    )


def run_ontology_analysis(
    config: OntologyConfig,
    obs: pd.DataFrame,
    probabilities_df: pd.DataFrame,
    output_dir: str | Path,
    embeddings: pd.DataFrame | None = None,
) -> OntologyResults:
    """
    Decode probabilities on the ontology, score them and draw the figures.

    Args:
    ----
        config: analysis settings.
        obs: cell metadata indexed by cell name, with the ground truth column.
        probabilities_df: probabilities indexed by cell name, see
            `OntologyAnalysis.compute_node_probabilities`. Aligned to `obs` by name.
        output_dir: directory for tables and figures.
        embeddings: embeddings indexed by cell name, used for the silhouette.

    Returns:
    -------
        OntologyResults
    """
    check_config_dependencies(config)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    graph = OntologyGraph(
        config.ontology_path, config.node_id_key, config.node_name_key
    )
    analysis = OntologyAnalysis(graph)

    probabilities_df = _align(probabilities_df, obs.index, "probabilities")
    emb = (
        None
        if embeddings is None
        else _align(embeddings, obs.index, "embeddings").to_numpy()
    )
    node_probs = analysis.compute_node_probabilities(
        probabilities_df, config.model_label_column
    )
    true = resolve_labels(graph, obs[config.ground_truth_column])

    predictions = pd.DataFrame({"true": true}, index=obs.index)
    results = OntologyResults(metrics=pd.DataFrame(), predictions=predictions)
    rows = []
    decoders = {"": analysis.predict}
    if config.topdown:
        decoders["topdown_"] = analysis.predict_topdown
    for threshold in config.thresholds:
        suffix = str(threshold).replace(".", "_")
        for prefix, decode in decoders.items():
            pred = decode(node_probs, threshold)
            uncertainty = analysis.uncertainty(node_probs, pred)
            predictions[f"pred_{prefix}t{suffix}"] = pred
            predictions[f"uncertainty_{prefix}t{suffix}"] = uncertainty
            rows.append(
                {
                    "decoder": "topdown" if prefix else "bottomup",
                    "threshold": threshold,
                    "n_unique_labels": len(set(pred)),
                    "mean_uncertainty": uncertainty.mean(),
                    **analysis.hierarchical_metrics(pred, true),
                    **analysis.clustering_metrics(
                        pred, true, emb, config.max_samples_asw
                    ),
                }
            )
        if config.sankey:
            fig = plots.plot_sankey(
                graph,
                predictions[f"pred_t{suffix}"].to_numpy(),
                true,
                config.sankey,
                f"Threshold {threshold}",
            )
            fig.write_html(out / f"sankey_t{suffix}.html", div_id=f"sankey_t{suffix}")
            results.sankeys[threshold] = fig
    results.metrics = pd.DataFrame(rows)
    results.metrics.to_csv(out / "ontology_metrics.csv", index=False)
    predictions.to_csv(out / "ontology_predictions.csv")

    if config.greedy_flow:
        nodes_df, links_df = analysis.greedy_flow(node_probs, config.greedy_flow)
        nodes_df.to_csv(out / "greedy_flow_nodes.csv", index=False)
        links_df.to_csv(out / "greedy_flow_links.csv", index=False)
        fig = plots.plot_greedy_flow(graph, nodes_df, links_df, config.greedy_flow)
        fig.write_html(out / "greedy_flow.html", div_id="greedy_flow")
        results.greedy_flow = fig

    if config.probability_graph:
        labels = pd.Series(true).value_counts()
        labels = labels.head(config.probability_graph.top_n_labels)
        for label, n_cells in labels.items():
            cells = true == label
            mean_probs = {n: float(p[cells].mean()) for n, p in node_probs.items()}
            dot = plots.build_probability_graph(
                graph, mean_probs, label, config.probability_graph
            )
            dot.attr(label=f"{graph.get_name(label)} ({n_cells} cells)", labelloc="t")
            stem = "probability_graph_" + "".join(
                c if c.isalnum() else "_" for c in str(label)
            )
            path = dot.render(filename=stem, directory=out, cleanup=True)
            results.probability_graphs[str(label)] = Path(path)
    if config.alignment:
        results.alignment = _run_alignment(config, graph, predictions, emb, out)
    logger.info("wrote ontology analysis to %s", out)
    return results


def _run_alignment(
    config: OntologyConfig,
    graph: OntologyGraph,
    predictions: pd.DataFrame,
    embeddings: np.ndarray | None,
    out: Path,
) -> pd.DataFrame:
    """Evaluate alignment for each configured source of type vectors."""
    settings = config.alignment
    columns = {}
    if settings.use_ground_truth:
        columns["true"] = "ground_truth"
    if settings.use_predictions:
        pred_columns = [c for c in predictions if c.startswith("pred_")]
        columns |= {c: c for c in pred_columns}
    if columns and embeddings is None:
        raise ValueError("Alignment of true or predicted labels needs embeddings.")
    sources = {
        name: alignment.type_centroids(
            graph, embeddings, predictions[column].to_numpy(dtype=object)
        )
        for column, name in columns.items()
    }
    if settings.weight_checkpoint_path:
        weights, labels = alignment.load_classifier_weights(
            settings.weight_checkpoint_path,
            config.model_label_column,
            settings.weight_key,
        )
        known = [i for i, label in enumerate(labels) if label in graph]
        sources["weights"] = (weights[known], [labels[i] for i in known])
    rows = []
    for name, (vectors, types) in sources.items():
        metrics, pairs, per_type = alignment.evaluate_alignment(
            graph, vectors, types, settings
        )
        pairs.to_csv(out / f"alignment_pairs_{name}.csv", index=False)
        per_type.to_csv(out / f"alignment_types_{name}.csv", index=False)
        rows.append({"source": name, **metrics})
    table = pd.DataFrame(rows)
    table.to_csv(out / "alignment_metrics.csv", index=False)
    return table


def resolve_labels(graph: OntologyGraph, labels: pd.Series) -> np.ndarray:
    """
    Map labels given as ontology ids or names to ids.

    Labels that are not in the ontology are kept as they are and are left out of the
    hierarchical metrics.
    """
    values = labels.astype(object)
    mapping = {v: graph.resolve(str(v)) or str(v) for v in values.dropna().unique()}
    unknown = sorted(v for v in mapping.values() if v not in graph)
    if unknown:
        n_cells = int(values.map(mapping).isin(unknown).sum())
        logger.warning(
            "%d of %d cells have a label that is not in the ontology and are left out "
            "of the hierarchical metrics: %s",
            n_cells,
            len(values),
            unknown,
        )
    return values.map(mapping).to_numpy(dtype=object)


def _align(df: pd.DataFrame, index: pd.Index, what: str) -> pd.DataFrame:
    missing = index.difference(df.index)
    if len(missing):
        raise ValueError(f"{len(missing)} cells have no {what}, e.g. {missing[0]!r}.")
    return df.loc[index]
