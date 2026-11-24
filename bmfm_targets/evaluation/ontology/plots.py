"""Figures of the ontology analysis. Needs the `ontology` extra."""

import colorsys
import shutil

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import to_hex

from .configs import GreedyFlowConfig, ProbabilityGraphConfig, SankeyConfig
from .graph import OntologyGraph

RELATIONSHIP_COLORS = {
    "exact": "rgba(76,175,80,0.6)",
    "descendant": "rgba(33,150,243,0.5)",
    "ancestor": "rgba(156,39,176,0.5)",
    "unrelated": "rgba(244,67,54,0.2)",
    "other": "rgba(158,158,158,0.5)",
}
RELATIONSHIP_CMAPS = {
    "exact": "Greens",
    "descendant": "Blues",
    "ancestor": "Purples",
    "unrelated": "Reds",
}
OTHER = "other"


def check_plot_dependencies(needs_plotly: bool, needs_graphviz: bool) -> None:
    """Raise if a package or executable needed by the requested plots is missing."""
    missing = []
    if needs_plotly:
        try:
            import plotly  # noqa: F401
        except ImportError:
            missing.append("the 'plotly' python package")
    if needs_graphviz:
        try:
            import graphviz  # noqa: F401
        except ImportError:
            missing.append("the 'graphviz' python package")
        if shutil.which("dot") is None:
            missing.append("the graphviz 'dot' executable (a system package)")
    if missing:
        raise ImportError(
            f"The requested ontology plots need {', '.join(missing)}. Python packages "
            "come with `pip install bmfm_targets[ontology]`. To run without a plot, "
            "set its config to null."
        )


def plot_sankey(
    graph: OntologyGraph,
    pred: np.ndarray,
    true: np.ndarray,
    config: SankeyConfig,
    title: str = "",
):
    """Flow from true (left) to predicted (right) labels, colored by relationship."""
    import plotly.graph_objects as go

    def top(labels, n):
        # value_counts leaves ties in order of appearance; sort by id as well
        counts = labels.value_counts().sort_index()
        kept = counts.sort_values(ascending=False, kind="stable").index[:n].tolist()
        return kept, labels.where(labels.isin(kept), OTHER)

    known = pd.notna(true)
    pred, true = pred[known], true[known]
    true_kept, true_mapped = top(pd.Series(true).astype(str), config.n_true_labels)
    pred_kept, pred_mapped = top(pd.Series(pred).astype(str), config.n_pred_labels)
    flows = pd.DataFrame({"true": true_mapped, "pred": pred_mapped}).value_counts()
    flows = flows.sort_index()
    left = true_kept + [OTHER] * int((true_mapped == OTHER).any())
    right = pred_kept + [OTHER] * int((pred_mapped == OTHER).any())

    def color(true_id, pred_id):
        if OTHER in (true_id, pred_id):
            return RELATIONSHIP_COLORS["other"]
        return RELATIONSHIP_COLORS[graph.relationship(pred_id, true_id)]

    fig = go.Figure(
        go.Sankey(
            node={
                "pad": 15,
                "thickness": 20,
                "label": [graph.get_name(label) for label in left + right],
                "color": "rgba(200,200,200,0.8)",
            },
            link={
                "source": [left.index(t) for t, _ in flows.index],
                "target": [len(left) + right.index(p) for _, p in flows.index],
                "value": flows.tolist(),
                "color": [color(t, p) for t, p in flows.index],
            },
        )
    )
    fig.update_layout(title_text=title, height=1000)
    return fig


def build_probability_graph(
    graph: OntologyGraph,
    mean_probs: dict[str, float],
    true_id: str,
    config: ProbabilityGraphConfig,
):
    """
    Ontology subgraph of nodes above the probability threshold, as a graphviz Digraph.

    Fill hue is the relationship to the true label and intensity is the probability.
    Leaves have a thick border. Nodes are emitted in id order, so equal inputs give
    equal output.
    """
    from graphviz import Digraph

    shown = sorted(n for n, p in mean_probs.items() if p > config.prob_threshold)
    if not shown:
        raise ValueError(f"No node has probability above {config.prob_threshold}.")
    names = {node: f"n{i}" for i, node in enumerate(shown)}
    max_prob = max(mean_probs[n] for n in shown)
    dot = Digraph(format="png")
    dot.attr(rankdir="TB", nodesep="0.4", ranksep="0.6", dpi=str(config.dpi))
    for node in shown:
        cmap = colormaps[RELATIONSHIP_CMAPS[graph.relationship(node, true_id)]]
        intensity = max(0.05, mean_probs[node] / max_prob)
        dot.node(
            names[node],
            label=f"{graph.get_name(node)}\\n{mean_probs[node]:.3f}",
            shape="box",
            style="rounded,filled",
            color=to_hex(cmap(0.9)),
            penwidth="3" if node in graph.leaf_ids else "1",
            fontname="Helvetica",
            fontsize="11",
            fontcolor="white" if intensity > 0.5 else "black",
            fillcolor=to_hex(cmap(intensity)),
        )
    for node in shown:
        for child in sorted(c for c in graph.children[node] if c in names):
            dot.edge(names[node], names[child], arrowsize="0.7", color="gray50")
    return dot


def plot_greedy_flow(
    graph: OntologyGraph,
    nodes_df: pd.DataFrame,
    links_df: pd.DataFrame,
    config: GreedyFlowConfig,
):
    """Sankey of `OntologyAnalysis.greedy_flow` tables, root on the left."""
    import plotly.graph_objects as go

    colors = _ontology_colors(graph, nodes_df.loc[~nodes_df.is_other, "cl_id"].tolist())
    rgb = [colors.get(cl_id, "160,160,160") for cl_id in nodes_df["cl_id"]]
    index = {k: i for i, k in enumerate(nodes_df["node_key"])}
    targets = [index[k] for k in links_df["tgt_key"]]
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node={
                "pad": 15,
                "thickness": 20,
                "label": nodes_df["name"].tolist(),
                "color": [f"rgba({c},0.85)" for c in rgb],
                "x": (nodes_df["layer"] / nodes_df["layer"].max()).tolist(),
            },
            link={
                "source": [index[k] for k in links_df["src_key"]],
                "target": targets,
                "value": links_df["count"].tolist(),
                "color": [f"rgba({rgb[t]},0.35)" for t in targets],
            },
        )
    )
    fig.update_layout(
        title_text=f"Greedy flow from {config.root_node}", width=1400, height=800
    )
    return fig


def _ontology_colors(graph: OntologyGraph, ids: list[str]) -> dict[str, str]:
    """Colors in which ontology neighbors look alike: classical MDS to hue/saturation."""
    n = len(ids)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            dist[i, j] = dist[j, i] = min(graph.distance(ids[i], ids[j]), 20.0)
    center = np.eye(n) - 1.0 / n
    values, vectors = np.linalg.eigh(-0.5 * center @ dist**2 @ center)
    coords = np.zeros((n, 2))
    top = vectors[:, ::-1][:, :2] * np.sqrt(np.clip(values[::-1][:2], 0, None))
    coords[:, : top.shape[1]] = top
    coords -= coords.min(axis=0)
    coords /= np.where(coords.max(axis=0) > 0, coords.max(axis=0), 1.0)
    max_height = max(graph.height.values()) or 1
    out = {}
    for (x, y), node in zip(coords, ids):
        lightness = 0.35 + 0.35 * graph.height[node] / max_height
        r, g, b = colorsys.hls_to_rgb(0.85 * x, lightness, 0.4 + 0.5 * y)
        out[node] = f"{int(255 * r)},{int(255 * g)},{int(255 * b)}"
    return out
