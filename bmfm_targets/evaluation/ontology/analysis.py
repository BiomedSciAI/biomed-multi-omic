"""Hierarchical cell type prediction from leaf probabilities, and its metrics."""

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from .configs import GreedyFlowConfig
from .graph import OntologyGraph

logger = logging.getLogger(__name__)


class OntologyAnalysis:
    """Rolls leaf probabilities up an ontology and decodes them."""

    def __init__(self, graph: OntologyGraph):
        self.graph = graph

    def compute_node_probabilities(
        self,
        probabilities_df: pd.DataFrame,
        label_column: str = "cell_type_ontology_term_id",
    ) -> dict[str, np.ndarray]:
        """
        Sum leaf probabilities into every ontology node.

        Args:
        ----
            probabilities_df: one row per cell, columns `{label_column}_{ontology id}`.
                Columns that are not leaves of the ontology are ignored.
            label_column: label column of the model head.

        Returns:
        -------
            One array of per-cell probabilities for each node of the ontology.
        """
        prefix = f"{label_column}_"
        leaves = [
            c.removeprefix(prefix)
            for c in probabilities_df.columns
            if c.startswith(prefix) and c.removeprefix(prefix) in self.graph.leaf_ids
        ]
        if not leaves:
            raise ValueError(
                f"No column named '{prefix}<leaf id>' matches a leaf of the ontology."
            )
        probs = probabilities_df[[prefix + leaf for leaf in leaves]].to_numpy()
        position = {leaf: i for i, leaf in enumerate(leaves)}
        node_probs = {}
        for node, below in self.graph.descendant_leaves.items():
            idx = [position[leaf] for leaf in below if leaf in position]
            node_probs[node] = probs[:, idx].sum(axis=1)
        return node_probs

    def predict(
        self, node_probs: dict[str, np.ndarray], threshold: float
    ) -> np.ndarray:
        """
        Most specific confident label of each cell, searching from the leaves up.

        Levels are shortest distances to a leaf. A cell takes the most probable node of
        the lowest level at which some node reaches the threshold. If no node reaches it, the cell takes the
        most probable node overall.
        """
        levels = defaultdict(list)
        for node in node_probs:
            levels[self.graph.leaf_distance[node]].append(node)
        n_cells = len(next(iter(node_probs.values())))
        pred = np.empty(n_cells, dtype=object)
        remaining = np.ones(n_cells, dtype=bool)
        fallback_prob = np.full(n_cells, -np.inf)
        fallback = np.empty(n_cells, dtype=object)
        for level in sorted(levels):
            nodes = np.asarray(levels[level], dtype=object)
            # float32 so that float64 rounding in the rollup does not split ties
            mat = np.column_stack([node_probs[n] for n in nodes]).astype(np.float32)
            best = mat.argmax(axis=1)
            best_prob = mat[np.arange(n_cells), best]
            take = remaining & (best_prob >= threshold)
            pred[take] = nodes[best[take]]
            remaining &= ~take
            better = best_prob > fallback_prob
            fallback[better] = nodes[best[better]]
            fallback_prob[better] = best_prob[better]
        pred[remaining] = fallback[remaining]
        return pred

    def predict_topdown(
        self,
        node_probs: dict[str, np.ndarray],
        threshold: float,
        root_node: str = "cell",
    ) -> np.ndarray:
        """
        Follow the most probable child from the root while it reaches the threshold.

        Higher thresholds stop closer to the root and give more general labels.
        """
        root = self.graph.resolve(root_node)
        if root is None:
            raise ValueError(f"Root node '{root_node}' is not in the ontology.")
        pred = np.full(len(node_probs[root]), root, dtype=object)
        active = {root: np.arange(len(pred))}
        while active:
            following = defaultdict(list)
            for node, cells in active.items():
                children = self.graph.children[node]
                if not children:
                    continue
                mat = np.column_stack([node_probs[c][cells] for c in children])
                best = mat.argmax(axis=1)
                descends = mat.max(axis=1) >= threshold
                for j in np.unique(best[descends]):
                    moved = cells[descends & (best == j)]
                    pred[moved] = children[j]
                    following[children[j]].append(moved)
            active = {n: np.concatenate(c) for n, c in following.items()}
        return pred

    def uncertainty(
        self, node_probs: dict[str, np.ndarray], pred: np.ndarray
    ) -> np.ndarray:
        """Entropy over the leaves below each predicted node, scaled to [0, 1]."""
        out = np.zeros(len(pred))
        # A leaf with no head column has zero probability in every cell.
        observed = {n for n in self.graph.leaf_ids if node_probs[n].any()}
        for node in pd.unique(pred):
            leaves = [n for n in self.graph.descendant_leaves[node] if n in observed]
            if len(leaves) < 2:
                continue
            cells = np.flatnonzero(pred == node)
            p = np.column_stack([node_probs[leaf][cells] for leaf in leaves])
            p = np.clip(p / p.sum(axis=1, keepdims=True), 1e-10, 1.0)
            out[cells] = -(p * np.log(p)).sum(axis=1) / np.log(len(leaves))
        return out

    def hierarchical_metrics(self, pred: np.ndarray, true: np.ndarray) -> dict:
        """
        Accuracy with credit for ontology relations, over cells with a known true id.

        `hierarchical_accuracy` counts exact matches and descendants of the true label.
        `hprecision`, `hrecall` and `hf1` compare ancestor sets, averaged over cells.
        """
        pairs = pd.DataFrame({"pred": pred, "true": true})
        pairs = pairs[pairs["true"].isin(self.graph.children)]
        counts = pairs.value_counts()
        n_valid = int(counts.sum())
        totals = dict.fromkeys(("exact", "ancestor", "descendant", "unrelated"), 0)
        precision = recall = f1 = 0.0
        for (pred_id, true_id), n in counts.items():
            totals[self.graph.relationship(pred_id, true_id)] += n
            anc_pred = self.graph.ancestors[pred_id]
            anc_true = self.graph.ancestors[true_id]
            shared = len(anc_pred & anc_true)
            precision += n * shared / len(anc_pred)
            recall += n * shared / len(anc_true)
            f1 += n * 2 * shared / (len(anc_pred) + len(anc_true))
        denom = n_valid or np.nan
        return {
            "exact_accuracy": totals["exact"] / denom,
            "ancestor_rate": totals["ancestor"] / denom,
            "descendant_rate": totals["descendant"] / denom,
            "hierarchical_accuracy": (totals["exact"] + totals["descendant"]) / denom,
            "hprecision": precision / denom,
            "hrecall": recall / denom,
            "hf1": f1 / denom,
            "n_valid": n_valid,
        }

    @staticmethod
    def clustering_metrics(
        pred: np.ndarray,
        true: np.ndarray,
        embeddings: np.ndarray | None = None,
        max_samples_asw: int = 50_000,
    ) -> dict:
        """NMI and ARI of predictions against truth, and the silhouette of predictions."""
        from sklearn.metrics import (
            adjusted_rand_score,
            normalized_mutual_info_score,
            silhouette_score,
        )

        known = pd.notna(true)
        pred, true = pred[known].astype(str), true[known].astype(str)
        out = {
            "nmi": normalized_mutual_info_score(true, pred),
            "ari": adjusted_rand_score(true, pred),
            "asw": np.nan,
        }
        if embeddings is not None:
            # Labels with a single cell have no silhouette.
            keep = pd.Series(pred).groupby(pred).transform("size").to_numpy() > 1
            if len(set(pred[keep])) > 1:
                out["asw"] = silhouette_score(
                    embeddings[known][keep],
                    pred[keep],
                    sample_size=min(max_samples_asw, int(keep.sum())),
                    random_state=0,
                )
        out["avg_bio"] = np.nanmean(list(out.values()))
        return {k: float(v) for k, v in out.items()}

    def greedy_flow(
        self, node_probs: dict[str, np.ndarray], config: GreedyFlowConfig
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Cell counts along greedy root-to-leaf paths.

        Each cell starts at the root and moves to its most probable child until it
        reaches a leaf. Nodes are placed in layers by height, so leaves share the last
        layer. Nodes outside the `max_nodes_per_layer` largest of their layer, and the
        targets of links below `min_flow_fraction` of cells, are merged into one
        "Other" node per layer.

        Returns
        -------
            Node table (node_key, cl_id, layer, count, name, is_other) and link table
            (src_key, tgt_key, src_layer, tgt_layer, count, mean_selected_prob).
        """
        root = self.graph.resolve(config.root_node)
        if root is None:
            raise ValueError(f"Root node '{config.root_node}' is not in the ontology.")
        n_layers = max(self.graph.height.values())
        layer = {n: n_layers - h for n, h in self.graph.height.items()}
        n_cells = len(node_probs[root])

        def key(node):
            return f"L{layer[node]}|{node}"

        visits: dict[str, int] = defaultdict(int)
        links = []
        active = {root: np.arange(n_cells)}
        while active:
            following = defaultdict(list)
            for node, cells in active.items():
                visits[node] += len(cells)
                children = self.graph.children[node]
                if not children:
                    continue
                choice = np.column_stack(
                    [node_probs[c][cells] for c in children]
                ).argmax(axis=1)
                for j in np.unique(choice):
                    child, moved = children[j], cells[choice == j]
                    following[child].append(moved)
                    links.append(
                        (
                            key(node),
                            key(child),
                            layer[node],
                            layer[child],
                            len(moved),
                            node_probs[child][moved].sum(),
                        )
                    )
            active = {n: np.concatenate(c) for n, c in following.items()}

        links_df = pd.DataFrame(
            links,
            columns=["src_key", "tgt_key", "src_layer", "tgt_layer", "count", "mass"],
        )
        nodes_df = pd.DataFrame(
            {
                "node_key": [key(n) for n in visits],
                "cl_id": list(visits),
                "layer": [layer[n] for n in visits],
                "count": list(visits.values()),
                "name": [self.graph.get_name(n) for n in visits],
                "is_other": False,
            }
        )

        def other_key(lay):
            return f"L{lay}|OTHER@L{lay}"

        if config.max_nodes_per_layer is not None:
            kept = nodes_df.groupby("layer")["count"].rank(
                method="first", ascending=False
            )
            dropped = nodes_df[
                (kept > config.max_nodes_per_layer) & (nodes_df["cl_id"] != root)
            ]
            remap = {k: other_key(v) for k, v in zip(dropped.node_key, dropped.layer)}
            links_df["src_key"] = links_df["src_key"].replace(remap)
            links_df["tgt_key"] = links_df["tgt_key"].replace(remap)
            nodes_df = nodes_df.drop(dropped.index)
        links_df = _merge_links(links_df)
        weak = links_df["count"] < np.ceil(config.min_flow_fraction * n_cells)
        links_df.loc[weak, "tgt_key"] = links_df.loc[weak, "tgt_layer"].map(other_key)
        links_df = _merge_links(links_df)

        others = sorted(
            {
                (lay, k)
                for k, lay in zip(
                    [*links_df.src_key, *links_df.tgt_key],
                    [*links_df.src_layer, *links_df.tgt_layer],
                )
                if k == other_key(lay)
            }
        )
        if others:
            other_df = pd.DataFrame(
                {
                    "node_key": [k for _, k in others],
                    "cl_id": [k.split("|")[1] for _, k in others],
                    "layer": [lay for lay, _ in others],
                    "count": 0,
                    "name": [f"Other (L{lay})" for lay, _ in others],
                    "is_other": True,
                }
            )
            nodes_df = pd.concat([nodes_df, other_df], ignore_index=True)

        incoming = links_df.groupby("tgt_key")["count"].sum()
        nodes_df["count"] = nodes_df["node_key"].map(incoming).fillna(0).astype(int)
        nodes_df.loc[nodes_df["cl_id"] == root, "count"] = n_cells
        links_df["mean_selected_prob"] = links_df["mass"] / links_df["count"]
        return nodes_df.reset_index(drop=True), links_df.drop(columns="mass")


def _merge_links(links_df: pd.DataFrame) -> pd.DataFrame:
    keys = ["src_key", "tgt_key", "src_layer", "tgt_layer"]
    return links_df.groupby(keys, as_index=False)[["count", "mass"]].sum()
