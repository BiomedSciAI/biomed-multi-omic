"""Lookups over a cell ontology DAG whose edges point from parent to child."""

from functools import cached_property
from pathlib import Path

import networkx as nx

from bmfm_targets.datasets import label_ontology
from bmfm_targets.datasets.label_ontology import LabelOntology

DEFAULT_ONTOLOGY_PATH = (
    Path(label_ontology.__file__).parent / "_celltypeont" / "ontology.graphml"
)


class OntologyGraph:
    """Ontology keyed by ontology id, with cached ancestor and leaf lookups."""

    def __init__(
        self,
        ontology_path: str | Path | None = None,
        node_id_key: str = "cell_type_id",
        node_name_key: str = "cell_type_name",
    ):
        path = Path(ontology_path or DEFAULT_ONTOLOGY_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"No ontology graph at {path}. The cell ontology is not shipped with "
                "this package; build it as described in "
                "bmfm_targets/datasets/label_ontology/README.md, or set ontology_path."
            )
        raw = LabelOntology._read_graphmlz(path)
        # Nodes and children keep file order, so argmax ties do not depend on hashing.
        order = {node: i for i, node in enumerate(raw)}
        ids = {node: raw.nodes[node][node_id_key] for node in raw}
        self.names = {
            ids[node]: raw.nodes[node].get(node_name_key, ids[node]) for node in raw
        }
        self.children = {
            ids[node]: [ids[c] for c in sorted(raw.successors(node), key=order.get)]
            for node in raw
        }
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.children)
        self.graph.add_edges_from((p, c) for p, cs in self.children.items() for c in cs)
        self.leaf_ids = {n for n, cs in self.children.items() if not cs}
        self._ids_by_name = {name.lower(): n for n, name in self.names.items()}

    def __contains__(self, node_id: str) -> bool:
        return node_id in self.children

    def get_name(self, node_id: str) -> str:
        """Readable name of an id, or the input if it is not in the ontology."""
        return self.names.get(node_id, node_id)

    def resolve(self, node_ref: str) -> str | None:
        """Ontology id for an id or a case-insensitive name, None if unknown."""
        if node_ref in self:
            return node_ref
        return self._ids_by_name.get(str(node_ref).lower())

    @cached_property
    def ancestors(self) -> dict[str, set[str]]:
        """Ancestors of each node, including the node itself."""
        return {n: nx.ancestors(self.graph, n) | {n} for n in self.graph}

    @cached_property
    def descendant_leaves(self) -> dict[str, list[str]]:
        """Leaves below each node; a leaf is its own descendant leaf."""
        out: dict[str, list[str]] = {n: [] for n in self.graph}
        for leaf in self.graph:
            if leaf in self.leaf_ids:
                for ancestor in self.ancestors[leaf]:
                    out[ancestor].append(leaf)
        return out

    @cached_property
    def height(self) -> dict[str, int]:
        """Longest path from each node down to a leaf (leaves are 0)."""
        out: dict[str, int] = {}
        for node in reversed(list(nx.topological_sort(self.graph))):
            out[node] = max((out[c] + 1 for c in self.children[node]), default=0)
        return out

    @cached_property
    def leaf_distance(self) -> dict[str, int]:
        """Shortest path from each node down to a leaf (leaves are 0)."""
        out: dict[str, int] = {}
        for node in reversed(list(nx.topological_sort(self.graph))):
            out[node] = min((out[c] + 1 for c in self.children[node]), default=0)
        return out

    @cached_property
    def depth(self) -> dict[str, int]:
        """Longest path from a root down to each node (roots are 0)."""
        out: dict[str, int] = {}
        for node in nx.topological_sort(self.graph):
            parents = self.graph.predecessors(node)
            out[node] = max((out[p] + 1 for p in parents), default=0)
        return out

    def relationship(self, pred_id: str, true_id: str) -> str:
        """Relation of the prediction to the truth: exact, ancestor, descendant or unrelated."""
        if pred_id == true_id:
            return "exact"
        if pred_id not in self or true_id not in self:
            return "unrelated"
        if pred_id in self.ancestors[true_id]:
            return "ancestor"
        if true_id in self.ancestors[pred_id]:
            return "descendant"
        return "unrelated"

    def distance(self, id_a: str, id_b: str) -> float:
        """Path length between two nodes through their deepest common ancestor."""
        common = self.ancestors[id_a] & self.ancestors[id_b]
        if not common:
            return float("inf")
        lca = max(common, key=lambda n: (self.depth[n], n))
        return float(
            nx.shortest_path_length(self.graph, lca, id_a)
            + nx.shortest_path_length(self.graph, lca, id_b)
        )
