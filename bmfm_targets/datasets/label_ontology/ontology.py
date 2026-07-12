import gzip
from importlib import resources
from importlib.resources import as_file
from pathlib import Path

import networkx as nx
import yaml
from pydantic import validate_call


class LabelOntology:
    ONTOLOGY_ROOT = "bmfm_targets.datasets.label_ontology"

    @classmethod
    def load_ontology(cls, ontology: str) -> "LabelOntology":
        """Load a packaged ontology by name from its ``_<ontology>`` resource dir."""
        package = f"{cls.ONTOLOGY_ROOT}._{ontology}"
        with resources.files(package).joinpath("metadata.yaml").open() as fp:
            metadata = yaml.safe_load(fp)
        with as_file(
            resources.files(package).joinpath("ontology.graphml")
        ) as graph_filename:
            return cls(graph_filename=graph_filename, **metadata)

    @classmethod
    def from_directory(cls, directory: str | Path) -> "LabelOntology":
        """Load an ontology from a directory holding metadata.yaml and ontology.graphml."""
        directory = Path(directory)
        with (directory / "metadata.yaml").open() as fp:
            metadata = yaml.safe_load(fp)
        return cls(graph_filename=directory / "ontology.graphml", **metadata)

    @validate_call
    def __init__(
        self,
        node_id: str,
        node_name: str,
        unknown_id: str,
        graph_filename: str | Path,
    ):
        self.node_id = node_id
        self.unknown_id = unknown_id
        self.node_name = node_name
        self.graph: nx.DiGraph = self._read_graphmlz(graph_filename)
        # Index ontology id -> graph node key once, for O(1) lookups in find_leaves.
        self._id_to_node: dict[str, str] = {}
        for node, data in self.graph.nodes(data=True):
            ontology_id = data.get(node_id)
            if ontology_id is not None:
                self._id_to_node.setdefault(ontology_id, node)

    @staticmethod
    def _read_graphmlz(graph_filename: str | Path) -> nx.DiGraph:
        """Read a gzip-compressed GraphML file into a directed graph."""
        with gzip.open(str(graph_filename), "rb") as fp:
            graph = nx.read_graphml(fp)
        if not graph.is_directed():
            graph = graph.to_directed()
        return graph

    def get_label_dictionary(self) -> dict[str, int]:
        """Builds label dictionary from graph leaves."""
        node_ids = sorted(
            data[self.node_id]
            for node, data in self.graph.nodes(data=True)
            if self.graph.out_degree(node) == 0
        )
        label_dictionary = {node_id: index for index, node_id in enumerate(node_ids)}
        return label_dictionary

    def find_leaves(self, node_id: str) -> list[str]:
        """
        Finds ontology ids of leaves of ontology subgraph with node_id as the root.

        Args:
        ----
        node_id : root id, e.g., in cell ontology, it is cell_type_ontology_term_id of the cell. Example - "CL:0000226".

        Returns:
        -------
        A list of ontology ids for leaves of the subgraph.

        """
        root = self._id_to_node.get(node_id)
        if root is None:
            return []
        # The descendant closure is downward-closed, so a reachable node's
        # children are all reachable too; out-degree in the full graph therefore
        # equals out-degree in the induced subgraph, and out-degree 0 ==> leaf.
        reachable = nx.descendants(self.graph, root)
        reachable.add(root)
        return [
            self.graph.nodes[node][self.node_id]
            for node in reachable
            if self.graph.out_degree(node) == 0
        ]
