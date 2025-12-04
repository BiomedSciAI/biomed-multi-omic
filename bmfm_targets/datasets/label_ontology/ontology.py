from importlib import resources

import igraph
import yaml
from pydantic import validate_call


class LabelOntology:
    ONTOLOGY_ROOT = "bmfm_targets.datasets.label_ontology"

    @classmethod
    def load_ontology(cls, ontology: str) -> "LabelOntology":
        metadata = resources.files(f"{cls.ONTOLOGY_ROOT}._{ontology}").joinpath(
            "metadata.yaml"
        )
        with metadata.open() as fp:
            obj = yaml.safe_load(fp)
        ontology = LabelOntology(ontology, **obj)
        return ontology

    @validate_call
    def __init__(self, ontology: str, node_id: str, node_name: str, unknown_id: str):
        self.node_id = node_id
        self.unknown_id = unknown_id
        self.node_name = node_name
        graph_filename = resources.files(
            f"{LabelOntology.ONTOLOGY_ROOT}._{ontology}"
        ).joinpath("ontology.graphml")
        self.graph: igraph.Graph = igraph.Graph.Read_GraphMLz(graph_filename)

    def get_label_dictionary(self) -> dict[str, int]:
        """Builds label dictionary from graph leaves."""
        node_ids = [
            self.graph.vs[i][self.node_id]
            for i, deg in enumerate(self.graph.degree(mode="out"))
            if deg == 0
        ]
        node_ids.sort()
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
        nodes = self.graph.vs.select(**{f"{self.node_id}_eq": node_id})
        if len(nodes) == 0:
            return []
        target_vertex = nodes[0]
        subgraph = self.graph.subcomponent(target_vertex, mode="out")
        subgraph = self.graph.subgraph(subgraph)
        leaves = [
            subgraph.vs[i][self.node_id]
            for i, deg in enumerate(subgraph.degree(mode="out"))
            if deg == 0
        ]
        return leaves


# if __name__ == "__main__":
#     ont = LabelOntology.load_ontology("cellxgene")
#     print(ont.get_label_dictionary())
# node_ids = ont.find_leaves(node_id="CL:0000226")

# print(node_ids)
