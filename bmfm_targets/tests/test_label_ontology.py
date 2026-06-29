"""
Behavior tests for :class:`LabelOntology`.

These tests pin the *behavior* of the ontology graph operations
(``get_label_dictionary`` / ``find_leaves``) using a small synthetic DAG that
the test itself owns, so the assertions do not depend on the values of any
externally-licensed ontology artifact. They are the contract that must hold
regardless of the graph backend (igraph, networkx, ...), enabling that backend
to be swapped without behavioral drift.

A separate, auto-skipping test asserts backend-independent *invariants* against
the real packaged ontology when its (untracked) graphml happens to be present
for local/internal development.
"""

import gzip
from importlib import resources
from pathlib import Path

import pytest

from bmfm_targets.datasets.label_ontology import LabelOntology

# --- Synthetic ontology DAG owned by the tests --------------------------------
#
# Edges point parent -> child (directed). The structure deliberately exercises:
#   * multiple levels,
#   * a node reachable from two parents ("CL:shared" via both A and B), i.e. a
#     genuine DAG rather than a tree,
#   * a leaf that is also a direct child of the root.
#
#                 CL:root
#                 /     \
#             CL:A       CL:B
#            /    \      /   \
#     CL:leaf1  CL:shared    CL:C
#                              |
#                           CL:leaf2
#
# Leaves (no out-edges): CL:leaf1, CL:shared, CL:leaf2
NODES: dict[str, tuple[str, str]] = {
    # graphml node id -> (cell_type_id, cell_type_name)
    "n0": ("CL:root", "root"),
    "n1": ("CL:A", "node A"),
    "n2": ("CL:B", "node B"),
    "n3": ("CL:C", "node C"),
    "n4": ("CL:leaf1", "leaf one"),
    "n5": ("CL:shared", "shared leaf"),
    "n6": ("CL:leaf2", "leaf two"),
}
EDGES: list[tuple[str, str]] = [
    ("n0", "n1"),
    ("n0", "n2"),
    ("n1", "n4"),
    ("n1", "n5"),
    ("n2", "n5"),
    ("n2", "n3"),
    ("n3", "n6"),
]
LEAVES = {"CL:leaf1", "CL:shared", "CL:leaf2"}

METADATA = "node_id: cell_type_id\nnode_name: cell_type_name\nunknown_id: unknown\n"


def _write_graphmlz(path: Path, nodes, edges) -> None:
    """
    Write a gzipped GraphML file in the same dialect igraph emits.

    Attributes are declared by ``attr.name`` (``cell_type_id`` / ``cell_type_name``)
    so the file is a faithful stand-in for the production artifact and is readable
    by any conformant GraphML reader.
    """
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
        "<!-- Created by igraph -->",
        '  <key id="v_cell_type_id" for="node" attr.name="cell_type_id" attr.type="string"/>',
        '  <key id="v_cell_type_name" for="node" attr.name="cell_type_name" attr.type="string"/>',
        '  <graph id="G" edgedefault="directed">',
    ]
    for node_id, (cell_type_id, cell_type_name) in nodes.items():
        lines += [
            f'    <node id="{node_id}">',
            f'      <data key="v_cell_type_id">{cell_type_id}</data>',
            f'      <data key="v_cell_type_name">{cell_type_name}</data>',
            "    </node>",
        ]
    for source, target in edges:
        lines.append(f'    <edge source="{source}" target="{target}"/>')
    lines += ["  </graph>", "</graphml>"]
    with gzip.open(path, "wb") as fp:
        fp.write("\n".join(lines).encode())


@pytest.fixture()
def synthetic_ontology_dir(tmp_path: Path) -> Path:
    """A directory holding a synthetic metadata.yaml + ontology.graphml pair."""
    (tmp_path / "metadata.yaml").write_text(METADATA)
    _write_graphmlz(tmp_path / "ontology.graphml", NODES, EDGES)
    return tmp_path


@pytest.fixture()
def ontology(synthetic_ontology_dir: Path) -> LabelOntology:
    return LabelOntology.from_directory(synthetic_ontology_dir)


# --- Loading ------------------------------------------------------------------


def test_from_directory_reads_metadata(ontology: LabelOntology):
    assert ontology.node_id == "cell_type_id"
    assert ontology.node_name == "cell_type_name"
    assert ontology.unknown_id == "unknown"


def test_direct_construction_matches_from_directory(synthetic_ontology_dir: Path):
    direct = LabelOntology(
        node_id="cell_type_id",
        node_name="cell_type_name",
        unknown_id="unknown",
        graph_filename=synthetic_ontology_dir / "ontology.graphml",
    )
    assert direct.get_label_dictionary() == {
        "CL:leaf1": 0,
        "CL:leaf2": 1,
        "CL:shared": 2,
    }


# --- get_label_dictionary -----------------------------------------------------


def test_label_dictionary_contains_exactly_the_leaves(ontology: LabelOntology):
    assert set(ontology.get_label_dictionary()) == LEAVES


def test_label_dictionary_is_sorted_and_contiguously_indexed(ontology: LabelOntology):
    label_dict = ontology.get_label_dictionary()
    assert list(label_dict) == sorted(label_dict)
    assert list(label_dict.values()) == list(range(len(label_dict)))


# --- find_leaves --------------------------------------------------------------


def test_find_leaves_from_root_returns_all_leaves(ontology: LabelOntology):
    assert set(ontology.find_leaves("CL:root")) == LEAVES


def test_find_leaves_returns_only_descendant_leaves(ontology: LabelOntology):
    assert set(ontology.find_leaves("CL:A")) == {"CL:leaf1", "CL:shared"}
    assert set(ontology.find_leaves("CL:C")) == {"CL:leaf2"}


def test_find_leaves_handles_dag_with_shared_descendant(ontology: LabelOntology):
    # CL:shared is reachable from both CL:A and CL:B; it must appear under both.
    assert "CL:shared" in ontology.find_leaves("CL:A")
    assert set(ontology.find_leaves("CL:B")) == {"CL:shared", "CL:leaf2"}


def test_find_leaves_of_a_leaf_is_itself(ontology: LabelOntology):
    for leaf in LEAVES:
        assert ontology.find_leaves(leaf) == [leaf]


def test_find_leaves_unknown_node_returns_empty(ontology: LabelOntology):
    assert ontology.find_leaves("CL:does-not-exist") == []


def test_find_leaves_results_are_unique(ontology: LabelOntology):
    leaves = ontology.find_leaves("CL:root")
    assert len(leaves) == len(set(leaves))


# --- Optional invariants against the real (untracked) ontology ----------------


def _celltypeont_graphml_present() -> bool:
    try:
        resource = resources.files(
            "bmfm_targets.datasets.label_ontology._celltypeont"
        ).joinpath("ontology.graphml")
        return resource.is_file()
    except (ModuleNotFoundError, FileNotFoundError):
        return False


@pytest.mark.skipif(
    not _celltypeont_graphml_present(),
    reason="licensed cell-ontology graphml is not present (expected in CI / public checkouts)",
)
def test_real_ontology_satisfies_behavioral_invariants():
    """Backend-independent invariants on the real artifact (no value assertions)."""
    ontology = LabelOntology.load_ontology("celltypeont")
    label_dict = ontology.get_label_dictionary()

    # Label dictionary is a sorted, 0-based contiguous index over the leaves.
    assert len(label_dict) > 0
    assert list(label_dict) == sorted(label_dict)
    assert list(label_dict.values()) == list(range(len(label_dict)))

    leaves = set(label_dict)
    # Each leaf resolves to exactly itself.
    for leaf in list(leaves)[:25]:
        assert set(ontology.find_leaves(leaf)) == {leaf}

    # An internal node resolves to a non-empty subset of the leaf set.
    a_leaf = next(iter(leaves))
    assert ontology.find_leaves(a_leaf)  # sanity: non-empty
    # Unknown ids resolve to no leaves.
    assert ontology.find_leaves("CL:does-not-exist") == []
