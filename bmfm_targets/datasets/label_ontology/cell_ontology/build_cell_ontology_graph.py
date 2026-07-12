#!/usr/bin/env python

import gzip
from collections.abc import Iterable
from importlib import resources

import cellxgene_census
import click
import networkx as nx
import pronto

UNKNOWN_ID = "unknown"
CELL_TYPE_ID = "cell_type_id"
CELL_TYPE_NAME = "cell_type_name"
CELL_TYPE_DEFINITION = "cell_type_definition"

CELL_TYPE_COLUMN = "cell_type"
CELL_TYPE_ONTOLOGY_TERM_ID_COLUMN = "cell_type_ontology_term_id"


def check_if_cellxgene_subset_of_ontology(
    ontology_filename: str,
    organism: str = "homo_sapiens",
    census_uri: str = None,
    census_version="2025-01-30",
) -> None:
    """Test if ontology covers all ids in CELLxGENE and creates map from cell_type_ontology_term_id to cell_type."""
    cl_ontology = pronto.Ontology(ontology_filename)
    all_terms = list(cl_ontology.terms())
    cl_ids = [i.id for i in all_terms]
    cl_ids = set(cl_ids)
    print(f"Ontology contains {len(cl_ids)} of cell ids.")

    with cellxgene_census.open_soma(census_version=census_version) as census:
        obs = census["census_data"][organism].obs
        tbl = (
            obs.read(
                value_filter=None,
                column_names=[CELL_TYPE_COLUMN, CELL_TYPE_ONTOLOGY_TERM_ID_COLUMN],
            )
            .concat()
            .to_pandas()
        )

    cxg_ids = sorted(
        x for x in tbl[CELL_TYPE_ONTOLOGY_TERM_ID_COLUMN].dropna().unique()
    )
    cxg_ids = set(cxg_ids)
    if UNKNOWN_ID in cxg_ids:
        cxg_ids.remove(UNKNOWN_ID)

    diff = cxg_ids - cl_ids
    if diff:
        raise ValueError(f"CELLxGENE cell types are not in ontology {diff}")


def get_cxg_cell(
    organism: str = "homo_sapiens", census_uri: str = None, census_version="2025-01-30"
) -> None:
    """Returns all cell types in CELLxGENE."""
    with cellxgene_census.open_soma(census_version=census_version) as census:
        obs = census["census_data"][organism].obs
        tbl = (
            obs.read(
                value_filter=None,
                column_names=[CELL_TYPE_COLUMN, CELL_TYPE_ONTOLOGY_TERM_ID_COLUMN],
            )
            .concat()
            .to_pandas()
        )

    cxg_ids = sorted(
        x for x in tbl[CELL_TYPE_ONTOLOGY_TERM_ID_COLUMN].dropna().unique()
    )
    cxg_ids = set(cxg_ids)
    return cxg_ids


def default_ontology_filename():
    filename = str(
        resources.files("bmfm_targets.datasets.label_ontology.cell_ontology").joinpath(
            "cl-basic.obo"
        )
    )
    return filename


def build_graph(ontology_filename: str) -> nx.DiGraph:
    """Converts ontology to a networkx graph and removes obsolete ontology ids."""
    ontology = pronto.Ontology(ontology_filename)
    non_obsolete_terms = [
        i for i in ontology.terms() if not i.obsolete and i.id != UNKNOWN_ID
    ]
    ids = {i.id for i in non_obsolete_terms}

    graph = nx.DiGraph()
    for term in non_obsolete_terms:
        graph.add_node(
            term.id,
            **{
                CELL_TYPE_ID: term.id,
                CELL_TYPE_NAME: term.name or "",
                CELL_TYPE_DEFINITION: str(term.definition) if term.definition else "",
            },
        )
    # Edges point superclass (parent) -> term (child), matching the previous
    # igraph construction; obsolete superclasses are not part of the graph.
    for term in non_obsolete_terms:
        for superclass in term.superclasses(distance=1):
            if superclass.id != term.id and superclass.id in ids:
                graph.add_edge(superclass.id, term.id)

    return graph


def write_graphmlz(graph: nx.DiGraph, filename: str) -> None:
    """Write a directed graph to a gzip-compressed GraphML file."""
    with gzip.open(filename, "wb") as fp:
        nx.write_graphml(graph, fp)


def remove_obsolete(terms: Iterable[pronto.term.Term]):
    return [i for i in terms if not i.obsolete]


def remove_dead_leaves(graph: nx.DiGraph, cxg_ids: set[str]):
    """Removes leaves from the graph until all graphs leaves are from CELLxGENE."""
    found_dead_leaves = True
    while found_dead_leaves:
        to_remove = [
            node
            for node in graph
            if graph.out_degree(node) == 0
            and graph.nodes[node][CELL_TYPE_ID] not in cxg_ids
        ]
        if to_remove:
            graph.remove_nodes_from(to_remove)
            found_dead_leaves = True
        else:
            found_dead_leaves = False


@click.command(
    help="""
               Builds cell ontology graphs and saves into graphmlz format.
               Load cell type ids from CELLxGENE and removes all leaves that are not in CELLxGENE.
        """.strip()
)
@click.option(
    "--organism",
    "-o",
    default="homo_sapiens",
    help="Organism name (default: homo_sapiens)",
)
@click.option("--census-uri", "-u", default=None, help="Census URI (optional)")
@click.option(
    "--census-version",
    "-v",
    default="2025-01-30",
    help="Census version (default: 2025-01-30)",
)
@click.option(
    "--output",
    "-f",
    default="ontology.graphml",
    help="Output graph filename (default: cell_ontology.graphml)",
)
@click.option(
    "--ontology_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default=default_ontology_filename(),
    help="Ontology file name",
)
@click.pass_context
def cli(ctx, organism, census_uri, census_version, output, ontology_file):
    click.echo(ctx.get_help())
    click.echo("\n" + "=" * 50 + "\n")
    click.echo("Running ...")

    build_ontology_graph(
        organism=organism,
        census_uri=census_uri,
        census_version=census_version,
        output_graph_filename=output,
        ontology_filename=ontology_file,
    )


def build_ontology_graph(
    organism: str,
    census_uri: str,
    census_version,
    output_graph_filename,
    ontology_filename,
):
    """Build an ontology graph with specified parameters."""
    graph = build_graph(ontology_filename)
    check_if_cellxgene_subset_of_ontology(
        ontology_filename, organism, census_uri, census_version
    )
    cxg_cells = get_cxg_cell(organism, census_uri, census_version)
    remove_dead_leaves(graph, cxg_cells)
    n_leaves = sum(1 for node in graph if graph.out_degree(node) == 0)
    print(f"Number of vertices in the final graph: {graph.number_of_nodes()}")
    print(f"Number of leaves in the final graph: {n_leaves}")

    write_graphmlz(graph, output_graph_filename)


if __name__ == "__main__":
    cli()
