#!/usr/bin/env python

from collections.abc import Iterable
from importlib import resources

import cellxgene_census
import click
import igraph
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


def build_graph(ontology_filename: str) -> igraph.Graph:
    """Converts ontology to a igraph graph and removes obsolete ontology ids."""
    ontology = pronto.Ontology(ontology_filename)
    non_obsolete_terms = [
        i for i in ontology.terms() if not i.obsolete and i.id != UNKNOWN_ID
    ]
    ids = [i.id for i in non_obsolete_terms]
    names = [i.name for i in non_obsolete_terms]
    definitions = [i.definition for i in non_obsolete_terms]

    id2index = {i: index for index, i in enumerate(ids)}
    n_nodes = len(ids)
    edges = []
    for term in non_obsolete_terms:
        for i in term.superclasses(distance=1):
            if i.id != term.id:
                edges.append([id2index[i.id], id2index[term.id]])
    graph = igraph.Graph(n_nodes, edges, directed=True)
    graph.vs[CELL_TYPE_ID] = ids
    graph.vs[CELL_TYPE_NAME] = names
    graph.vs[CELL_TYPE_DEFINITION] = definitions

    return graph


def remove_obsolete(terms: Iterable[pronto.term.Term]):
    return [i for i in terms if not i.obsolete]


def remove_dead_leaves(graph, cxg_ids: set[str]):
    """Removes leaves from the graph until all graphs leaves are from CELLxGENE."""
    found_dead_leaves = True
    while found_dead_leaves:
        leaves = {
            graph.vs[i][CELL_TYPE_ID]: i
            for i, deg in enumerate(graph.degree(mode="out"))
            if deg == 0
        }
        leaf_set = set(leaves)
        leaf_set -= cxg_ids
        if leaf_set:
            to_remove = [leaves[i] for i in leaf_set]
            graph.delete_vertices(to_remove)
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
    n_leaves = len([i for i in graph.degree(mode="out") if i == 0])
    print(f"Number of vertices in the final graph: {graph.vcount()}")
    print(f"Number of leaves in the final graph: {n_leaves}")

    graph.write_graphmlz(output_graph_filename)


if __name__ == "__main__":
    cli()
