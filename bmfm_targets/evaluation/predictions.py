import pandas as pd
import scanpy as sc
from anndata import AnnData


def _create_general_cell_type_mapping(label_dict: dict, level: str = "sub_class"):
    from cellxgene_ontology_guide.curated_ontology_term_lists import (
        get_curated_ontology_term_list,
    )
    from cellxgene_ontology_guide.entities import CuratedOntologyTermList, Ontology
    from cellxgene_ontology_guide.ontology_parser import OntologyParser
    from cellxgene_ontology_guide.supported_versions import (
        CXGSchema,
        load_supported_versions,
    )

    load_supported_versions()
    cxg_schema = CXGSchema(version="5.3.0")
    cxg_schema.get_ontology_download_url(Ontology.CL)
    ontology_parser = OntologyParser(schema_version="v5.3.0")

    if level == "sub_class":
        cell_terms = get_curated_ontology_term_list(
            CuratedOntologyTermList.CELL_SUBCLASS,
        )
    elif level == "class":
        cell_terms = get_curated_ontology_term_list(
            CuratedOntologyTermList.CELL_CLASS,
        )
    else:
        raise ValueError("Must be class or sub_class")

    no_cl_labels = {}
    specific_cell_types = {}

    for c in label_dict["cell_type"]:
        term = ontology_parser.get_term_id_by_label(c, "CL")
        if term is not None:
            specific_cell_types[c] = term
        else:
            no_cl_labels[c] = c

    mapping_results = ontology_parser.map_high_level_terms(
        term_ids=list(specific_cell_types.values()),
        high_level_terms=cell_terms,
    )

    sub_label_names = {
        t: ontology_parser.get_term_label(mapping_results[t][0])
        if len(mapping_results[t]) > 0
        else ontology_parser.get_term_label(t)
        for t in mapping_results
    }

    general_labels = no_cl_labels
    general_labels.update(
        {l: sub_label_names[t] for l, t in specific_cell_types.items()}
    )

    return level, general_labels


def get_general_cell_type(
    cell_labels: list | AnnData,
    key: str = "cell_type",
    level: str = "cell_type_descendants",
) -> dict:
    label_dict = {"cell_type": cell_labels.obs["cell_type"].unique().tolist()}
    level, general_mapping = _create_general_cell_type_mapping(label_dict, level)

    if isinstance(cell_labels, AnnData):
        cell_labels = cell_labels.copy()
        cell_type_general = [general_mapping[c] for c in cell_labels.obs[key].tolist()]
        cell_labels.obs[f"{level}_{key}"] = cell_type_general
        return cell_labels
    elif isinstance(cell_labels, list):
        cell_type_general = [general_mapping[c] for c in cell_labels]
        return cell_type_general
    else:
        raise ValueError("Must provide a list or adata object with a key.")


def convert_ids_to_label_names(label_dict: dict, label_ids: dict) -> list:
    label_dict = {value: key for key, value in label_dict.items()}
    label_names = [label_dict[c] for c in label_ids]
    return label_names


def get_label_map(key: str, predictions: dict, label_dict: dict) -> dict:
    return {
        v: k
        for k, v in label_dict[key].items()
        if v in predictions[f"{key}_predictions"]
    }


def create_results_obs(results: dict, label_dict: dict | None = None) -> sc.AnnData:
    df_embed = pd.DataFrame(index=results["cell_name"], data=results["embeddings"])
    cell_barcodes = results["cell_name"]
    preds = {
        y: convert_ids_to_label_names(
            label_dict[f"{y.replace('_predictions', '')}"], results[f"{y}"]
        )
        for y in results.keys()
        if "prediction" in y
    }

    obs = pd.DataFrame(preds, index=cell_barcodes)
    adata = sc.AnnData(X=df_embed, obs=obs)

    return adata


def merge_bmfm_adata(bmfm_adata: AnnData, reference_adata: AnnData) -> AnnData:
    bmfm_adata = bmfm_adata.copy()
    reference_adata = reference_adata.copy()
    bmfm_adata.obsm["X_bmfm"] = bmfm_adata.X
    reference_adata.obsm["X_bmfm"] = bmfm_adata.obsm["X_bmfm"]
    reference_adata.obsm["X_umap"] = bmfm_adata.obsm["X_umap"]
    reference_adata.obs = reference_adata.obs.join(
        bmfm_adata.obs, how="left", lsuffix="_bmfm"
    )
    return reference_adata
