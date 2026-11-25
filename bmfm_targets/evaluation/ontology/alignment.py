"""
Agreement between embedding similarity and ontology similarity of cell types.

Ontology similarity is personalized PageRank (PPR) on the undirected ontology graph.
Embedding similarity is the cosine between type vectors, which are either mean cell
embeddings or the classifier weight vectors of a checkpoint. Pairs of types share
types, so the pairs are not independent and analytic p-values of their correlation do
not hold; significance comes from permuting the types.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .configs import AlignmentConfig
from .graph import OntologyGraph

logger = logging.getLogger(__name__)

MIN_TYPES = 4


def ppr_similarity(
    graph: OntologyGraph, types: Sequence[str], alpha: float = 0.85
) -> np.ndarray:
    """
    Symmetric PPR similarity of ontology nodes.

    The PPR scores of both directions are averaged and the diagonal is set to 1. Rows
    are then scaled to unit length and the matrix is averaged with its transpose.

    Args:
    ----
        graph: the ontology.
        types: ontology ids, in the order of the returned matrix.
        alpha: PageRank damping; higher values reach further from the start node.

    Returns:
    -------
        Matrix of shape (n_types, n_types) with ones on the diagonal.
    """
    undirected = graph.graph.to_undirected()
    ppr = np.empty((len(types), len(types)))
    for i, node in enumerate(types):
        # The default tolerance scales with graph size and stops before rank order settles.
        scores = nx.pagerank(
            undirected, alpha, personalization={node: 1.0}, tol=1e-14, max_iter=2000
        )
        ppr[i] = [scores[t] for t in types]
    sim = 0.5 * (ppr + ppr.T)
    np.fill_diagonal(sim, 1.0)
    sim /= np.linalg.norm(sim, axis=1, keepdims=True)
    sim = (0.5 * (sim + sim.T)).round(10)
    np.fill_diagonal(sim, 1.0)
    return sim


def type_centroids(
    graph: OntologyGraph, embeddings: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """Mean embedding of each label that is in the ontology, with the sorted labels."""
    types = sorted({t for t in labels if isinstance(t, str) and t in graph})
    return (
        np.array([embeddings[labels == t].mean(axis=0) for t in types]),
        types,
    )


def evaluate_alignment(
    graph: OntologyGraph,
    vectors: np.ndarray,
    types: Sequence[str],
    config: AlignmentConfig,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Compare the cosine similarity of type vectors with ontology similarity.

    Args:
    ----
        graph: the ontology.
        vectors: one row per type.
        types: ontology id of each row.
        config: PPR, resampling and neighborhood settings.

    Returns:
    -------
        Summary metrics, a table of type pairs with both similarities, and a table of
        neighborhood metrics per type.
    """
    n = len(types)
    if n < MIN_TYPES:
        raise ValueError(f"Alignment needs at least {MIN_TYPES} types, got {n}.")
    emb_sim = _cosine(vectors)
    ont_sim = ppr_similarity(graph, types, config.alpha)
    rows, cols = np.triu_indices(n, k=1)
    pairs = pd.DataFrame(
        {
            "type_a": np.asarray(types)[rows],
            "type_b": np.asarray(types)[cols],
            "ontology_similarity": ont_sim[rows, cols],
            "embedding_similarity": emb_sim[rows, cols],
        }
    )
    k = neighborhood_size(n, config.knn_k)
    per_type = neighborhood_metrics(ont_sim, emb_sim, k)
    per_type.insert(0, "type", list(types))
    per_type.insert(1, "name", [graph.get_name(t) for t in types])

    observed = _correlations(ont_sim, emb_sim)
    observed |= per_type[["jaccard", "recall", "map"]].mean().to_dict()
    metrics = {"n_types": n, "n_pairs": len(pairs), "k": k, **observed}
    if config.bootstrap_reps:
        metrics |= bootstrap_interval(
            ont_sim, emb_sim, config.bootstrap_reps, config.ci_level, config.seed
        )
    if config.perm_reps:
        metrics |= permutation_test(
            ont_sim, emb_sim, observed, k, config.perm_reps, config.seed
        )
    return metrics, pairs, per_type


def alpha_sweep(
    graph: OntologyGraph,
    vectors: np.ndarray,
    types: Sequence[str],
    alphas: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95),
) -> pd.DataFrame:
    """Correlation of embedding and ontology similarity at several PPR dampings."""
    emb_sim = _cosine(vectors)
    return pd.DataFrame(
        {"alpha": a, **_correlations(ppr_similarity(graph, types, a), emb_sim)}
        for a in alphas
    )


def bootstrap_interval(
    ont_sim: np.ndarray,
    emb_sim: np.ndarray,
    n_reps: int,
    ci_level: float = 0.95,
    seed: int | None = 0,
) -> dict:
    """
    Percentile intervals of the correlations from resampling types.

    Each replicate uses the distinct types of the sample, because repeated types would
    add pairs of identical vectors.
    """
    rng = np.random.default_rng(seed)
    n = len(ont_sim)
    reps = []
    for _ in range(n_reps):
        idx = np.unique(rng.choice(n, size=n))
        if len(idx) >= MIN_TYPES:
            sub = np.ix_(idx, idx)
            reps.append(_correlations(ont_sim[sub], emb_sim[sub]))
    tail = 100 * (1 - ci_level) / 2
    out = {"bootstrap_reps": len(reps)}
    for name, values in pd.DataFrame(reps).items():
        out[f"{name}_ci_lower"], out[f"{name}_ci_upper"] = (
            float(v) for v in np.nanpercentile(values, [tail, 100 - tail])
        )
    return out


def permutation_test(
    ont_sim: np.ndarray,
    emb_sim: np.ndarray,
    observed: dict,
    k: int,
    n_reps: int,
    seed: int | None = 0,
) -> dict:
    """
    One-sided permutation test that shuffles which vector belongs to which type.

    p-values are (count of null >= observed + 1) / (n_reps + 1).
    """
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(n_reps):
        perm = rng.permutation(len(emb_sim))
        shuffled = emb_sim[np.ix_(perm, perm)]
        hood = neighborhood_metrics(ont_sim, shuffled, k)[["jaccard", "recall", "map"]]
        nulls.append(_correlations(ont_sim, shuffled) | hood.mean().to_dict())
    out = {"perm_reps": n_reps}
    for name, null in pd.DataFrame(nulls).items():
        std = null.std(ddof=0)
        out[f"{name}_null_mean"] = float(null.mean())
        out[f"{name}_null_std"] = float(std)
        out[f"{name}_z"] = float((observed[name] - null.mean()) / (std + 1e-10))
        out[f"{name}_p"] = float(((null >= observed[name]).sum() + 1) / (n_reps + 1))
    return out


def neighborhood_size(n_types: int, k: int) -> int:
    """Neighbors to compare: k, at most a third of the other types but at least 3."""
    return min(k, max(3, (n_types - 1) // 3))


def neighborhood_metrics(
    ont_sim: np.ndarray, emb_sim: np.ndarray, k: int
) -> pd.DataFrame:
    """Per-type Jaccard, recall and average precision of the k nearest neighbors."""
    ont_rank, emb_rank = _neighbor_order(ont_sim), _neighbor_order(emb_sim)
    rows = []
    for ont, emb in zip(ont_rank[:, :k], emb_rank[:, :k]):
        hit = np.isin(emb, ont)
        precision = np.cumsum(hit) / np.arange(1, k + 1)
        rows.append(
            {
                "jaccard": hit.sum() / (2 * k - hit.sum()),
                "recall": hit.sum() / k,
                "map": (precision * hit).sum() / k,
            }
        )
    return pd.DataFrame(rows)


def load_classifier_weights(
    checkpoint_path: str | Path,
    label_column: str = "cell_type_ontology_term_id",
    weight_key: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Classifier weight vectors of a Lightning checkpoint, and the label of each.

    The checkpoint is unpickled, so only load files from a trusted source.

    Args:
    ----
        checkpoint_path: `.ckpt` file.
        label_column: label column of the classifier.
        weight_key: key of the weight matrix in the state dict. None searches for it.

    Returns:
    -------
        Array of shape (n_labels, hidden_size) and the labels in row order.
    """
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    if weight_key is None:
        suffixes = (
            f"label_decoders.{label_column}.decoder.weight",
            f"classifiers.{label_column}.out_proj.weight",
            f"classifiers.{label_column}.weight",
        )
        found = [k for s in suffixes for k in state_dict if k.endswith(s)]
        if not found:
            raise ValueError(
                f"No classifier weights for '{label_column}'; pass weight_key."
            )
        weight_key = found[0]
    weights = state_dict[weight_key].float().numpy()
    label_dict = _checkpoint_label_dict(checkpoint, label_column)
    if sorted(label_dict.values()) != list(range(len(weights))):
        raise ValueError(
            f"{len(weights)} weight vectors do not match the "
            f"{len(label_dict)} labels of '{label_column}'."
        )
    return weights, sorted(label_dict, key=label_dict.get)


def _checkpoint_label_dict(checkpoint: dict, label_column: str) -> dict[str, int]:
    hparams = checkpoint.get("hyper_parameters", {})
    model_config = hparams.get("model_config")
    if isinstance(model_config, dict):
        columns = model_config.get("label_columns")
    else:
        columns = getattr(model_config, "label_columns", None)
    for column in columns or []:
        if not isinstance(column, dict):
            column = vars(column)
        if column.get("label_column_name") == label_column:
            label_dict = column.get("label_dict") or column.get("_label_dict")
            if label_dict:
                return label_dict
    label_dict = (hparams.get("label_dict") or {}).get(label_column)
    if not label_dict:
        raise ValueError(
            f"The checkpoint has no label dictionary for '{label_column}'."
        )
    return label_dict


def _cosine(vectors: np.ndarray) -> np.ndarray:
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return unit @ unit.T


def _neighbor_order(sim: np.ndarray) -> np.ndarray:
    """Other types of each row, most similar first; ties keep the input order."""
    # Symmetric ontology positions tie up to rounding error.
    order = np.argsort(-sim.round(10), axis=1, kind="stable")
    n = len(sim)
    return order[order != np.arange(n)[:, None]].reshape(n, n - 1)


def _correlations(ont_sim: np.ndarray, emb_sim: np.ndarray) -> dict:
    upper = np.triu_indices(len(ont_sim), k=1)
    return {
        "spearman": float(spearmanr(ont_sim[upper], emb_sim[upper])[0]),
        "pearson": float(pearsonr(ont_sim[upper], emb_sim[upper])[0]),
    }
