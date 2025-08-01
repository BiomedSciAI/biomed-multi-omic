import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn import metrics
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def compute_cohesion(distances, labels):
    """
    Compute the cohesion (sum of squared distances from data points to their centroid) of each cluster, and the average over all clusters.

    Parameters
    ----------
    distances: numpy.ndarray
        Embedding distances as a numpy array.
    labels: numpy.ndarray
        True labels as a numpy array.

    Returns
    -------
    numpy.ndarray
        An array of size n_clusters (or n_labels) containing the cohesion of each cluster.
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    cohesion = np.zeros(n_clusters)

    for i, label in enumerate(unique_labels):
        indices = np.where(labels == label)[0]
        distances_within_cluster = distances[indices, :]
        centroid = np.mean(distances_within_cluster, axis=0)
        squared_distances_to_centroid = np.sum(
            np.square(distances_within_cluster - centroid), axis=1
        )
        cohesion[i] = np.sum(squared_distances_to_centroid)

    average_cohesion = np.mean(cohesion)

    return [round(i, 2) for i in cohesion], round(average_cohesion, 2)


def compute_separation(distances, labels):
    """
    Compute the total separation between all clusters, defined by sum of squares between cluster to overall centroid  (weighted by the size of each cluster).

    Parameters
    ----------
    distances: numpy.ndarray
        Embedding distances as a numpy array.
    labels: numpy.ndarray
        True labels as a numpy array.

    Returns
    -------
    float
        The total separation between all clusters.
    """
    unique_labels = np.unique(labels)
    overall_mean = np.mean(distances, axis=0)

    separation = 0.0

    for label in unique_labels:
        indices = np.where(labels == label)[0]
        distances_within_cluster = distances[indices, :]
        centroid = np.mean(distances_within_cluster, axis=0)

        squared_difference = np.sum(np.square(centroid - overall_mean))

        cluster_size = len(indices)
        squared_difference *= cluster_size

        separation += squared_difference

    return round(separation, 2)


def calculate_metrics_embedding(distances, labels):
    """
    Computes evaluation metrics for embedding data considering ground-truth to be the supplied labels.

    Parameters
    ----------
    distances: numpy.ndarray
        Embedding distances as a numpy array.
    labels: numpy.ndarray
        True labels as a numpy array.

    The function computes the following clustering metrics:
    - Average silhouette width (ASW): Measures the quality of clustering.
    - Calinski-Harabasz Index: Measures cluster density.
    - Davies-Bouldin Index: Measures cluster separation.
    - Cohesion: Evaluates how closely the elements of the same cluster are to each other.
    - Separation: Quantifies the level of separation between clusters
    """
    silhouette = round(metrics.silhouette_score(distances, labels), 2)
    calinski_harabasz = round(metrics.calinski_harabasz_score(distances, labels), 2)
    davies_bouldin = round(metrics.davies_bouldin_score(distances, labels), 2)
    cohesion, average_cohesion = compute_cohesion(distances, labels)
    separation = compute_separation(distances, labels)

    metrics_dict = {
        "Average silhouette width (ASW):": silhouette,
        "Calinski-Harabasz Index:": calinski_harabasz,
        "Davies-Bouldin Index:": davies_bouldin,
        "Cohesion per cluster:": cohesion,
        "Average cohesion:": average_cohesion,
        "Clusters separation:": separation,
    }

    return metrics_dict


def generate_clusters(
    adata,
    n_components=2,
    label="CellType",
    clustering_method="kmeans",
    **kwargs,
):
    adata_copy = sc.tl.pca(adata, n_comps=n_components, copy=True)
    sc.pp.neighbors(adata_copy, use_rep="X_pca")
    n_classes_in_data = adata.obs[label].nunique()
    clusters = get_clusters(clustering_method, adata_copy, n_classes_in_data, **kwargs)

    return clusters


def get_clusters(
    clustering_method: str,
    adata_dim_reduced: sc.AnnData,
    n_classes_in_data: int,
    **kwargs,
):
    if clustering_method == "louvain":
        if not "resolution" in kwargs:
            kwargs["resolution"] = 0.6
        clusters = sc.tl.louvain(adata_dim_reduced, copy=True, **kwargs)
    elif clustering_method == "leiden":
        if not "resolution" in kwargs:
            kwargs["resolution"] = 0.6
        clusters = sc.tl.leiden(adata_dim_reduced, copy=True, **kwargs)
    elif clustering_method == "kmeans":
        if not "n_clusters" in kwargs:
            kwargs["n_clusters"] = n_classes_in_data
        kmeans = KMeans(**kwargs).fit(adata_dim_reduced.obsm["X_pca"])
        clusters = adata_dim_reduced.copy()
        clusters.obs["kmeans"] = pd.Categorical(kmeans.labels_)
    elif clustering_method == "dbscan":
        db = DBSCAN(**kwargs).fit(adata_dim_reduced.obsm["X_pca"])
        clusters = adata_dim_reduced.copy()
        clusters.obs["dbscan"] = pd.Categorical(db.labels_)
    elif clustering_method == "hierarchical":
        if not "n_clusters" in kwargs:
            kwargs["n_clusters"] = n_classes_in_data
        clusterer = AgglomerativeClustering(**kwargs)
        hierarchical = clusterer.fit(adata_dim_reduced.obsm["X_pca"])
        clusters = adata_dim_reduced.copy()
        clusters.obs["hierarchical"] = pd.Categorical(hierarchical.labels_)
    else:
        raise ValueError(f"clustering_method {clustering_method} is not supported")

    return clusters


def evaluate_clusters(clusters, clustering_method, label="CellType", normalize=False):
    eval_res_dict = {}
    eval_res_dict["method"] = clustering_method
    eval_res_dict["ARI"] = metrics.adjusted_rand_score(
        labels_true=clusters.obs[label], labels_pred=clusters.obs[clustering_method]
    )
    eval_res_dict["AMI"] = metrics.adjusted_mutual_info_score(
        labels_true=clusters.obs[label], labels_pred=clusters.obs[clustering_method]
    )
    eval_res_dict["ASW"] = metrics.silhouette_score(
        clusters.obsm["X_pca"], clusters.obs[clustering_method]
    )
    eval_res_dict["NMI"] = metrics.normalized_mutual_info_score(
        labels_true=clusters.obs[label], labels_pred=clusters.obs[clustering_method]
    )
    eval_res_dict["AvgBio"] = np.mean(
        [
            eval_res_dict["NMI"],
            eval_res_dict["ARI"],
            eval_res_dict["ASW"],
        ]
    )
    if normalize:
        # rewrite metrics with their normalized value
        scaler = MinMaxScaler()
        eval_res_dict["ASW"] = scaler.fit_transform([[eval_res_dict["ASW"]]])[0][0]
        eval_res_dict["AvgBio"] = scaler.fit_transform([[eval_res_dict["AvgBio"]]])[0][
            0
        ]

    return eval_res_dict


def load_predictions(working_dir: Path | str, to_adata: bool = True) -> dict:
    working_dir = Path(working_dir)
    try:
        results_files = {
            "embeddings": working_dir / "embeddings.csv",
            "logits": working_dir / "logits.csv",
            "predictions": working_dir / "predictions.csv",
            "probabilities": working_dir / "probabilities.csv",
        }
        results = {
            i: pd.read_csv(results_files[i], index_col=0)
            if i != "embeddings"
            else pd.read_csv(results_files[i], index_col=0, header=None)
            for i in results_files
        }
    except FileNotFoundError:
        raise FileNotFoundError("Check your working directory.")

    if to_adata:
        results["embeddings"].index.name = "cellnames"
        results["embeddings"].index = results["embeddings"].index.astype(str)
        results["embeddings"].columns = results["embeddings"].columns.astype(str)

        adata = sc.AnnData(X=results["embeddings"])
        adata.X = adata.X.astype("float64")

        results["predictions"].index.name = "cellnames"
        adata.obs = adata.obs.join(results["predictions"], how="left", lsuffix="_bmfm")
        results["adata"] = adata

    return results


def load_prediction_data_to_anndata(df_emb, df_labels, df_pred):
    if df_emb.shape[1] == 769:
        df_cellname = df_emb.loc[:, 0].to_frame()
        df_cellname.columns = ["cellname"]
        df_labels = pd.concat([df_labels, df_cellname], axis=1)
        df_emb = df_emb.loc[:, 1:]

    adata = sc.AnnData(X=df_emb)
    adata.obs = pd.concat([df_pred, df_labels], axis=1)

    adata.X = adata.X.astype("float64")
    return adata
