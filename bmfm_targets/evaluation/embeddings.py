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
        clusters = sc.tl.louvain(
            adata_dim_reduced, copy=True, flavor="igraph", **kwargs
        )
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


def concat_embeddings(
    embeddings_file1, embeddings_file2, output_dir, output_file_name="embeddings.csv"
):
    """
    Concatenate two embedding files horizontally and save the result.

    This function loads two embedding CSV files, aligns them by their indices (handling mismatches
    by using only common samples), concatenates them column-wise, and saves the combined embeddings
    to a new file.

    Parameters
    ----------
    embeddings_file1 : pathlib.Path
        Path to the first embeddings CSV file (no header, first column is index).
    embeddings_file2 : pathlib.Path
        Path to the second embeddings CSV file (no header, first column is index).
    output_dir : pathlib.Path
        Directory where the concatenated embeddings will be saved.
    output_file_name : str, optional
        Name of the output file (default is "embeddings.csv").

    Returns
    -------
    None
        The function saves the concatenated embeddings to disk and prints progress information.

    Raises
    ------
    FileNotFoundError
        If either of the input embedding files does not exist.

    Notes
    -----
    - If the indices of the two embedding files don't match, the function will use only the
      common samples and print a warning.
    - The output file is saved without a header and with the index column.
    - Progress information is printed to stdout during execution.
    """
    # Check if both files exist
    if not embeddings_file1.exists():
        raise FileNotFoundError(f"Embeddings not found at {embeddings_file1}")
    if not embeddings_file2.exists():
        raise FileNotFoundError(f"Embeddings not found at {embeddings_file2}")

    # Load embeddings
    embeddings1 = pd.read_csv(embeddings_file1, header=None, index_col=0)
    logger.info(f"Embeddings 1 shape: {embeddings1.shape}")

    logger.info(f"Loading embeddings from {embeddings_file2}")
    embeddings2 = pd.read_csv(embeddings_file2, header=None, index_col=0)
    logger.info(f"Embeddings 2 shape: {embeddings2.shape}")

    # Verify indices match
    if not embeddings1.index.equals(embeddings2.index):
        logger.warning("Index mismatch between embeddings")
        # Align on common indices
        common_idx = embeddings1.index.intersection(embeddings2.index)
        logger.info(f"Using {len(common_idx)} common samples")
        embeddings1 = embeddings1.loc[common_idx]
        embeddings2 = embeddings2.loc[common_idx]

    # Concat embeddings horizontally (concatenate columns)
    concatenated_embeddings = pd.concat([embeddings1, embeddings2], axis=1)
    logger.info(f"Combined shape: {concatenated_embeddings.shape}")

    # Save combined embeddings
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / output_file_name
    concatenated_embeddings.to_csv(output_path, header=False)


def calculate_batch_integration_metrics(
    adata_emb: sc.AnnData,
    batch_column_name: str | None,
    target_column_name: str,
    embed_key: str = "concatenated_embeddings",
) -> pd.DataFrame:
    """
    Calculate batch integration metrics (avg_bio and avg_batch).

    This function is extracted from BatchIntegrationCallback._generate_metrics_table()
    to work independently without PyTorch Lightning. It computes scIB metrics for
    evaluating batch integration quality.

    Parameters
    ----------
    adata_emb : sc.AnnData
        AnnData object with embeddings in obsm[embed_key] and metadata in obs
    batch_column_name : str or None
        Column name for batch information in adata_emb.obs. If None or single batch,
        only bio metrics will be calculated.
    target_column_name : str
        Column name for target labels (e.g., cell type) in adata_emb.obs
    embed_key : str, default="concatenated_embeddings"
        Key in adata_emb.obsm where embeddings are stored

    Returns
    -------
    pd.DataFrame
        DataFrame with metrics including:
        - NMI_cluster_by_{label}_(bio): Normalized Mutual Information
        - ARI_cluster_by_{label}_(bio): Adjusted Rand Index
        - ASW_by_{label}_(bio): Average Silhouette Width
        - Avg_bio: Average of bio metrics
        - graph_conn_by_{batch}_(batch): Graph connectivity (if multi-batch)
        - ASW_by_{batch}_(batch): Silhouette width for batch (if multi-batch)
        - Avg_batch: Average of batch metrics (if multi-batch)

    Notes
    -----
    This implementation follows the same logic as BatchIntegrationCallback._generate_metrics_table()
    from bmfm_targets/training/callbacks.py (lines 360-444).

    Examples
    --------
    >>> import scanpy as sc
    >>> adata = sc.read_h5ad("data.h5ad")
    >>> adata.obsm["concatenated_embeddings"] = embeddings_array
    >>> metrics = calculate_batch_integration_metrics(
    ...     adata,
    ...     batch_column_name="batch",
    ...     target_column_name="cell_type"
    ... )
    >>> print(f"Avg Bio: {metrics['Avg_bio'].iloc[0]}")
    """
    import scib.metrics as scm
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        silhouette_score,
    )

    batch_col, label_col = batch_column_name, target_column_name

    # Compute neighbors graph using the embeddings
    sc.pp.neighbors(adata_emb, use_rep=embed_key)

    # Check if we have batch information and multiple batches
    has_batch = batch_col and batch_col in adata_emb.obs
    single_batch = (not has_batch) or (
        adata_emb.obs[batch_col].nunique(dropna=False) == 1  # noqa: PD101
    )
    results = {}

    if single_batch:
        # Single batch case: use sklearn metrics directly
        cluster_key = "__tmp_scib_cluster"
        sc.tl.leiden(adata_emb, key_added=cluster_key, random_state=0)
        clusters = adata_emb.obs[cluster_key].to_numpy()
        labels = adata_emb.obs[label_col].to_numpy()
        X = adata_emb.obsm[embed_key]

        # Filter out NaN labels
        mask = pd.notna(labels)
        clusters, labels, X = clusters[mask], labels[mask], X[mask]

        results["NMI_cluster/label"] = normalized_mutual_info_score(labels, clusters)
        results["ARI_cluster/label"] = adjusted_rand_score(labels, clusters)
        vc = pd.Series(labels).value_counts()
        results["ASW_label"] = (
            float(silhouette_score(X, labels))
            if (vc.size > 1 and vc.min() >= 2)
            else np.nan
        )
    else:
        # Multiple batches: use scib.metrics for comprehensive evaluation
        results_df = scm.metrics(
            adata_emb,
            adata_int=adata_emb,
            batch_key=str(batch_col),
            label_key=str(label_col),
            embed=embed_key,
            isolated_labels_asw_=False,
            silhouette_=True,
            hvg_score_=False,
            pcr_=False,
            isolated_labels_f1_=False,
            trajectory_=False,
            nmi_=True,
            ari_=True,
            cell_cycle_=False,
            kBET_=False,
            ilisi_=False,
            clisi_=False,
            graph_conn_=True,
        )
        results = results_df.iloc[:, 0].to_dict()

    # Calculate average bio score
    bio_keys = ["NMI_cluster/label", "ARI_cluster/label", "ASW_label"]
    results["avg_bio"] = np.nanmean([results.get(k, np.nan) for k in bio_keys])

    # Calculate average batch score (only for multiple batches)
    if not single_batch:
        batch_keys = ["graph_conn", "ASW_label/batch"]
        results["avg_batch"] = np.nanmean([results.get(k, np.nan) for k in batch_keys])

    # Rename metrics for clarity
    rename = {
        "NMI_cluster/label": f"NMI_cluster_by_{label_col}_(bio)",
        "ARI_cluster/label": f"ARI_cluster_by_{label_col}_(bio)",
        "ASW_label": f"ASW_by_{label_col}_(bio)",
        "graph_conn": f"graph_conn_by_{batch_col}_(batch)",
        "ASW_label/batch": f"ASW_by_{batch_col}_(batch)",
        "avg_bio": "Avg_bio",
        "avg_batch": "Avg_batch",
    }
    output = {}
    for old, new in rename.items():
        v = results.get(old, np.nan)
        if not (isinstance(v, float) and np.isnan(v)):
            output[new] = np.round(v, 3)

    return pd.DataFrame([output])
