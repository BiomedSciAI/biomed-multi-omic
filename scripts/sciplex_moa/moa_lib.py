"""
Shared code for the Sci-Plex 3 MoA scripts: cell filtering, drug-holdout CV, metrics.

Importing this module has no side effects (no warning filters, no matplotlib state).
"""
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

REQUIRED_OBS = ["target", "product_name", "dose", "viability"]


def load_and_prepare_data(
    adata_path: str,
    embedding_keys: list[str],
    cell_type: str | None = None,
    skip_vehicle_filter: bool = False,
    dose: list[float] | None = None,
    viability_threshold: float = 0.85,
    skip_viability_filter: bool = False,
    skip_others_filter: bool = False,
    min_drugs_per_moa: int = 2,
) -> tuple[ad.AnnData, dict[str, np.ndarray]]:
    """
    Load the AnnData, filter to the MoA task set and extract embeddings.

    Filters, in order: optional cell type, drop vehicle, keep `dose`, keep
    viability < `viability_threshold`, drop the "Others" MoA class, drop MoA
    classes with fewer than `min_drugs_per_moa` drugs.

    Args:
    ----
        adata_path: path to the h5ad.
        embedding_keys: obsm keys to return.
        cell_type: if set, keep only cells with obs["cell_type"] == cell_type.
        skip_vehicle_filter: keep vehicle-control cells.
        dose: doses (nM) to keep; default [10000].
        viability_threshold: keep cells with viability below this.
        skip_viability_filter: do not filter on viability.
        skip_others_filter: keep the "Others" MoA class.
        min_drugs_per_moa: drop MoA classes with fewer drugs than this.

    Returns:
    -------
        The filtered AnnData and a dict of the requested embeddings.

    """
    print(f"Loading data from {adata_path}...")
    adata = ad.read_h5ad(adata_path)

    required_cols = REQUIRED_OBS if not skip_viability_filter else REQUIRED_OBS[:3]
    for col in required_cols:
        if col not in adata.obs.columns:
            raise ValueError(f"Required column '{col}' not found in adata.obs")

    print(f"Loaded {adata.n_obs} cells")
    print(f"MoA classes (before filtering): {adata.obs['target'].nunique()}")
    print(f"Unique products (before filtering): {adata.obs['product_name'].nunique()}")

    if dose is None:
        dose = [10000]

    if cell_type is not None:
        if "cell_type" not in adata.obs.columns:
            raise ValueError("Column 'cell_type' not found in adata.obs")
        print(f"\nFiltering to cell type: {cell_type}...")
        adata = adata[adata.obs["cell_type"] == cell_type].copy()
        print(f"After cell type filtering: {adata.n_obs} cells")

    if not skip_vehicle_filter:
        print("\nFiltering out vehicle control...")
        adata = adata[adata.obs["target"].str.lower() != "vehicle"].copy()
        print(f"After vehicle filtering: {adata.n_obs} cells")

    print(f"\nFiltering to dose(s): {dose}...")
    adata = adata[adata.obs["dose"].isin(dose)].copy()
    _print_counts(adata, "dose filtering")

    if not skip_viability_filter:
        print(f"\nFiltering to cells with viability < {viability_threshold}...")
        adata = adata[adata.obs["viability"] < viability_threshold].copy()
        _print_counts(adata, "viability filtering")

    if not skip_others_filter:
        print("\nRemoving 'Others' MoA class...")
        adata = adata[adata.obs["target"].str.lower() != "others"].copy()
        _print_counts(adata, "removing 'Others'")

    print(f"\nRemoving MoA classes with fewer than {min_drugs_per_moa} drug(s)...")
    moa_drug_counts = adata.obs.groupby("target")["product_name"].nunique()
    moas_to_keep = moa_drug_counts[moa_drug_counts >= min_drugs_per_moa].index.tolist()
    n_moas_removed = len(moa_drug_counts) - len(moas_to_keep)
    if n_moas_removed > 0:
        print(
            f"  Removing {n_moas_removed} MoA classes with fewer than {min_drugs_per_moa} drug(s)"
        )
        adata = adata[adata.obs["target"].isin(moas_to_keep)].copy()
        _print_counts(adata, "MoA filtering")
    else:
        print(f"  All MoA classes have at least {min_drugs_per_moa} drug(s)")

    embeddings = {}
    for key in embedding_keys:
        if key not in adata.obsm:
            raise ValueError(f"Embedding '{key}' not found in adata.obsm")
        embeddings[key] = adata.obsm[key]
        print(f"  - {key}: shape {embeddings[key].shape}")

    return adata, embeddings


def _print_counts(adata: ad.AnnData, step: str) -> None:
    print(f"After {step}: {adata.n_obs} cells")
    print(f"MoA classes: {adata.obs['target'].nunique()}")
    print(f"Unique products: {adata.obs['product_name'].nunique()}")


def create_stratified_drug_folds(
    products: np.ndarray,
    moa_labels: np.ndarray,
    n_folds: int = 5,
    random_seed: int = 42,
) -> list[list[str]]:
    """
    Split drugs into folds so that every fold holds out at least one drug per MoA class.

    If the smallest MoA class has fewer drugs than `n_folds`, the number of folds is
    reduced to that size (with the paper's data, 5 becomes 2).

    Args:
    ----
        products: drug name per cell.
        moa_labels: MoA label per cell.
        n_folds: requested number of folds.
        random_seed: seed for shuffling drugs within each MoA class.

    Returns:
    -------
        One list of held-out drug names per fold.

    Raises:
    ------
        ValueError: if a drug maps to more than one MoA, or a MoA class has one drug.

    """
    rng = np.random.RandomState(random_seed)

    moa_per_drug = pd.Series(moa_labels).groupby(pd.Series(products)).nunique()
    if (moa_per_drug > 1).any():
        bad = moa_per_drug[moa_per_drug > 1].index.tolist()
        raise ValueError(f"Drugs with more than one MoA label: {bad[:5]}")

    moa_to_drugs: dict = {}
    for product in np.unique(products):
        moa = moa_labels[products == product][0]
        moa_to_drugs.setdefault(moa, []).append(product)

    for moa in moa_to_drugs:
        rng.shuffle(moa_to_drugs[moa])

    min_drugs_per_moa = min(len(drugs) for drugs in moa_to_drugs.values())
    if min_drugs_per_moa < n_folds:
        print(
            f"  Warning: Some MoA classes have fewer than {n_folds} drugs (min={min_drugs_per_moa})"
        )
        if min_drugs_per_moa < 2:
            raise ValueError(
                f"Cannot perform cross-validation: some MoA classes have only {min_drugs_per_moa} drug(s). "
                f"Need at least 2 drugs per MoA class to create train/test splits."
            )
        print(
            f"  Adjusting to {min_drugs_per_moa} folds to ensure each fold has all MoA classes"
        )
        n_folds = min_drugs_per_moa

    # one drug per MoA per fold first, then the remaining drugs round-robin
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for drugs in moa_to_drugs.values():
        for fold_idx in range(min(n_folds, len(drugs))):
            folds[fold_idx].append(drugs[fold_idx])
    for drugs in moa_to_drugs.values():
        for i, drug in enumerate(drugs[n_folds:]):
            folds[i % n_folds].append(drug)

    return folds


def stratified_drug_group_cv(
    X: np.ndarray,
    y: np.ndarray,
    products: np.ndarray,
    n_folds: int = 5,
    n_repeats: int = 5,
    random_seed: int = 42,
    max_iter: int = 1000,
) -> dict[str, list]:
    """
    Logistic-regression probing with drug-holdout folds, repeated with different seeds.

    Repeat r builds folds with seed `random_seed + r`, so repeats are paired across
    embeddings. Features are used as given (no scaling); the classifier is sklearn's
    LogisticRegression with lbfgs, C=1 and balanced class weights.

    Args:
    ----
        X: embedding, one row per cell.
        y: MoA labels encoded as 0..K-1.
        products: drug name per cell.
        n_folds: requested folds (see create_stratified_drug_folds).
        n_repeats: number of CV repeats.
        random_seed: base seed.
        max_iter: logistic-regression iteration limit.

    Returns:
    -------
        Dict with per-cell y_true, y_pred, test_products, fold, repeat, and y_prob as
        a list with one (n_test, K) array per fold.

    """
    n_classes = len(np.unique(y))
    results: dict[str, list] = {
        "y_true": [],
        "y_pred": [],
        "y_prob": [],
        "test_products": [],
        "fold": [],
        "repeat": [],
    }

    fold_idx = 0
    for repeat in range(n_repeats):
        folds = create_stratified_drug_folds(
            products, y, n_folds=n_folds, random_seed=random_seed + repeat
        )
        for fold_num, test_drugs in enumerate(folds):
            test_mask = np.isin(products, test_drugs)
            train_mask = ~test_mask
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            if len(y_test) < 10 or len(np.unique(y_test)) < 2:
                print(
                    f"  Warning: Skipping fold {fold_num} in repeat {repeat}: insufficient test samples"
                )
                continue

            print(
                f"  Repeat {repeat}, Fold {fold_num}: {len(test_drugs)} drugs, "
                f"{len(np.unique(y_test))} MoA classes in test set, {len(y_test)} cells"
            )

            clf = LogisticRegression(
                max_iter=max_iter,
                random_state=random_seed + repeat,
                class_weight="balanced",
                solver="lbfgs",
                n_jobs=-1,
            )
            clf.fit(X_train, y_train)
            # predict_proba columns follow clf.classes_; downstream code indexes them as 0..K-1
            if not np.array_equal(clf.classes_, np.arange(n_classes)):
                raise RuntimeError(
                    f"Training fold lacks MoA classes {set(range(n_classes)) - set(clf.classes_)}"
                )

            results["y_true"].extend(y_test.tolist())
            results["y_pred"].extend(clf.predict(X_test).tolist())
            results["y_prob"].append(clf.predict_proba(X_test))
            results["test_products"].extend(products[test_mask].tolist())
            results["fold"].extend([fold_idx] * len(y_test))
            results["repeat"].extend([repeat] * len(y_test))
            fold_idx += 1

    return results


def micro_auc(
    y_true: np.ndarray, y_prob: np.ndarray, n_classes: int
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Micro-averaged one-vs-rest ROC AUC: all (cell, class) pairs scored as one binary problem.

    Returns
    -------
        AUC, false-positive rates and true-positive rates of the pooled ROC curve.

    """
    y_bin = label_binarize(y_true, classes=range(n_classes))
    fpr, tpr, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
    return auc(fpr, tpr), fpr, tpr


def scib_bio_metrics(
    obs: pd.DataFrame, X: np.ndarray, label_key: str = "target"
) -> dict[str, float]:
    """
    NMI, ARI and ASW_label of an embedding against `label_key`, from scib.metrics.metrics.

    One dummy batch, so only the bio-conservation metrics are computed. NMI and ARI are
    on the Leiden clustering at the NMI-optimal resolution (scib's default search).

    Returns
    -------
        Dict with NMI, ARI, ASW and avg_bio (their mean, ignoring NaN).

    """
    import scib.metrics as scm  # heavy import, only needed here

    a = ad.AnnData(X=np.zeros((len(obs), 1), dtype=np.float32), obs=obs.copy())
    a.obs[label_key] = a.obs[label_key].astype("category")
    a.obs["batch"] = "batch1"
    a.obsm["X_emb"] = X
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = (
            scm.metrics(
                a,
                adata_int=a,
                batch_key="batch",
                label_key=label_key,
                embed="X_emb",
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
                graph_conn_=False,
            )
            .iloc[:, 0]
            .to_dict()
        )
    nmi = r.get("NMI_cluster/label", np.nan)
    ari = r.get("ARI_cluster/label", np.nan)
    asw = r.get("ASW_label", np.nan)
    return {"NMI": nmi, "ARI": ari, "ASW": asw, "avg_bio": np.nanmean([nmi, ari, asw])}
