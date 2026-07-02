"""
Extract contrastive CLS embeddings from a scConcept checkpoint, then compare
KNN accuracy to the PCA-50 baseline on PBMC3k.

Usage:
  python run/extract_embeddings.py --ckpt /tmp/scconcept_pretrain/last.ckpt
"""
import argparse
import os
import sys

import anndata as ad
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def knn_accuracy(X, y, k=15, n_splits=5, seed=42):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X, y_enc, cv=cv, scoring="accuracy")
    return scores.mean(), scores.std()


def pca_baseline(adata, n_components=50):
    import scipy.sparse as sp

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    X = np.log1p(X / row_sums * 1e4)
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X)


def extract_cls_embeddings(ckpt_path, adata, max_length=64):
    """Extract raw CLS hidden state (no contrastive projection) for comparison."""
    import scipy.sparse as sp

    from bmfm_targets.models import get_model_from_config
    from bmfm_targets.tokenization import load_tokenizer
    from bmfm_targets.tokenization.multifield_collator import MultiFieldCollator
    from bmfm_targets.tokenization.multifield_instance import MultiFieldInstance

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_config = ckpt["hyper_parameters"]["model_config"]
    model = get_model_from_config(model_config)
    state_dict = {
        k[len("model.") :]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    tokenizer = load_tokenizer("all_genes")
    fields = model_config.fields
    collator = MultiFieldCollator(
        tokenizer=tokenizer, fields=fields, label_columns=[], max_length=max_length
    )

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    gene_names = list(adata.var_names)

    all_cls = []
    batch_size = 64
    for start in range(0, len(adata), batch_size):
        end = min(start + batch_size, len(adata))
        instances = []
        for i in range(end - start):
            row = X[start + i]
            nz = np.nonzero(row)[0]
            inst = MultiFieldInstance(
                data={
                    "genes": [gene_names[j] for j in nz],
                    "expressions": row[nz].tolist(),
                }
            )
            instances.append(inst)
        batch = collator(instances)
        with torch.no_grad():
            out = model(
                input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask")
            )
            # CLS = first token of sequence_output
            cls = out.embeddings  # pooler output (CLS)
            all_cls.append(F.normalize(cls, dim=-1).cpu().numpy())

    return np.vstack(all_cls)


def extract_contrastive_embeddings(ckpt_path, adata):
    """Run single-view forward pass and extract CLS contrastive projection."""
    from bmfm_targets.models import get_model_from_config

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt["hyper_parameters"]
    model_config = hparams["model_config"]

    # Build model
    model = get_model_from_config(model_config)

    # Load weights (filter to matching keys only)
    state_dict = {
        k[len("model.") :]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Load tokenizer — use the named identifier since the saved dir doesn't have the wrapper structure
    from bmfm_targets.tokenization import load_tokenizer

    tokenizer = load_tokenizer("all_genes")

    # Tokenize PBMC3k
    import scipy.sparse as sp

    # Build SingleCellDataset-style instances from adata
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    gene_names = list(adata.var_names)

    # Tokenize using the tokenizer's encode API
    from bmfm_targets.tokenization.multifield_instance import MultiFieldInstance

    all_z = []
    batch_size = 64
    fields = model_config.fields

    from bmfm_targets.tokenization.multifield_collator import MultiFieldCollator

    # match training config: max_length=64
    collator = MultiFieldCollator(
        tokenizer=tokenizer, fields=fields, label_columns=[], max_length=64
    )

    print(f"Encoding {len(adata)} cells...")
    for start in range(0, len(adata), batch_size):
        end = min(start + batch_size, len(adata))
        batch_X = X[start:end]

        instances = []
        for i in range(end - start):
            row = batch_X[i]
            nonzero = np.nonzero(row)[0]
            genes_i = [gene_names[j] for j in nonzero]
            exprs_i = row[nonzero].tolist()
            inst = MultiFieldInstance(data={"genes": genes_i, "expressions": exprs_i})
            instances.append(inst)

        batch = collator(instances)
        with torch.no_grad():
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask")
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            # Extract contrastive projection: [B, D]
            z = out.logits["expressions_contrastive"]
            z = F.normalize(z, dim=-1)
            all_z.append(z.cpu().numpy())

        if (start // batch_size) % 5 == 0:
            print(f"  {end}/{len(adata)}", flush=True)

    return np.vstack(all_z)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to scConcept .ckpt file")
    parser.add_argument("--data", default="data/real/pbmc3k.h5ad")
    args = parser.parse_args()

    adata = ad.read_h5ad(args.data)
    y = adata.obs["celltype"].values
    print(f"Data: {adata.shape}, {len(np.unique(y))} cell types")

    print("\n--- Baseline: PCA-50 ---")
    X_pca = pca_baseline(adata)
    acc_pca, std_pca = knn_accuracy(X_pca, y)
    print(f"KNN-15 accuracy: {acc_pca:.4f} ± {std_pca:.4f}")

    print("\n--- scConcept (contrastive projection) ---")
    X_sc = extract_contrastive_embeddings(args.ckpt, adata)
    acc_sc, std_sc = knn_accuracy(X_sc, y)
    print(f"KNN-15 accuracy: {acc_sc:.4f} ± {std_sc:.4f}")

    delta = acc_sc - acc_pca
    print(f"\nDelta vs PCA: {delta:+.4f}  ({'better' if delta > 0 else 'worse'})")

    print("\n--- CLS (backbone, no projection head) ---")
    X_cls = extract_cls_embeddings(args.ckpt, adata)
    acc_cls, std_cls = knn_accuracy(X_cls, y)
    print(f"KNN-15 accuracy: {acc_cls:.4f} ± {std_cls:.4f}")

    print("\nSummary:")
    print(f"  PCA-50:         {acc_pca:.4f} ± {std_pca:.4f}")
    print(f"  CLS backbone:   {acc_cls:.4f} ± {std_cls:.4f}")
    print(f"  scConcept proj: {acc_sc:.4f} ± {std_sc:.4f}")
    print(f"  Delta (proj vs PCA): {delta:+.4f}")
    print(f"  Delta (CLS vs PCA):  {acc_cls - acc_pca:+.4f}")


if __name__ == "__main__":
    main()
