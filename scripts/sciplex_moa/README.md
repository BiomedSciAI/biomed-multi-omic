# Sci-Plex 3 mechanism-of-action (MoA) task

Code for the drug mechanism-of-action evaluation in the BMFM-RNA paper (Figure 6): frozen-embedding probing of MoA on drug-treated A549 cells from Sci-Plex 3, with drugs held out across cross-validation folds. These are standalone scripts and do not import `bmfm_targets`.

## Input

One h5ad with the Sci-Plex 3 cells, with:

- `obs`: `target` (MoA class), `product_name` (drug), `dose` (nM), `viability`.
- `obsm`: one cell embedding per model, computed beforehand. In the paper the BMFM embeddings come from `bmfm_targets` inference, and the external models (Geneformer 38M and 316M, scGPT, TranscriptFormer) from their own packages. The WCED MULTITASK embedding is the reconstruction [CLS] token (position 0, key `X_bmfm_wced_cls_0`), not the cell-type [CLS] token.

## Steps

```bash
pip install -r requirements.txt     # Python 3.11

python prepare_data.py --input sciplex3_a549.h5ad --out data
python run_auc.py                    # results/auc_per_repeat.csv, roc_curves.npz
python run_avg_bio.py --n-boot 10    # results/avg_bio.csv
python run_umap.py                   # results/umap_coords.npz
python make_fig6.py                  # results/Figure6.pdf, fig6_summary.csv
```

Each script takes `--data` and `--results` (defaults `data/` and `results/`) and `--keys` to restrict the embeddings. `make_fig6.py` expects the six paper keys listed in its `MODELS` dict.

## What each step does

- **`prepare_data.py`**: keeps drug-treated cells (no vehicle) at 10,000 nM with viability < 0.85, removes the "Others" MoA class and every MoA class with a single drug, because holding out drugs needs at least two per class. With the paper's data: 31,229 → 8,330 → 4,667 cells, 37 drugs, 9 MoA classes.
- **`run_auc.py`**: logistic regression (balanced class weights) on each embedding, with stratified drug-group cross-validation: each fold holds out at least one complete drug from every MoA class. Five folds are requested and reduced automatically to two, which is the most the class with the fewest drugs allows. Five repeats, seed 42. Reports micro-averaged ROC AUC on the held-out predictions, pooled over repeats and per repeat; the paper gives the pooled value and the standard deviation over repeats.
- **`run_avg_bio.py`**: Avg Bio = mean of NMI, ARI and ASW against the MoA labels, from `scib.metrics` (NMI and ARI on the Leiden clustering at the NMI-optimal resolution). Computed on all cells, and on `--n-boot` random 90% subsamples (the same subsamples for every embedding) for the error bars.
- **`run_umap.py`**: UMAP of each embedding for panel E.
- **`moa_predict.py`**: the original end-to-end script (`python moa_predict.py --input data.h5ad --output results/`). The scripts above import its filtering and cross-validation functions.

## Package versions

NMI and ARI depend on the Leiden implementation: with newer scanpy and leidenalg, the selected resolution and so ARI change. Use the versions in `requirements.txt` to reproduce the paper's Avg Bio values. The AUCs are not sensitive to versions.
