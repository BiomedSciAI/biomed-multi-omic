# Sci-Plex 3 mechanism-of-action (MoA) task

Code for the drug mechanism-of-action evaluation in the BMFM-RNA paper (Figure 6): frozen-embedding probing of MoA on drug-treated A549 cells from Sci-Plex 3, with drugs held out across cross-validation folds. The analysis scripts are standalone and do not import `bmfm_targets`. The embeddings are computed with `bmfm_targets` and with [helical](https://github.com/helicalAI/helical).

## 1. Data

Sci-Plex 3 (Srivatsan et al. 2020, *Science*, [doi:10.1126/science.aax6234](https://doi.org/10.1126/science.aax6234)), A549 dataset as curated by CZ CELLxGENE Discover ([collection](https://cellxgene.cziscience.com/collections/00109df5-7810-4542-8db5-2288c46e0424)):

```bash
curl -O https://datasets.cellxgene.cziscience.com/8dc61aa8-8d55-4180-854e-9dde86aeaad2.h5ad
python build_input.py --input 8dc61aa8-8d55-4180-854e-9dde86aeaad2.h5ad --out a549_hvg5000.h5ad
```

The file has 143,015 cells with raw counts in `X`. Its `obs` carries the Sci-Plex annotations used here: `target` (MoA class), `product_name` (drug), `dose` (nM) and `viability`. `build_input.py` keeps protein-coding genes, then cells with at least 300 expressed genes, then the 5,000 seurat_v3 highly variable genes (5,002 with duplicate gene symbols). The result is 141,814 cells × 5,002 genes of raw counts. This reproduces the paper's input exactly (same cells, genes and counts).

## 2. Embeddings

Every model gets the raw counts from step 1, batch size 20, and writes one `obsm` key. These calls are transcribed from the logs of the paper's run. The BMFM checkpoints used there have weights identical to the HuggingFace releases named below. The embeddings themselves have not been recomputed from this README.

```python
import bmfm_targets
# WCED MULTITASK: the reconstruction [CLS] token at position 0
bmfm_targets.inference(adata, checkpoint="ibm-research/biomed.rna.llama.47m.wced.multitask.v1",
                       embedding_key="X_bmfm_wced_cls_0", pooling_method=0,
                       log_normalize_transform=True, batch_size=20)
# MLM MULTITASK: the checkpoint's own pooling (first_token)
bmfm_targets.inference(adata, checkpoint="ibm-research/biomed.rna.llama.32m.mlm.multitask.v1",
                       embedding_key="X_bmfm_mlm", log_normalize_transform=True, batch_size=20)
```

`log_normalize_transform` computes log1p(10,000 × count / (total + 1)), with the total taken over the genes passed to the model (at most `max_length`). The run left the other arguments at their defaults (`limit_genes="protein_coding"`, `max_length=1024` tokens per cell).

External models, helical 1.5.3, default settings except as listed:

| `obsm` key | helical config |
|---|---|
| `X_scGPT` | `scGPTConfig(emb_mode="cls")` |
| `X_38M_geneformer` | `GeneformerConfig(model_name="gf-12L-38M-i4096")` |
| `X_316M_geneformer` | `GeneformerConfig(model_name="gf-18L-316M-i4096")` |
| `X_transcriptformer` | `TranscriptFormerConfig(model_name="tf_sapiens")` |

## 3. Analysis

```bash
pip install -r requirements.txt     # Python 3.11

python prepare_data.py --input a549_hvg5000_with_embeddings.h5ad --out data
python run_auc.py                    # results/auc_per_repeat.csv, roc_curves.npz
python run_avg_bio.py                # results/avg_bio.csv
python run_umap.py                   # results/umap_coords.npz
python make_fig6.py                  # results/Figure6.pdf, fig6_summary.csv
```

Each script takes `--data` and `--results` (defaults `data/` and `results/`) and `--keys` to restrict the embeddings. `make_fig6.py` redraws the paper figure for the six keys in its `MODELS` dict. Its axis limits and label placements are set for those results.

- **`prepare_data.py`** keeps drug-treated cells (no vehicle) at 10,000 nM with viability < 0.85. It then removes the "Others" MoA class and every MoA class with a single drug, because holding out drugs needs at least two per class. With the paper's data: 31,229 → 8,330 → 4,667 cells, 37 drugs, 9 MoA classes.
- **`run_auc.py`** fits a logistic regression (lbfgs, C=1, balanced class weights, features unscaled) on each embedding, with stratified drug-group cross-validation: each fold holds out at least one complete drug from every MoA class. Five folds are requested and reduced automatically to two, the most the class with the fewest drugs allows. There are five repeats with seeds 42–46, and they are paired across embeddings. The script reports micro-averaged ROC AUC on the held-out predictions, pooled over repeats and per repeat, plus macro one-vs-rest AUC. The paper gives the pooled micro AUC and the standard deviation over repeats.
- **`run_avg_bio.py`** computes Avg Bio, the mean of NMI, ARI and ASW against the MoA labels, from `scib.metrics`. NMI and ARI are on the Leiden clustering at the NMI-optimal resolution. It is computed on all cells, and on `--n-subsamples` random 90% subsamples (the same subsamples for every embedding) for the error bars.
- **`run_umap.py`** computes a UMAP of each embedding for panel E.
- **`moa_lib.py`** holds the filtering, cross-validation and metric code shared by the scripts above.
- **`moa_predict.py`** is the original end-to-end script (`python moa_predict.py --input data.h5ad --output results/ --embedding_keys ...`). It writes more metrics and diagnostic plots, and uses the same `moa_lib` code.

## Package versions

`requirements.txt` lists the environment that produced Figure 6. The AUCs do not depend on these versions (they agree to 1e-4 with scikit-learn 1.9). NMI and ARI do depend on them: Leiden's NMI-optimal resolution changes with the scanpy/leidenalg version and with subsampling, and this moves ARI and so Avg Bio.
