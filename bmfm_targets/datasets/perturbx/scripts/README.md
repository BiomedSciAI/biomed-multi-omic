# Preparation of Perturbx dataset

Our goal is to create a dataset that have

1. H1 cell line with train/dev split.
2. Full Replogle with only one index.
3. Alternative Replogle train/dev split.


Train/dev Replogle split will be used only for debugging purproses.
In training, we will use full Replogle and H1 train split for training, and H1 dev split for validation.

## H1 internal split

Assume that we have files with two splits: `train_data.h5ad` and `test_data.h5ad` (validation cell set). Target gene is in `target_gene` column, with control label `non-targeting`. Only one of the files has controls. Lets combine them and add "split" column.

```python
import numpy as np
import anndata as ad

train = ad.read_h5ad("train_data.h5ad")
test = ad.read_h5ad("test_data.h5ad")
merged = ad.concat({"train": train, "dev": test}, label = "split")
merged.obs.loc[merged.obs["target_gene"] == "non-targeting", "split"] = np.nan
merged.write("h1.h5ad")
```

To shuffle and create index use command

```bash
python transform.py --adata-path ${H1_DATASET}/h1.h5ad --rename-control --target-gene-column target_gene --control-label non-targeting
python shuffle_and_build_index.py --adata-path ${H1_DATASET}/transformed_h1.h5ad --target-gene-column target_gene --control-label Control --chunk-size 2000 --split-column split --splits train --splits dev
```

The script above shuffles the file. It sorts cells in multiple sections: control, train, dev to improve file reading locality. Then, each section is shuffled, `shuffled_h1.h5ad` file is produced. Next, the script creates litdata indices for `train` and `dev` splits. Each index also have a `control.bin` file with an index for all control cells (hdf format).

## Replogle

Assume that we already have `replogle.h5ad` with `scgpt_split` column. Target gene column is `gene`, control label is `Control` To create full dataset use

```bash
./shuffle_and_build_index.py --adata-path ${PERTURBX_DATASET}/transformed_replogle.h5ad --target-gene-column gene --control-label Control --chunk-size 2000 --splits full
mv ${PERTURBX_DATASET}/shuffled_transformed_replogle.h5ad ${PERTURBX_DATASET}/shuffled_transformed_replogle_full.h5ad
```

To create train/dev split use

```bash
./shuffle_and_build_index.py --adata-path ${PERTURBX_DATASET}/replogle.h5ad --target-gene-column gene --control-label Control --chunk-size 2000 --split-column scgpt_split --splits train --splits dev --splits test
```

We duplicated shuffled file because `shuffled_replogle.h5ad` and `shuffled_replogle_full.h5ad` have different spatial file reading locality.

### Replogle cell line specific files

Replogle has two cell lines that can be created in cell-line specific files, K562 (~320k cells, ~8.5k genes, ~2k perturbations) and RPE1 (~250k cells, ~9k genes, ~2.4k perturbations). The K562 cell line also has a second, very large 'GWPS' dataset (~2M cells, ~8k genes, ~10k perturbations).

To prepare perturbx dataset files from the raw files for the two different cell line files we can use:

```bash
export REPLOGLE_DATASET=</path/to/replogle/downloaded/h5ad/files>

# Transform
python transform.py --adata-path ${REPLOGLE_DATASET}/K562_essential_raw_singlecell_01.h5ad --reindex-var-names --rename-control --target-gene-column gene --control-label non-targeting
python transform.py --adata-path ${REPLOGLE_DATASET}/rpe1_raw_singlecell_01.h5ad --reindex-var-names --rename-control --target-gene-column gene --control-label non-targeting

# Shuffle and build index
python shuffle_and_build_index.py --adata-path ${REPLOGLE_DATASET}/transformed_K562_essential_raw_singlecell_01.h5ad --target-gene-column gene --control-label Control  --chunk-size 2000 --splits train
python shuffle_and_build_index.py --adata-path ${REPLOGLE_DATASET}/transformed_rpe1_raw_singlecell_01.h5ad --target-gene-column gene --control-label Control --chunk-size 2000 --splits train
```

Then similar for the GWPS dataset, but it takes much longer to process.

```bash
export REPLOGLE_DATASET=</path/to/replogle/downloaded/h5ad/files>

# Transform
python transform.py --adata-path ${REPLOGLE_DATASET}/K562_gwps_raw_singlecell_01.h5ad --reindex-var-names --rename-control --target-gene-column gene --control-label non-targeting

# Shuffle and build index
python shuffle_and_build_index.py --adata-path ${REPLOGLE_DATASET}/transformed_K562_gwps_raw_singlecell_01.h5ad --target-gene-column gene --control-label Control  --chunk-size 2000 --splits train
```

## Creating mini dataset for pytest

An example is based on Replogle dataset. To create a subset with minimal size, we drop unnecessary columns, all `var` columns, `uns`, and entry of `X` based on drop fraction.

```bash
./build_mini_dataset.py --input-file ${PERTURBX_DATASET}/replogle.h5ad --output-file ${PERTURBX_DATASET}/mini_replogle.h5ad --gene-column gene --n-samples 2 --drop-fraction 0.95 --drop-col gem_group --drop-col transcript --drop-col gene_transcript --drop-col gene_id --drop-col sgID_AB --drop-col UMI_count --drop-col z_gemgroup_UMI --drop-col core_scale_factor --drop-col mitopercent --drop-col core_adjusted_UMI_count --drop-uns --drop-var
```

For each gene in perturbed gene column, we leave only few samples (`n-samples` option). Since dataset has control cells, control also will have `n_samples` samples.

Similar for H1 cell line

```bash
./build_mini_dataset.py --input-file ${PERTURBX_DATASET}/h1.h5ad --output-file ${PERTURBX_DATASET}/perturbx/mini_h1.h5ad --gene-column "target_gene" --n-samples 2 --drop-fraction 0.95
```

# Other datasets

## Nadig 2025

Two h5ad files for two cell lines, hepg2 and jurkat, downloaded using:

```bash
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/GSE264667%5Fhepg2%5Fraw%5Fsinglecell%5F01.h5ad
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/GSE264667%5Fjurkat%5Fraw%5Fsinglecell%5F01.h5ad
```

Both files will be handled in the same way, assigned as 'train' data for the PerturbX dataset.
These files have 'var_names' (index) as 'gene_id', so needs to be replaced with 'gene_names',
which is done as part of the transform script, using the `--reindex-var-names` flag.

```bash
export NADIG_DATASET=</path/to/nadig/downloaded/h5ad/files>

# Transform
python transform.py --adata-path ${NADIG_DATASET}/GSE264667_hepg2_raw_singlecell_01.h5ad --reindex-var-names --rename-control --target-gene-column gene --control-label non-targeting
python transform.py --adata-path ${NADIG_DATASET}/GSE264667_jurkat_raw_singlecell_01.h5ad --reindex-var-names --rename-control --target-gene-column gene --control-label non-targeting

# Shuffle and build index
python shuffle_and_build_index.py --adata-path ${NADIG_DATASET}/transformed_GSE264667_hepg2_raw_singlecell_01.h5ad --target-gene-column gene --control-label Control  --chunk-size 2000 --splits train
python shuffle_and_build_index.py --adata-path ${NADIG_DATASET}/transformed_GSE264667_jurkat_raw_singlecell_01.h5ad --target-gene-column gene --control-label Control --chunk-size 2000 --splits train
```

## Jiang 2025

This is split into files for 5 different pathways that each contain 6 different cell types.
The files in `/proj/bmfm/vcc/datasets/jiang_v2.1` were preprocessed from Seurat into h5ad.
This script loads the data, extracts each cell type, and writes new h5ad files for
individual cell types (takes ~15 mins):

```python
import anndata
import os
import gc

os.environ["JIANG_DATASET"] = "/proj/bmfm/datasets/vcc/jiang2025"

pathway_files = [
    f"{os.environ["JIANG_DATASET"]}/Seurat_object_IFNB_Perturb_seq.h5ad",
    f"{os.environ["JIANG_DATASET"]}/Seurat_object_IFNG_Perturb_seq.h5ad",
    f"{os.environ["JIANG_DATASET"]}/Seurat_object_INS_Perturb_seq.h5ad",
    f"{os.environ["JIANG_DATASET"]}/Seurat_object_TGFB_Perturb_seq.h5ad",
    f"{os.environ["JIANG_DATASET"]}/Seurat_object_TNFA_Perturb_seq.h5ad",
]

pathway_adatas = [
    anndata.read_h5ad(pathway_file) for pathway_file in pathway_files
]

cell_types_found = set()
for adata in pathway_adatas:
    cell_types_found.update(adata.obs['cell_type'].unique())

for cell_type in cell_types_found:
    output_file_path = f"{os.environ["JIANG_DATASET"]}/Seurat_object_{cell_type}_Perturb_seq.h5ad"
    subsets = []
    for adata in pathway_adatas:
        pathway = adata.obs['pathway'].unique()[0]
        mask = adata.obs['cell_type'] == cell_type
        subset = adata[mask].copy()
        subset.obs_names = [f"{pathway}_{bc}" for bc in subset.obs_names]
        subsets.append(subset)
    combined = anndata.concat(subsets, axis=0, join='outer')
    combined.write_h5ad(output_file_path)
    print(f"Saved {cell_type}: {combined.shape[0]} cells")

    # In case of OOM:
    del combined, subsets
    gc.collect()
```

This gives 6 cell type files:
```
celltype_files = [
    f"{os.environ["JIANG_DATASET"]}/Seurat_object_A549_Perturb_seq.h5ad",
    f"{os.environ["JIANG_DATASET"]}/Seurat_object_BXPC3_Perturb_seq.h5ad",
    f"{os.environ["JIANG_DATASET"]}/Seurat_object_MCF7_Perturb_seq.h5ad",
    f"{os.environ["JIANG_DATASET"]}/Seurat_object_HT29_Perturb_seq.h5ad",
    f"{os.environ["JIANG_DATASET"]}/Seurat_object_K562_Perturb_seq.h5ad",
    f"{os.environ["JIANG_DATASET"]}/Seurat_object_HAP1_Perturb_seq.h5ad",
]
```

Processing them into 6 perturbx datasets. We rename target values from `NT` to `control` for compatibility with perturbation datset transformer in main code:

```bash
celltypes=(A549 BXPC3 MCF7 HT29 K562 HAP1)
TARGET_COL="gene"
CONTROL_LABEL="NT"
CHUNK_SIZE=2000
SPLITS="train"

# Currently need to switch to 'vcc_data' branch now:

for ct in "${celltypes[@]}"; do
  in_file="${JIANG_DATASET}/Seurat_object_${ct}_Perturb_seq.h5ad"

  echo "=== Transform: ${ct}"
  python transform.py \
    --adata-path "${in_file}" \
    --rename-control \
    --target-gene-column "${TARGET_COL}" \
    --control-label "${CONTROL_LABEL}"
done

# Currently need to switch to 'ptb_data' branch now:

for ct in "${celltypes[@]}"; do
  transformed_file="${JIANG_DATASET}/transformed_Seurat_object_${ct}_Perturb_seq.h5ad"

  echo "=== Shuffle + Build Index: ${ct}"
  python shuffle_and_build_index.py \
    --adata-path "${transformed_file}" \
    --target-gene-column "${TARGET_COL}" \
    --control-label Control \
    --chunk-size "${CHUNK_SIZE}" \
    --splits "${SPLITS}"

  echo "=== Done: ${ct}"
done
```

with 6 entries in the `dataset_args` found in `configs/scbert_train_perturbx_mixed_jiang.yaml`

# scPerturb CRISPRi datasets

AdamsonWeissman2016 datasets don't seem to have control cells and use limited number of specific pairs of perturbations

GasperiniShendure2019 have very confusing perturbation strings like `chr1.10201_top_two_chr11.1778_second_two_chr11.9_top_two_chr12.3850_top_two_chr1.3349_top_two_chr16.4939_top_two_chr16.5184_top_two_chr2.181_top_two_chr2.2107_top_two_chr2.2686_top_two_chr2.3453_top_two_chr3.3203_top_two_chr7.2795_top_two_chrX.333_top_two_GNPDA1`



## Tian 2019

Tian2019 have mostly single gene perturbations in iPSCs with clearly labelled controls

3 Tian 2019 datasets:

| File | Cell Line | n cells | n genes | n perturbations |
|---|---|---|---|---|
| TianKampmann2019_day7neuron | iPSC-induced neuron | 182790 | 33752 | 41358 |
| TianKampmann2019_iPSC | iPSC | 275708 | 33752 | 39162 |
| TianKampmann2021_CRISPRi | iPSC-induced neuron | 32300 | 33538 | 185 |

File preparation is standard:

```bash
export SCPERTURB_DATASET=/path/to/scperturb/datasets/

ADFILES=(TianKampmann2019_day7neuron TianKampmann2019_iPSC TianKampmann2021_CRISPRi )
TARGET_COL="perturbation"
CONTROL_LABEL="control"
CHUNK_SIZE=2000
SPLITS="train"

# Currently need to switch to 'vcc_data' branch now:

for adf in "${ADFILES[@]}"; do
  in_file="${SCPERTURB_DATASET}/${adf}.h5ad"

  echo "=== Transform: ${adf}"
  python transform.py \
    --adata-path "${in_file}" \
    --rename-control \
    --target-gene-column "${TARGET_COL}" \
    --control-label "${CONTROL_LABEL}"

  echo "=== Done transforms: ${adf}"
done

# Currently need to switch to 'ptb_data' branch now:

for adf in "${ADFILES[@]}"; do
  transformed_file="${SCPERTURB_DATASET}/transformed_${adf}.h5ad"

  echo "=== Shuffle + Build Index: ${adf}"
  python shuffle_and_build_index.py \
    --adata-path "${transformed_file}" \
    --target-gene-column "${TARGET_COL}" \
    --control-label Control \
    --chunk-size "${CHUNK_SIZE}" \
    --splits "${SPLITS}"

  echo "=== Done shuffle and build index: ${adf}"
done
```

## Tian 2019

3 Tian 2019 datasets:

| File | Cell Line | n cells | n genes | n perturbations |
|---|---|---|---|---|
| TianKampmann2019_day7neuron | iPSC-induced neuron | 182790 | 33752 | 41358 |
| TianKampmann2019_iPSC | iPSC | 275708 | 33752 | 39162 |
| TianKampmann2021_CRISPRi | iPSC-induced neuron | 32300 | 33538 | 185 |

File preparation is standard:


```bash
export SCPERTUB_DATASET=/path/to/scperturb/datasets/

ADFILES=(TianKampmann2019_day7neuron TianKampmann2019_iPSC TianKampmann2021_CRISPRi )
TARGET_COL="perturbation"
CONTROL_LABEL="control"
CHUNK_SIZE=2000
SPLITS="train"

# Currently need to switch to 'vcc_data' branch now:

for adf in "${ADFILES[@]}"; do
  in_file="${SCPERTUB_DATASET}/${adf}.h5ad"

  echo "=== Transform: ${adf}"
  python transform.py \
    --adata-path "${in_file}" \
    --rename-control \
    --target-gene-column "${TARGET_COL}" \
    --control-label "${CONTROL_LABEL}"

  echo "=== Done transforms: ${adf}"
done

# Currently need to switch to 'ptb_data' branch now:

for adf in "${ADFILES[@]}"; do
  transformed_file="${SCPERTUB_DATASET}/transformed_${adf}.h5ad"

  echo "=== Shuffle + Build Index: ${adf}"
  python shuffle_and_build_index.py \
    --adata-path "${transformed_file}" \
    --target-gene-column "${TARGET_COL}" \
    --control-label Control \
    --chunk-size "${CHUNK_SIZE}" \
    --splits "${SPLITS}"

  echo "=== Done shuffle and build index: ${adf}"
done
```


## Shifrut 2018

1 dataset, produced from matrix tar files and sgRNA csv files using the scripts in:
<link to vcc repo>

| File | Cell Line | n cells | n genes | n perturbations | n controls |
|---|---|---|---|---|---|
| GSE119450_perturbseq | CD8 T cell | 25094 | 33694 | 41358 | 20 | 3541 |

File preparation is standard:


```bash
export SHIFRUT_DATASET=/path/to/shifrut/datasets/

# Currently need to switch to 'vcc_data' branch now:
python transform.py --adata-path ${SHIFRUT_DATASET}/GSE119450_perturbseq.h5ad --rename-control --target-gene-column target_gene --control-label NonTarget

# Currently need to switch to 'ptb_data' branch now:
# Shuffle and build index
python shuffle_and_build_index.py --adata-path ${SHIFRUT_DATASET}/transformed_GSE119450_perturbseq.h5ad --target-gene-column target_gene --control-label Control  --chunk-size 2000 --splits train
```

# Dataset summaries

## After processing

| Dataset | Cell Line | n cells | n genes | n perturbations | n control |
|---|---|---:|---:|---:|---:|
| H1_train | H1 | 221273 | 18080 | 151 | 38176 |
| Replogle2022_gwps | K562 | 1546137 | 8246 | 7682 | 75328 |
| Replogle2022 | RPE1 | 213813 | 8748 | 2106 | 11485 |
| Jiang2025 | A549 | 207261 | 34494 | 219 | 9634 |
| Jiang2025 | BXPC3 | 314758 | 34494 | 219 | 18092 |
| Jiang2025 | MCF7 | 260545 | 34494 | 219 | 10819 |
| Jiang2025 | HT29 | 360963 | 34494 | 219 | 20023 |
| Jiang2025 | K562 | 207688 | 34494 | 219 | 9917 |
| Jiang2025 | HAP1 | 277261 | 34494 | 219 | 15784 |
| Nadig2025 | HEPG2 | 132566 | 9623 | 2168 | 4976 |
| Nadig2025 | jurkat | 232243 | 8881 | 2163 | 12013 |
| TianKampmann2019 | iPSC-induced neuron | 182790 | 33752 | 36081 | 15580 |
| TianKampmann2019| iPSC | 275708 | 33752 | 32992 | 10687 |
| TianKampmann2021 | iPSC-induced neuron | 32300 | 33538 | 32992 | 10687 |
| Shifrut2018 | CD8 T cell | 25094 | 33694 | 20 | 3541 |


# Building aggregate data

To build aggregate means for calculation of metrics at validation and test stages, use `build_aggregate_data` script

```bash
./build_aggregate_data.py --input-path shuffled_h1.h5ad --output-path agg_h1.h5ad --target-gene-column target_gene --control-label non-targeting --split-column split --train-split-label train
```
