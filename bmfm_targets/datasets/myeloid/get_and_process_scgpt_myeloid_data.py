"""
The myeloid dataset30 can be accessed from the Gene Expression
Omnibus (GEO) database using accession number GSE154763. The
dataset consists of nine different cancer types, but, for the purpose of
training and evaluating the model, six cancer types were selected in
the reference set for training, while three cancer types were used for
the query set. The reference set contains myeloid cancer types UCEC,
PAAD, THCA, LYM, cDC2 and kidney, while the query set contains MYE,
OV-FTC and ESCA. The dataset was also randomly subsampled. The final
cell counts were 9,748 in the reference set and 3,430 in the query set.
Three thousand HVGs were selected during data processing.
scGPT share their processed datasets at:
https://figshare.com/articles/dataset/Processed_datasets_used_in_the_scGPT_foundation_model/24954519/1?file=43939560.
"""


import os
import zipfile

import anndata as ad
import numpy as np
import requests
import scanpy as sc
from datasets_utils import obs_key_wise_subsampling
from scipy.sparse import csr_matrix


def replace_str_in_list(list, old, new):
    list = list.str.replace(old, new)
    return list


zip_url = "https://figshare.com/ndownloader/files/43939560"
mye_dir = "/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/finetune/myeloid"

download_dir = os.path.join(mye_dir, "downloads")
zip_path = os.path.join(download_dir, "myeloid_scgpt_processed.zip")
os.makedirs(download_dir, exist_ok=True)
os.makedirs(os.path.join(mye_dir, "h5ad"), exist_ok=True)
print("Downloading the zip file...")
response = requests.get(zip_url)
with open(zip_path, "wb") as file:
    file.write(response.content)
print(f"Downloaded zip file to {zip_path}")

print("Extracting the files from zip file...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(download_dir)
print(f"Files extracted to {download_dir}")

mye_zip_path = os.path.join(
    download_dir,
    "scGPT_processed_datasets/Fig2_Annotation/mye-20240103T153920Z-001.zip",
)
print("Extracting the Myeloid files from zip file...")
with zipfile.ZipFile(mye_zip_path, "r") as zip_ref:
    zip_ref.extractall(download_dir)
print(f"Files extracted to {download_dir}")

mye_train = sc.read_h5ad(os.path.join(download_dir, "mye/reference_adata.h5ad"))
mye_test = sc.read_h5ad(os.path.join(download_dir, "mye/query_adata.h5ad"))

split_train_dev = np.random.choice(
    ["train", "dev"], size=mye_train.shape[0], p=[0.7, 0.3]
)
mye_train.obs["split_cross"] = split_train_dev
mye_test.obs["split_cross"] = "test"

mye_all = ad.concat([mye_train, mye_test], axis=0, join="outer", merge="same")


# preprocessing
mye_all.obs.columns = replace_str_in_list(mye_all.obs.columns, "cell_type", "celltype")

mye_all.obs.index.name = "index"
# Convert the counts_matrix to CSR format to enable saving to h5ad
mye_all.X = csr_matrix(mye_all.X)

# Save processed myeloid dataset
mye_all.write_h5ad(os.path.join(mye_dir, "h5ad/myeloid_scgpt_split.h5ad"))

# Save a random 500 samples for tests
mye_for_tests = obs_key_wise_subsampling(adata=mye_all, obs_key="celltype", N=500)
mye_for_tests.write_h5ad(
    os.path.join(
        os.getcwd(),
        "bmfm_targets/tests/resources/finetune/myeloid/h5ad/myeloid_scgpt_split.h5ad",
    )
)
