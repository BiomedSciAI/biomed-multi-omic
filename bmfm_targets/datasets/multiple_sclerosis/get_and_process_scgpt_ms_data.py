"""
The scGPT paper description for the MS split:
Nine healthy control samples and 12 MS samples are included in the dataset.
We split the control samples into the reference set for model fine-tuning and
held out the MS samples as the query set for evaluation. This setting
serves as an example of out-of-distribution data.
We excluded three cell types: B cells, T cells and oligodendrocyte B cells, which only existed
in the query dataset.
The final cell counts were 7,844 in the training reference set and 13,468 in the query set.
The provided cell type labels from the original publication were used as ground truth labels for
evaluation. The data-processing protocol involved selecting HVGs to retain 3,000 genes.
scGPT share their processed datasets at:
https://figshare.com/articles/dataset/Processed_datasets_used_in_the_scGPT_foundation_model/24954519/1?file=43939560.
"""


import os
import zipfile

import anndata as ad
import numpy as np
import requests
import scanpy as sc

from bmfm_targets.datasets.datasets_utils import obs_key_wise_subsampling


def remove_col_by_regex(df_data, string):
    df_data = df_data[df_data.columns.drop(list(df_data.filter(regex=string)))]
    return df_data


def replace_str_in_list(list, old, new):
    list = list.str.replace(old, new)
    return list


def ensembl_to_gene_symbol(ids):
    server = "https://rest.ensembl.org"
    endpoint = "/lookup/id/"
    headers = {"Content-Type": "application/json"}

    ensembl_ids = [id_ for id_ in ids if id_.startswith("ENSG")]
    gene_symbols = {}
    for ensembl_id in ensembl_ids:
        url = f"{server}{endpoint}{ensembl_id}?content-type=application/json"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            gene_symbols[ensembl_id] = data.get("display_name", ensembl_id)
        else:
            gene_symbols[ensembl_id] = ensembl_id
            print(f"Failed to retrieve symbol for {ensembl_id}: {response.status_code}")

    result_list = [gene_symbols.get(id_, id_) for id_ in ids]

    return result_list


zip_url = "https://figshare.com/ndownloader/files/43939560"
ms_dir = (
    "/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/finetune/multiple_sclerosis"
)

download_dir = os.path.join(ms_dir, "downloads")
zip_path = os.path.join(download_dir, "ms_scgpt_processed.zip")
os.makedirs(download_dir, exist_ok=True)
os.makedirs(os.path.join(ms_dir, "h5ad"), exist_ok=True)
print("Downloading the zip file...")
response = requests.get(zip_url)
with open(zip_path, "wb") as file:
    file.write(response.content)
print(f"Downloaded zip file to {zip_path}")

print("Extracting the files from zip file...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(download_dir)
print(f"Files extracted to {download_dir}")

ms_zip_path = os.path.join(
    download_dir, "scGPT_processed_datasets/Fig2_Annotation/ms-20240103T153851Z-001.zip"
)
print("Extracting the MS files from zip file...")
with zipfile.ZipFile(ms_zip_path, "r") as zip_ref:
    zip_ref.extractall(download_dir)
print(f"Files extracted to {download_dir}")

ms_train = sc.read_h5ad(os.path.join(download_dir, "ms/c_data.h5ad"))
ms_test = sc.read_h5ad(os.path.join(download_dir, "ms/filtered_ms_adata.h5ad"))

# translate Ensembl IDs to gene names
ms_train.var_names = ensembl_to_gene_symbol(ms_train.var_names)
ms_test.var_names = ensembl_to_gene_symbol(ms_test.var_names)

split_train_dev = np.random.choice(
    ["train", "dev"], size=ms_train.shape[0], p=[0.7, 0.3]
)
ms_train.obs["split_cross"] = split_train_dev
ms_test.obs["split_cross"] = "test"

ms_all = ad.concat([ms_train, ms_test], axis=0, join="outer", merge="same")


# preprocessing
ms_all.obs = remove_col_by_regex(ms_all.obs, "Ontology")
ms_all.obs = remove_col_by_regex(ms_all.obs, "Factor Value\\[disease\\]")
ms_all.obs = remove_col_by_regex(ms_all.obs, "Factor Value\\[sampling site\\]")

ms_all.obs.columns = replace_str_in_list(
    ms_all.obs.columns, "Sample Characteristic[", ""
)
ms_all.obs.columns = replace_str_in_list(ms_all.obs.columns, "Factor Value[", "")
ms_all.obs.columns = replace_str_in_list(ms_all.obs.columns, "]", "")
ms_all.obs.columns = replace_str_in_list(
    ms_all.obs.columns, "celltype", "scgpt_version_celltype"
)
ms_all.obs.columns = replace_str_in_list(
    ms_all.obs.columns, "inferred cell type - ontology labels", "celltype"
)

ms_all.obs.index.name = "index"
# Convert the counts_matrix to CSR format to enable saving to h5ad
ms_all.X = ms_all.X.tocsr()

ms_all.write_h5ad(os.path.join(ms_dir, "h5ad/multiple_sclerosis_scgpt_split.h5ad"))

# Save a random 500 samples for tests
ms_for_tests = obs_key_wise_subsampling(adata=ms_all, obs_key="celltype", N=800)
ms_for_tests.write_h5ad(
    os.path.join(
        os.getcwd(),
        "bmfm_targets/tests/resources/finetune/multiple_sclerosis/h5ad/multiple_sclerosis_scgpt_split.h5ad",
    )
)
