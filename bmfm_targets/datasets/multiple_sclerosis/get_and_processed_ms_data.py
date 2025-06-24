import os
import zipfile

import pandas as pd
import requests
import scanpy as sc
import scipy.io


def remove_col_by_regex(df_data, string):
    df_data = df_data[df_data.columns.drop(list(df_data.filter(regex=string)))]
    return df_data


def replace_str_in_list(list, old, new):
    list = list.str.replace(old, new)
    return list


zip_url = "https://www.ebi.ac.uk/gxa/sc/experiment/E-HCAD-35/download/zip?fileType=quantification-raw&accessKey="
metadata_url = "https://www.ebi.ac.uk/gxa/sc/experiment/E-HCAD-35/download?fileType=experiment-design&accessKey="
ms_dir = (
    "/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/finetune/multiple_sclerosis"
)
download_dir = os.path.join(ms_dir, "downloads")
download_exp_path = os.path.join(
    download_dir, "E-HCAD-35-quantification-raw-files.zip"
)  # raw data
download_metadata_path = os.path.join(download_dir, "metadata.tsv")


os.makedirs(download_dir, exist_ok=True)
os.makedirs(os.path.join(ms_dir, "h5ad"), exist_ok=True)

print("Downloading the zip file...")
response = requests.get(zip_url)
with open(download_exp_path, "wb") as file:
    file.write(response.content)
print(f"Downloaded zip file to {download_exp_path}")


print("Extracting the files from zip file...")
with zipfile.ZipFile(download_exp_path, "r") as zip_ref:
    zip_ref.extractall(download_dir)
print(f"Files extracted to {download_dir}")


print("Downloading the metadata TSV file...")
metadata_response = requests.get(metadata_url)
with open(download_metadata_path, "wb") as metadata_file:
    metadata_file.write(metadata_response.content)
print(f"Downloaded metadata TSV file to {download_metadata_path}")

# Load the metadata
metadata = pd.read_csv(download_metadata_path, sep="\t")


metadata = remove_col_by_regex(metadata, "Ontology")
metadata = remove_col_by_regex(metadata, "Factor Value\\[disease\\]")
metadata = remove_col_by_regex(metadata, "Factor Value\\[sampling site\\]")

metadata.columns = replace_str_in_list(metadata.columns, "Sample Characteristic[", "")
metadata.columns = replace_str_in_list(metadata.columns, "Factor Value[", "")
metadata.columns = replace_str_in_list(metadata.columns, "]", "")
metadata.columns = replace_str_in_list(
    metadata.columns, "inferred cell type - ontology labels", "celltype"
)


# Define paths to the files extracted from the zip file
mtx_file = os.path.join(download_dir, "E-HCAD-35.aggregated_filtered_counts.mtx")

mtx_cols_file = os.path.join(
    download_dir, "E-HCAD-35.aggregated_filtered_counts.mtx_cols"
)

mtx_rows_file = os.path.join(
    download_dir, "E-HCAD-35.aggregated_filtered_counts.mtx_rows"
)


counts_matrix = scipy.io.mmread(mtx_file)
genes = pd.read_csv(mtx_rows_file, header=None, sep="\t")
barcodes = pd.read_csv(mtx_cols_file, header=None, sep="\t")

print(f"Matrix shape: {counts_matrix.shape}")
print(f"Genes (rows) shape: {genes.shape}")
print(f"Barcodes (cols) shape: {barcodes.shape}")

adata = sc.AnnData(
    X=counts_matrix.T
)  # Transpose to make the dimensions consistent with scanpy (cells x genes)

# Assign the gene names (rows) and barcodes (columns) to the AnnData object
adata.var["gene_symbols"] = genes[0].values
adata.obs["barcodes"] = barcodes[0].values

# Merge the downloaded metadata with the existing barcodes
adata.obs = adata.obs.merge(metadata, left_on="barcodes", right_on="Assay", how="left")

# Convert the counts_matrix to CSR format to enable saving to h5ad
adata.X = adata.X.tocsr()

# Save the AnnData object to an h5ad file
h5ad_file_path = os.path.join(ms_dir, "h5ad/multiple_sclerosis_raw_data.h5ad")
adata.write(h5ad_file_path)
print(f"Data has been saved to '{h5ad_file_path}'.")
