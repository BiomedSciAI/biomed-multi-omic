### CellXGene dataset

#### CellXGeneNexus TileDB Support

The recommended way to access CellXGene data is via the litdata streaming dataset called CellXGeneNexusDataset.

Before using it, the user must have:
  1.  a `URI` pointing to a path of a TileDB copy of the cellxgene data. If URI=None, the public version will be requested
from AWS S3. This works, but is slow.
  2. an `index_dir` with a litdata index created for the copy of the CellXGene data reqeusted. This is not optional. To create a litdata index, look at the dataset-level split generating code in [cellxgene_dataset_split](./cellxgene_dataset_split.ipynb) and the litdata index creation code in [cellxgene_soma_utils](./cellxgene_soma_utils.py) and [cellxgene_nexus_index](./cellxgene_nexus_index.ipynb).

Examples of training yamls with existing nexus index dirs can be found in the tasks dir.

As of March 2025, the 2023-12-15 release of CellXGene is available on CCC and the 2025-01-30 release is available on Zuvela.
This is subject to change. The splits for the 2023-12-15 release are in this folder, and the splits for the 2025-01-30 release
are in ["2025-01-30"](./2025-01-30/).


#### H5AD File support
The cellXgene dataset contains over 33M samples from different tissues, suspension types, developmental stages and disease states.
For details on the dataset and the metadata it holds, visit: https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/3.0.0/schema.md#cell_type_ontology_term_id.
In the [review_cellxgene_metadata.ipynb](bmfm_targets/datasets/cellxgene/review_cellxgene_metadata.ipynb) notebook there is a detailed print of the whole dataset metadata and the unique values it holds in each field.

The cellXgene dataset was partitioned to 167 files (166 out of them contains 200K samples) and is saved on CCC at:
[/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/pretrain/cellxgene/anndata_all/all_unique_cells/](/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/pretrain/cellxgene/anndata_all/all_unique_cells/)

In order to use this dataset for training/prediction, the steps are:

1. First, the user need to select how many partition files he want to use.
Then, make a copy of a concatenated file to the h5ad subfolder by running the preprocessing script:

    - [bmfm_targets/datasets/cellxgene/process_cellxgene_partition_files.py](./process_cellxgene_partition_files.py)
    and adjusting the `n_files` parameter at `line 82`.
    Preprocessing includes editing the obs object indices (it shouldn't include only numbers) and fix gene names from number coding to their names.
    - In this script the user also need to decide on filter terms to apply on the cellxgene dataset. The default is to filter all records of 'suspension_type' equals 'cell' (leaving out all nucleus records).

2. Given the h5ad processed file, the user should create the cellXgene cell type labels dictionary by running the following file:
[bmfm_targets/datasets/cellxgene/create_cellxgene_labels_json_file.py](bmfm_targets/datasets/cellxgene/create_cellxgene_labels_json_file.py).
The cell type labels dictionary can also be generated on the fly by setting in [scbert_train_cellxgene.yaml](bmfm_targets/tasks/scbert/scbert_train_cellxgene.yaml) file the following data_module arg:
```
data_module:
  dataset_kwargs:
    label_columns: ["cell_type_ontology_term_id"]
```
