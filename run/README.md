# biomed-multi-omic entrypoint: `bmfm-targets-run`

## scRNA zero shot inference

To get embeddings for an h5ad file from the checkpoints discussed in the manuscript ( <https://arxiv.org/abs/2506.14861> ) run the following code snippets, after installing the package.

The only thing you need is an h5ad file with raw gene counts to run inference, and a writable directory `working_dir` for output. For convenience, this page assumes that the location of the file is stored to an environment variable. Checkpoints will be downloaded automatically from HuggingFace.

```bash
export MY_DATA_FILE=# h5ad file with raw counts and genes identified by gene symbol
```

The program will produce embeddings in `working_dir/embeddings.csv` and predictions in `working_dir/predictions.csv` as csv files indexed with the same `obs` index as the initial h5ad file.

### MLM+RDA

```bash
bmfm-targets-run -cn predict input_file=$MY_DATA_FILE working_dir=/tmp data_module.collation_strategy=language_modeling ++data_module.rda_transform=auto_align data_module.log_normalize_transform=false data_module.max_length=4096 checkpoint=ibm-research/biomed.rna.bert.110m.mlm.rda.v1
```

### MLM+Multitask

```bash
bmfm-targets-run -cn predict input_file=$MY_DATA_FILE working_dir=/tmp data_module.max_length=4096 checkpoint=ibm-research/biomed.rna.bert.110m.mlm.multitask.v1
```

### WCED+Multitask

```bash
bmfm-targets-run -cn predict input_file=$MY_DATA_FILE working_dir=/tmp checkpoint=ibm-research/biomed.rna.bert.110m.wced.multitask.v1
```

### WCED 10 pct

```bash
bmfm-targets-run -cn predict input_file=$MY_DATA_FILE working_dir=/tmp data_module.collation_strategy=language_modeling checkpoint=ibm-research/biomed.rna.bert.110m.wced.v1
```

## scRNA fine-tuning

The requirements are the same as for inference, except that to do fine-tuning you need a split defined, and you need to supply the name of the column that you would like to predict.
You can either add one manually to a column of your choice of the `obs` DataFrame, or you can get a quick random split with this utility:

```python
from os import environ
import scanpy as sc
import bmfm_targets.datasets.datasets_utils as du

ad = sc.read_h5ad(environ["MY_DATA_FILE"])
ad.obs["split_random"] = du.get_random_split(ad.obs, {"train":0.8, "dev": 0.1, "test": 0.1},random_state=42)
ad.write_h5ad(environ["MY_DATA_FILE"])
```

If you use your own split column modify `split_column_name=null` below to your chosen column name, otherwise it will look for a `split_random` column, as created above.

### MLM+RDA

```bash
bmfm-targets-run -cn finetune label_column_name=celltype split_column_name=null input_file=$MY_DATA_FILE working_dir=/tmp ++data_module.rda_transform=auto_align data_module.log_normalize_transform=false data_module.max_length=4096 checkpoint=ibm-research/biomed.rna.bert.110m.mlm.rda.v1
```

### MLM+Multitask

```bash
bmfm-targets-run -cn finetune label_column_name=celltype split_column_name=null input_file=$MY_DATA_FILE working_dir=/tmp data_module.max_length=4096 checkpoint=ibm-research/biomed.rna.bert.110m.mlm.multitask.v1
```

### WCED+Multitask

```bash
bmfm-targets-run  -cn finetune label_column_name=celltype split_column_name=null input_file=$MY_DATA_FILE working_dir=/tmp checkpoint=ibm-research/biomed.rna.bert.110m.wced.multitask.v1
```

### WCED 10 pct

```bash
bmfm-targets-run -cn finetune label_column_name=celltype split_column_name=null input_file=$MY_DATA_FILE working_dir=/tmp checkpoint=ibm-research/biomed.rna.bert.110m.wced.v1
```

## scRNA Pre-training

As above, the user must prepare an h5ad file of raw counts

The only thing you need is an h5ad file with raw gene counts to run inference, and a writable directory `working_dir` for output. For convenience, this page assumes that the location of the file is stored to an environment variable. Checkpoints will be downloaded automatically from HuggingFace.

```bash
export MY_DATA_FILE=# h5ad file with raw counts and genes identified by gene symbol
```

If you have not defined a split column `"split_random"` as above, please define a split column and set it in the yaml.

### MLM+RDA

```bash
bmfm-targets-run -cn rda_mlm input_file=$MY_DATA_FILE checkpoint=ibm-research/biomed.rna.bert.110m.mlm.rda.v1
```

## DNA Fine-tuning

### Fine-tuning on a biological task containing DNA-sequences

For fine-tuning DNA pre-trained model on a new biological task involves first creating a dataset folder with three files train.csv, test.csv and dev.csv. The framework will
look for these files for model development automatically. Each file should contains at least two columns. The first column must contain the dna sequence and then followed by the class labels, where column names are passed in the LabelColumnInfo yaml.
Additional columns (e.g., seq_id) can follow in each of the files which will not be used.

As an example of 'Sample' dataset with the multiclass prediction problem where there are two regression labels measuring gene expression in types of genes: development and housekeeping (Dev_enrichment, HK_enrichment), the dataset siles should be like follows:

```csv
sequence,Dev_enrichment,HK_enrichment,seq_id
ACGTTTACCCCTGGGTAAG,-0.24,0.35,seq_99
```

Next, the yaml file has to be created properly. A simple finetuning yaml for single classification task is provided [here](./dna_finetune_train_and_test_config.yaml).

For a new dataset such as the drosophilla expression prediction task, the corresponding datamodule and LabelInfo yaml should be overridden as belows:

```yaml
label_columns:
- _target_: bmfm_targets.config.LabelColumnInfo
  label_column_name: "Dev_log2_enrichment"
  is_regression_label: true
- _target_: bmfm_targets.config.LabelColumnInfo
  label_column_name: "Hk_log2_enrichment"
  is_regression_label: true

data_module:
    defaults: dna_base_seq_cls
    max_length: 80
    dataset_kwargs:
      processed_data_source: ${input_directory}
      dataset_name: ${dataset_name}
      label_dict_path: ${input_directory}/${dataset_name}_all_labels.json


trainer:
  learning_rate: ${learning_rate}
  losses:
    - name: mse
      label_column_name: ${label_columns[0].label_column_name}
    - name: mse
      label_column_name: ${label_columns[1].label_column_name}

  ```


```bash
export INPUT_DIRECTORY=... # path to the three (train/test/dev.csv) files
bmfm-targets-run -cn dna_finetune_train_and_test_config input_directory=$INPUT_DIRECTORY output_directory=/tmp checkpoint=ibm-research/biomed.dna.snp.modernbert.113m.v1
```

### Running benchmarking fine-tuning tasks of DNA

Please refer to the [readme](../bmfm_targets/evaluation/benchmark_configs_dna/README.md) for running the 6 benchmarking finetuning tasks of DNA, discussed in the preprint.



## DNA Pretraining


### Running pre-training framework

Our framework supports running pretraining framework using MLM or supervised loss on a class label or both.

For pre-processing DNA datasets using both reference and SNPified version, please use the [steps](../bmfm_targets/README_SNP_PREPROCESSING.md) for pre-processing before running the pre-training framework.


### Snpification of the finetuning data

We preprocessed a few datasets to impute SNPs extracting from the reference genome. The easiest way to impute such SNPs is to map each input dna sequence to the reference geneome if the chromose and position location of the sequence is availabe. For example, we extracted the promoter location from [here](https://genome.ucsc.edu/cgi-bin/hgTables) provided by EPDNew. Then we use the [notebook script](../bmfm_targets/datasets/dnaseq/preprocess_dataset/snpify_promoter_dnabert2_v1.ipynb) to preprocess the promoter dataset to impute SNPs. In this version, the negative sequences were imputed with random SNPs coming from the same distribution of the positive set (Class 1 of the paper). Note that the notebook requires reference genome fasta data (fasta_path), preprocessed SNPified chromosome-wise data (cell 4 of the notebook) for both forward and reverse strands, which can be downloaded from [here](https://zenodo.org/records/15981429).

For other types of SNPification of data, we had different scripts which are available upon request.
