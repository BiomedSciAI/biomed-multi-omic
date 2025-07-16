# biomed-multi-omic

Biomedical foundational models for omics data. This package supports the development of foundation models for scRNA or for DNA data. But, in this readme, we will specifically focus on how to use it for DNA data.

### Highlights
- 🧬 A single package for DNA Foundation models which can pre-train on reference human genome (GRCh38/hg38) and also variant imputed genome based on common SNPs available from GWAT catalog and ClinVar datasets
- 🚀 Built around  HuggingFace transformers and PyTorchLighting to integrate into exsisting software stacks
- 📈 Increased performance on predicting several biological tasks involving DNA sequences, e.g., promoter prediction task and regulatory regions using Massively parallel reporter assays (MPRAs)
- 🔬 Most expensive package for Trancriptomic Foundation Models (TPM) and Genomics Foundation Models (GFM)

## Contents

1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
3. [Quick start](#3-quick-start)
4. [Contributing](#4-contributing)
5. [Citation](#5-citation)

## Introduction

`biomed-multi-omic` is a comprehensive software package that not only facilitates this combinatorial exploration but is also inherently flexible and easily extensible to incorporate novel methods as the field continues to advance. It also facilitates downstream tasks such as promoter prediction task, splice site prediction, transcription factor prediction and promoter affect on gene expression prediction task using benchmark datasets such as GUE from DNA-BERT2 and MPRA data. 

## Package Architecture

### DNA Modules

The `bmfm-dna` framework addresses key limitations of existing DNA language models by incorporating natural genomic variations into the pre-training process, rather than relying solely on the reference genome. This allows the model to better capture critical biological properties, especially in regulatory regions where many disease-associated variants reside. As a result, `bmfm-dna` offers a more comprehensive and biologically meaningful representation, advancing the field beyond traditional DNALM strategies.

`bmfm-dna` framework diagram schematic shows the modules avaliable for multiple strategies to encode natural genomic variations; multiple architectures such as BERT, Performer, ModernBERT to build genomic foundation models; fine-tuning and benchmarking of the foundation models on well-established biologically meaningful tasks. In particular, the package incorporates most of the benchmarking datasets from Genomic Understanding and Evaluation (GUE) package released in DNABERT-2. In addition, the package also supports promoter activity prediction on datasets created using Massive Parallel Reporting Assays (MPRA), and SNP-disease association prediction.

![bmfm_dna](../docs/images/dna_fig1.png)


## Installation

We recommend using [uv](https://github.com/astral-sh/uv) to create your enviroment due to it's 10-100x speed up over pip.

Install uisng `uv` (recommended):
```sh
python -m venv .venv
source .venv/bin/activate
pip install uv
uv pip install "git+github.com/BiomedSciAI/biomed-multi-omic.git"
```

Install uisng `pip`:
```sh
python -m venv .venv
source .venv/bin/activate
pip install "git+github.com/BiomedSciAI/biomed-multi-omic.git"
```

Install using cloned repo:
```sh
# Clone the repository
git clone git@github.com:BiomedSciAI/biomed-multi-omic.git

# Change directory to the root of the cloned repository
cd biomed-multi-omics
# recommended venv setup (vanilla pip and conda also work)
uv venv .venv -p3.12
source ./.venv/bin/activate
uv pip install -e .
```
<!-- 
### Optional dependencies
In addition to the base package there are additional optional dependencies which extends `bmfm-mulit-omics` capabilites further. These include:
- `bulk_rna`: Extends modules for extracting and preprocessing bulk RNA-seq data
- `benchmarking`: Installs additional models used benchmark `bmfm-mulit-omics` against. These include scib, scib-metrics, pyliger, scanorama and harmony-pytorch.
- `test`: Unittest suite which is recommended for development use

To install optional dependencies from this GitHub repository you can use the following structure:

```sh
uv pip install "git+github.com/BiomedSciAI/biomed-multi-omic.git#egg=package[bulk_rna,benchmarking,test,notebook]"
``` -->

## Quick start


### Downloading DNA checkpoint

The model's weights can be aquired from [IBM's HuggingFace collection](https://huggingface.co/ibm-research). The following DNA models are avaliable:

- MLM+REF_GENOME [ibm-research/biomed.dna.ref.modernbert.113m](https://huggingface.co/ibm-research/biomed.dna.ref.modernbert.113m.v1)
- MLM+REFSNP_GENOME [ibm-research/biomed.dna.snp.modernbert.113m](https://huggingface.co/ibm-research/biomed.dna.snp.modernbert.113m.v1)

For details on how the models were trained, please refer to [the BMFM-DNA preprint](https://arxiv.org/abs/2507.05265).

### Zero-shot inference (CLI)
`biomed-multi-omic` allows for multiple input data types, but the core data object at the heart of the tool is based around the adata object. Once your dataset is created.

To get scRNA embeddings and zero shot cell-type predictions:

```bash
export MY_DATA_FILE=... # path to h5ad file with raw counts and gene symbols
bmfm-targets-run -cn predict input_file=$MY_DATA_FILE working_dir=/tmp checkpoint=ibm-research/biomed.dna.snp.modernbert.113m.v1
```

### Zero-shot inference (programmatically)
To run inference programmatically, you can see a zero-shot example in this [notebook](notebooks/1_zero_shot_cell_type_anno.ipynb).

Note to use the notebook you will need to install the `notebook` optional dependencies (see [Installation](#2-installation))


## Contributing

### Running pre-training framework 

Our framework supports running pretraining framework using MLM or supervised loss on a class label or both. Please refer to this [readme](README.md) for details on running pre-training on both scRNA and DNA framework. 

For pre-processing DNA datasets using both reference and SNPified version, please use the [steps](README_SNP_PREPROCESSING.md) for pre-processing before running the pre-training framework.


### Snpification of the finetuning data

We preprocessed a few datasets to impute SNPs extracting from the reference genome. The easiest way to impute such SNPs is to map each input dna sequence to the reference geneome if the chromose and position location of the sequence is availabe. For example, we extracted the promoter location from [here](https://genome.ucsc.edu/cgi-bin/hgTables) provided by EPDNew. Then we use the [notebook script](datasets/dnaseq/preprocess_dataset/snpify_promoter_dnabert2_v1.ipynb) to preprocess the promoter dataset to impute SNPs. In this version, the negative sequences were imputed with random SNPs coming from the same distribution of the positive set (Class 1 of the paper). 

For other types of SNPification of data, we had different scripts which are available upon request. 


### Running fine-tuning tasks of DNA

Please refer to the [readme](evaluation/benchmark_configs_dna/README.md) for running the 6 finetuning tasks of DNA. 


### Fine-tuning on a biological task containing DNA-sequences

For fine-tuning DNA pre-trained model on a new biological task involves first creating a dataset folder with three files train.csv, test.csv and dev.csv. The framework will 
look for these files for model development automatically. Each file should contains at least two columns. The first column must contain the dna sequence and then followed by the class labels, where column names are passed in the LabelColumnInfo yaml. 
Additional columns (e.g., seq_id) can follow in each of the files which will not be used.

As an example of 'Sample' dataset with the multiclass prediction problem where there are two regression lables measuring gene expression in types of genes: development and housekeeping (Dev_enrichment, HK_enrichment), the dataset siles should be like follows:

```csv
sequence, Dev_enrichment, HK_enrichment, seq_id
ACGTTTACCCCTGGGTAAG, -0.24, 0.35, seq_99
```

and the corresponding datamodule and LabelInfo yaml cofig should be modified as:

```
drosophilla_labelcolumns.yaml
- _target_: bmfm_targets.config.LabelColumnInfo
  label_column_name: "Dev_log2_enrichment"
  is_regression_label: true
- _target_: bmfm_targets.config.LabelColumnInfo
  label_column_name: "Hk_log2_enrichment"
  is_regression_label: true


drosophilla_datamodule.yaml 
  defaults:
  - base_seq_cls

  _target_: bmfm_targets.datasets.dnaseq.DNASeqDrosophilaEnhancerDataModule
  max_length: 80
  dataset_kwargs:
    processed_data_source: ${input_directory}/drosophila_enhancer_prediction
    dataset_name: ${dataset_name}
  
  ```

Then run the [yaml](run/dna_finetune_train_and_test_config.yaml).

## Citation

To cite the tool for both RNA and DNA, please cite both the following articles:
```
@misc{dandala2025bmfmrnaopenframeworkbuilding,
      title={BMFM-RNA: An Open Framework for Building and Evaluating Transcriptomic Foundation Models},
      author={Bharath Dandala and Michael M. Danziger and Ella Barkan and Tanwi Biswas and Viatcheslav Gurev and Jianying Hu and Matthew Madgwick and Akira Koseki and Tal Kozlovski and Michal Rosen-Zvi and Yishai Shimoni and Ching-Huei Tsou},
      year={2025},
      eprint={2506.14861},
      archivePrefix={arXiv},
      primaryClass={q-bio.GN},
      url={https://arxiv.org/abs/2506.14861},
}

@misc{li2025bmfmdnasnpawarednafoundation,
      title={BMFM-DNA: A SNP-aware DNA foundation model to capture variant effects}, 
      author={Hongyang Li and Sanjoy Dey and Bum Chul Kwon and Michael Danziger and Michal Rosen-Tzvi and Jianying Hu and James Kozloski and Ching-Huei Tsou and Bharath Dandala and Pablo Meyer},
      year={2025},
      eprint={2507.05265},
      archivePrefix={arXiv},
      primaryClass={q-bio.GN},
      url={https://arxiv.org/abs/2507.05265}, 
}
```
