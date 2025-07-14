# biomed-multi-omic

Biomedical foundational models for omics data.
This package supports the development of foundation models for scRNA or for DNA data.
For details see the preprints:

- BMFM-RNA: An Open Framework for Building and Evaluating Transcriptomic Foundation Models, Bharath Dandala, Michael M. Danziger, Ella Barkan, Tanwi Biswas, Viatcheslav Gurev, Jianying Hu, Matthew Madgwick, Akira Koseki, Tal Kozlovski, Michal Rosen-Zvi, Yishai Shimoni, Ching-Huei Tsou, [arXiv:2506.14861](https://arxiv.org/abs/2506.14861)
- BMFM-DNA: A SNP-aware DNA foundation model to capture variant effects
Hongyang Li, Sanjoy Dey, Bum Chul Kwon, Michael M. Danziger, Michal Rosen-Tzvi, Jianying Hu, James Kozloski, Ching-Huei Tsou, Bharath Dandala, Pablo Meyer [arXiv:2507.05265](https://www.arxiv.org/abs/2507.05265)

## Install

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

## Example usage

To get scRNA embeddings and zero shot cell-type predictions:

```bash
export MY_DATA_FILE=... # path to h5ad file with raw counts and gene symbols
bmfm-targets-run -cn predict input_file=$MY_DATA_FILE working_dir=/tmp checkpoint=ibm-research/biomed.rna.bert.110m.wced.multitask.v1
```

For more options see [run](./run/README.md).

## Checkpoints

### scRNA

- MLM+RDA [ibm-research/biomed.rna.bert.110m.mlm.rda.v1](https://huggingface.co/ibm-research/biomed.rna.bert.110m.mlm.rda.v1)
- MLM+Multitask [ibm-research/biomed.rna.bert.110m.mlm.multitask.v1](https://huggingface.co/ibm-research/biomed.rna.bert.110m.mlm.multitask.v1)
- WCED+Multitask [ibm-research/biomed.rna.bert.110m.wced.multitask.v1](https://huggingface.co/ibm-research/biomed.rna.bert.110m.wced.multitask.v1)
- WCED 10 pct [ibm-research/biomed.rna.bert.110m.wced.v1](https://huggingface.co/ibm-research/biomed.rna.bert.110m.wced.v1)

For details on how the models were trained, please refer to [the BMFM-RNA preprint](https://arxiv.org/abs/2506.14861).

### DNA

- MLM+REF_GENOME [ibm-research/biomed.dna.ref.modernbert.113m](https://huggingface.co/ibm-research/biomed.dna.ref.modernbert.113m.v1)
- MLM+REFSNP_GENOME [ibm-research/biomed.dna.snp.modernbert.113m](https://huggingface.co/ibm-research/biomed.dna.snp.modernbert.113m.v1)


## RNA processing overview

![Diagram of package flow for RNA model building](./docs/images/package_diagram.png)
For more details, check out the [the BMFM-RNA preprint](https://arxiv.org/abs/2506.14861)..


### DNA processing overview

The `bmfm-dna` framework addresses key limitations of existing DNA language models by incorporating natural genomic variations into the pre-training process, rather than relying solely on the reference genome. This allows the model to better capture critical biological properties, especially in regulatory regions where many disease-associated variants reside. 

![bmfm_dna](docs/images/dna_fig1.png)


`bmfm-dna` framework diagram schematic shows the modules avaliable for multiple strategies to encode natural genomic variations; fine-tuning and benchmarking of the pre-trained models on well-established biologically meaningful tasks, such as Evaluation (GUE) package released in DNABERT-2, promoter activity prediction on Massive Parallel Reporting Assays (MPRA) dataset, and SNP-disease association prediction.

For more details, check out the [the BMFM-DNA preprint](https://www.arxiv.org/abs/2507.05265)..


## Credits

This project was created in IBM Research.
