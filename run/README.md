# biomed-multi-omic entrypoint: `bmfm-targets-run`

To get embeddings for an h5ad file from the checkpoints discussed in the manuscript ( <https://arxiv.org/abs/2506.14861> ) run the following code snippets, after installing the package.

The only thing you need is an h5ad file with raw gene counts to run inference, and a writable directory `working_dir` for output. For convenience, this page assumes that the location of the file is stored to an environment variable. Checkpoints will be downloaded automatically from HuggingFace.

```bash
export MY_DATA_FILE=# h5ad file with raw counts and genes identified by gene symbol
```

The program will produce embeddings in `working_dir/embeddings.csv` and predictions in `working_dir/predictions.csv` as csv files indexed with the same `obs` index as the initial h5ad file.

## MLM+RDA

```bash
bmfm-targets-run -cn predict input_file=$MY_DATA_FILE working_dir=/tmp data_module.collation_strategy=language_modeling checkpoint=ibm-research/biomed.rna.bert.110m.mlm.rda.v1
```

## MLM+Multitask

```bash
bmfm-targets-run -cn predict input_file=$MY_DATA_FILE working_dir=/tmp checkpoint=ibm-research/biomed.rna.bert.110m.mlm.multitask.v1
```

## WCED+Multitask

```bash
bmfm-targets-run -cn predict input_file=$MY_DATA_FILE working_dir=/tmp checkpoint=ibm-research/biomed.rna.bert.110m.wced.multitask.v1
```

## WCED 10 pct

```bash
bmfm-targets-run -cn predict input_file=$MY_DATA_FILE working_dir=/tmp data_module.collation_strategy=language_modeling checkpoint=ibm-research/biomed.rna.bert.110m.wced.v1
```
