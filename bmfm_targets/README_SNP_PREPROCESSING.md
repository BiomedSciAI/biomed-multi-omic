# bmfm-targets preprocessing

Instructions for how to run preprocessing for the datasets that require dedicated preprocessing before loading them with the DataModule.

[LitData](https://github.com/Lightning-AI/litdata)  is a python package to transform and optimize data in both on-prem and cloud storage environments efficiently and intuitively, at any scale. We use litdata for efficient distributed training as it enables users to seamlessly stream data of any size to one or multiple machines.

## Create litdata for SNP genomic sequences

### Split raw sequences to parquet chunks
```
python bmfm_targets/datasets/SNPdb/snp_data_splitter -i ./raw/pretraining_data/biallele_20kb -o ./raw/parquet/biallele_20kb
```

### Create SNP tokenizer using parquet files
```
python bmfm_targets/tokenization/create/create_snp2vec_tokenizer.py -i ./raw/parquet/biallele_20kb -o bmfm_targets/tokenization/snp_vocab_2kb/snp2vec_tokenizer
```

### Convert raw chunks to tokenized lit data format
```
python -m bmfm_targets.datasets.data_conversion.parquet2litdata -i ./raw/parquet/biallele_1kb -o ./litdata/biallele_1kb --tokenizer bmfm_targets/tokenization/snp_vocab/tokenizers/dna_chunks
```

### Add environment variables before training e2e
```
export DATA_OPTIMIZER_CACHE_FOLDER=
export DATA_OPTIMIZER_DATA_CACHE_FOLDER=
```
