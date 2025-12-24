# Perturbx dataset

This dataset is desined to combine multiple cell lines in perturbation training and validation. It is designed to yield samples of randomly selected pairs of control and perturbed cell from the same cell line and mix training batches with samples from different cell lines.

## Dataset preparation

You can read about dataset preparation [here](scripts/README.md).


## Usage manual

See example of configs in [configs](configs)

```yaml
data_module:
  _target_: bmfm_targets.datasets.perturbx.PerturbxDataModule
  _partial_: true

  padding: longest
  num_workers: 8
  max_length: 4096
  sequence_order: sorted
  batch_size: 20
  collation_strategy: "sequence_labeling"
  transform_datasets: false
  log_normalize_transform: false
  limit_genes: tokenizer
  shuffle: True
  dataset_kwargs:
    iterate_over_all: True
    dataset_pars:
      - dataset_path: ${oc.env:PERTURBX_DATASET}/shuffled_replogle.h5ad
        index_dir: ${oc.env:PERTURBX_DATASET}/shuffled_replogle_train
        split: train
      - dataset_path: ${oc.env:PERTURBX_DATASET}/shuffled_replogle.h5ad
        index_dir: ${oc.env:PERTURBX_DATASET}/shuffled_replogle_dev
        split: dev
    transforms:
      - transform_name: NormalizeTotalTransform
        transform_args:
          exclude_highly_expressed: false
          max_fraction: 0.05
          target_sum: null # use median normalization
      - transform_name: LogTransform
        transform_args:
           add_one: True
```

The datamodule class is `bmfm_targets.datasets.perturbx.PerturbxDataModule`. All datasets are defined in `dataset_pars` of `dataset_kwars`. In the example, we use only a single dataset for degug proposes by providing list of splits. Each split consists of `h5ad` file, litdata index folder and in which split dataloader should use this dataset. Note that `transforms` are integrated into datasets and their parameters are given in  dataset level.

### Dataset weights

More complicated example is

```yaml
data_module:
  _target_: bmfm_targets.datasets.perturbx.PerturbxDataModule
  _partial_: true
  padding: longest
  num_workers: 0
  max_length: 4096
  sequence_order: sorted
  batch_size: 2
  collation_strategy: "sequence_labeling"
  transform_datasets: false
  log_normalize_transform: false
  limit_genes: tokenizer
  shuffle: True
  dataset_kwargs:
    iterate_over_all: True
    dataset_pars:
      - dataset_path: ${oc.env:PERTURBX_DATASET}/shuffled_replogle_full.h5ad
        index_dir: ${oc.env:PERTURBX_DATASET}/shuffled_replogle_full
        split: train
      - dataset_path: ${oc.env:PERTURBX_DATASET}/shuffled_h1.h5ad
        index_dir: ${oc.env:PERTURBX_DATASET}/shuffled_h1_train
        split: train
      - dataset_path: ${oc.env:PERTURBX_DATASET}/shuffled_h1.h5ad
        index_dir: ${oc.env:PERTURBX_DATASET}/shuffled_h1_dev
        split: dev
    transforms:
      - transform_name: NormalizeTotalTransform
        transform_args:
          exclude_highly_expressed: false
          max_fraction: 0.05
          target_sum: null # use median normalization
      - transform_name: LogTransform
        transform_args:
           add_one: True

```

Here, for train split we mix full replogle dataset and H1 train dataset from our internal split. H1 test split is used for validation.
Each dataset could have additional parameter `weight`.
```yaml
dataset_pars:
    - dataset_path: ${oc.env:PERTURBX_DATASET}/shuffled_replogle_full.h5ad
      index_dir: ${oc.env:PERTURBX_DATASET}/shuffled_replogle_full
      split: train
      weight: 0.1
```
This parameter is ignored if `iterate_over_all: True` and instead calculated by litdata from the size of the datasets. Our combined datasets are based on [CombinedStreamingDataset](https://github.com/Lightning-AI/litData) with stratified batching. Weights corresponds to the fraction of samples in the batch from a specific dataset. For instance, weight 0.1 means that 10% of the batch is taken from the dataset. The iterator over dataset stops as soon as one of the dataset does not have samples any more. That is why it is important to use `shuffle: True` to ensure that all data is used during training.


### gene_join_mode

The dataset supports `limit_genes`. In addition, genes are filtered by another important parameter,`gene_join_mode`. Its values can be

|value|filter|
|-|-|
|"all"| no additional filtering|
|"control_only"| Only genes in the control sample are taken in profiles of both control and perturbed samples|
|"alternate| "all" or "control_only" strategy is randomly selected for every sample. `alternate_prob` is used to define probability of "control_only", default probability is 0.5|

An example

```yaml
dataset_kwargs:
    iterate_over_all: True
    gene_join_mode: alternate
    alternate_prob: 0.5
```


### adding aggregate gene expression file

```yaml
dataset_kwargs:
  iterate_over_all: True
  aggregate_file_path: ${oc.env:H1_DATASET}/agg_h1.h5ad
```
