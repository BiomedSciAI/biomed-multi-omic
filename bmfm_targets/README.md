# bmfm_targets

Package to pretrain foundational AI models using single-cell data or genomic sequences

## Source code and issue tracker

Available on Github, [biomed-multi-omics](https://github.com/BiomedSciAI/biomed-multi-omic).
Please report bugs, issues and feature extensions there.

## Training

After installing the package, open the mlm_train_config.yaml file in the run directory, edit the paths to the data to direct to your data and run

```bash
bmfm-targets-run -cn mlm_train_config -cd run
```

For finetuning you will probably want to select a checkpoint and provide it to the `model.checkpoint` field of the
config and then set `mlm` to false in the data module. These parameters could be set in the command line in teh following manner:

```bash
bmfm-targets-run -cn prediction_train_config data_module.mlm=false -cd run
```

## Configuration

The configuration is structured into different sections to define various aspects of the experiment. These sections are depicted in the folder structure under the run directory, and in each of these you'll find default yamls for common cases.

### Seed

The seed value is used to ensure reproducibility across runs of the experiment.

```yaml
seed:
  seed_value: 1234
```

### Fields

The settings shared between data module and module. In order for a loss to be valid for a field in language_modeling/multitask mode, the field must have `is_masked=True`. For sequence labeling, the field with the loss should have `is_input=False`.

```yaml
fields:
  - _target_: bmfm_targets.config.FieldInfo
    field_name: "genes"
    pretrained_embedding: null
    is_masked: false
  - _target_: bmfm_targets.config.FieldInfo
    field_name: "expressions"
    pretrained_embedding: null
    is_masked: true
```

### Data Module

The data module settings selects which data to use and also what kind of task will be executed. The dataset can refer to one of the packaged datasets or a generic h5ad file based dataset.

By selecting a `collation_strategy`, the same dataset can be used for different tasks.
The recommended option is `"multitask"` which supports masked language modeling and sequence classification/regression simoultaniously.
To perform just MLM `"language_modeling"` can be used and `"sequence_classification"` for just sequence classification.
In "multitask" mode, any combination of field and label_column losses is supported.

The `"sequence_labeling"` option is only supported for datasets with paired "control" and perturbed cells.

Full documentation can be found in [data_module.py](./training/data_module.py)

#### Package datasets

```yaml
data_module:
  _target_: bmfm_targets.datasets.zheng68k.Zheng68kDataModule
  _partial_: true
  num_workers: 8
  collation_strategy: "multitask"
  batch_size: 20
  max_length: 512
  pad_to_multiple_of: 16
  change_ratio: 0.15
  mask_ratio: 0.5
  switch_ratio: 0.5
  limit_dataset_samples: null
  shuffle: true
  sequence_order: "random"
  mlm: True
  data_dir: ${oc.env:BMFM_TARGETS_ZHENG68K_DATA}
```


#### Panglao data module
The Panglao data module behaves differently than the others because it is a dataset of datasets and has transforms defined within the dataset object:

```yaml
data_module:
  _target_: bmfm_targets.datasets.panglaodb.PanglaoDBDataModule
  _partial_: true
  num_workers: 8
  mlm: true
  collation_strategy: "language_modeling"
  batch_size: 2
  transform_datasets: false
  dataset_kwargs:
    data_dir: ${oc.env:BMFM_TARGETS_PANGLAO_DATA}
    data_info_path: ${oc.env:BMFM_TARGETS_PANGLAO_METADATA}
    filter_query: 'Species == "Homo sapiens"'
    num_workers: 8
    convert_rdata_to_h5ad: false
```

#### Dataset without dedicated data module

The package has many datasets that have been processed and have default settings.
If you want to use a new dataset you can either add it in a PR to the package or access
it directly as an h5ad file:

```yaml
data_module:
  _target_: bmfm_targets.training.data_module.DataModule
  _partial_: true
  num_workers: 8
  collation_strategy: "sequence_classification"
  batch_size: 2
  transform_datasets: true
  num_workers: 8
  transform_kwargs:
    source_h5ad_file_name: /path/to/raw_data.h5ad # you must supply this
    processed_data_source: /path/to/processed.h5ad # will be created if transform_datasets=True
    stratifying_label: "celltype"
  dataset_kwargs:
    processed_data_source: /path/to/processed.h5ad # must be identical to transform_kwargs.processed_data_source
    label_columns: ["celltype"]
    label_dict_path: /path/to/label_dict.json #will be created if absent, should be persistent for a given dataset
```

After running once with `transform_datasets=True`, future runs can run with `transform_datasets=False` and omit `transform_kwargs`.

### Model

Configuration options for the model.
Most of the parameters are standard HuggingFace Transformers settings.
Note that passing a checkpoint in the model config is necessary if beginning a training job based on a pretrained MLM checkpoint, or otherwise constructing a new model on top of another model. For test or predict tasks, this is not necessary or desirable.

```yaml
model:
  _target_: bmfm_targets.config.SCBertConfig
  _partial_: true
  num_hidden_layers: 12
  num_attention_heads: 12
  intermediate_size: 3072
  hidden_act: "gelu"
  hidden_size: 768
  hidden_dropout_prob: 0.1
  attention_probs_dropout_prob: 0.1
  initializer_range: 0.02
  layer_norm_eps: 1e-12
  pad_token_id: 0
  use_cache: true
  classifier_dropout: 0.1
  attention: torch
  checkpoint: null
```

### Trainer
Multiple losses can be used simultaneously with different weights. If there are masked fields in the `fields` section, or if the task is `sequence_labeling` then field losses (identified by `field_name`) can be used.
If the dataset has labels and the `label_columns` section is set, then `label_column` losses can be used.

For futher details see [training_config.py](./config/training_config.py) and [loss_handling.py](./training/metrics/loss_handling.py)

```yaml
trainer:
  _target_: bmfm_targets.config.TrainerConfig
  warmup_steps: 1000
  weight_decay: 0.01
  losses:
    - label_column_name: celltype
      name: focal
    - field_name: expressions
      name: mse
      ignore_zero: true
      # link_function: exp
    - field_name: expressions
      name: is_zero_bce
    - field_name: genes
      name: cross_entropy
```

### Task

Task-specific parameters for training the model:

```yaml
task:
  _target_: bmfm_targets.config.TrainingTaskConfig
  default_root_dir: ${oc.env:BMFM_TARGETS_TRAINING_DIR}
  max_epochs: 20
  max_steps: 1000000
  precision: "16-mixed"
  val_check_interval: 0.1
  accelerator: "gpu"
  accumulate_grad_batches: 4

```

Task-specific parameters for testing the model:

```yaml
task:
  _target_: bmfm_targets.config.TestTaskConfig
  default_root_dir: ${oc.env:BMFM_TARGETS_TRAINING_DIR}
  precision: "16-mixed"
  accelerator: "gpu"
  tf32_mode: "medium" # permitted nonNone values of tf32_mode:  "highest","high","medium"
  checkpoint: path/to/trained/model/checkpoint.ckpt
  num_bootstrap_runs: 0
  ci_method: binomial

```

During testing, three types of CI for the evaluation metrics are available thorugh the ci_method and num_bootstrap_runs arguments.
    - bootstrap_quantiles: takes the 2.5% and 97.5% quantiles of the bootstrap sample.
    - bootstrap_t_interval: based on the bootstrap sample distribution around the sample's mean.
        If len(data)=1 returns None values for the CI's bounds.
    - binomial: CI based on the binomial distribution for proportion metrics. Does not require repeated samplings from the data, thus num_bootstrap_runs should be set to 0. In case num_bootstrap_runs >=1
        The binomial CI uses the metric's average value across the bootstrap samples as the estimate.
