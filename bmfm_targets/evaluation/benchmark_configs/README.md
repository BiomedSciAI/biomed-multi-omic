# Automated benchmarking of pretrained model ckpts

This directory contains yamls for datasets that are used for benchmarking, and scripts for running them.

## Installation

Ensure that you have installed the package before running the commands. It is assumed that the benchmarks will be run on the cluster.

## Usage

To run the benchmark, use:

```bash
#if running from repo root
bash bmfm_targets/evaluation/benchmark_configs/benchmark_run.sh
```

## Instructions

When a new checkpoint is obtained, modify the `bmfm_targets/evaluation/benchmark_configs/config.yaml` fields:
- `checkpoint_path` path to ckpt file eg `/dccstor/bmfm-targets/models/omics/transcriptome/scRNA/pretrain/bmfm.targets.slate.bert.110m.scRNA.RDA.v1/last-v3.ckpt`
- `checkpoint_name`: name that will be used on clearml and on the file system for artifacts created, eg `rda_v1`
- `output_directory`: where all the artifacts will be created. Can be reused for many checkpoints or shared by users. New subdirectories will be created for each new `checkpoint_name` Warning: If running with the same checkpoint name the artifacts will be overwritten. eg `/dccstor/bmfm-targets1/users/dmichael/benchmarking/`

In `benchmark_run.sh` choose a way to launch your job by modifying the `PREFIX_CMD` amd `SUFFIX_CMD` commands. When you are ready simply run `bash benchmark_run.sh` and the benchmarking tasks will be launched.

### Key Configuration Details

- The benchmarks run sequence classification (train and test) and embedding generation, using the main entry point for the scRNA foundation model, `bmfm-targets-scbert`.
- For the predict task, we are assuming the model was trained with MLM and therefore require `data_module.collation_strategy=language_modeling`. We also remove the model settings `~model` so that everything is loaded from the ckpt.
- Tasks will be created in ClearML `bmfm-targets/evaluation/{checkpoint_name}/${DATASET}ft` and ` bmfm-targets/evaluation/{checkpoint_name}/${DATASET}zero_shot`
- If a job fails, users should check the ClearML dashboard for logs and debugging information.
- Checkpoints can be overwritten; but note that the test runs that require the ckpts are run in the same session so the user does not need to keep track of them.

## Managing Datasets

Datasets are manually listed in `benchmark_run.sh`:

```bash
declare -a datasets=(...)
```
Users need to modify this list manually to include new datasets.

### Dataset settings

The settings are constructed hierarchically and set in [hydra's override grammar](https://hydra.cc/docs/advanced/override_grammar/basic/). TLDR you can override whatever you want in the final `config.yaml` or by modifying the overrides in the `benchmark_run` script.

The basic settings are in
[base_seq_cls.yaml](bmfm_targets/evaluation/benchmark_configs/data_module/base_seq_cls.yaml)
which are updated/overwritten by the [sceval_default](bmfm_targets/evaluation/benchmark_configs/data_module/sceval_default.yaml) for the sceval datasets, and then updated per dataset in files such as [immune_atlas](/Users/dmichael/bmfm-targets/bmfm_targets/evaluation/benchmark_configs/data_module/immune_atlas.yaml).

If you need to change something that affects all of the datasets it should go either in `base_seq_cls` if it should be permanent, `sceval_default` if it affects all the sceval datasets and most importantly, in a dataset-specific file if it affects that dataset only. That is where things like dataset specific transforms and splits should go.

If the change is for a particular run you could simply override the default yaml fields in the `config.yaml` by adding something like

```yaml
data_module:
  max_length: 4096
  sequence_order: sorted
```

and it will affect all of the datasets that you run. It is also possible to supply this via the script but probably editing the config will be easier.

## Modifications for models and datasets

Some checkpoints require continuous values, some discrete, sometimes you will want to test with longer sequence lengths etc etc. All of these settings can be modified directly in `config.yaml`. Simply add a `data_module` section with the shared settings you want applied to all the datasets and they will be applied. If individual datasets require special data processing like different transforms, those should be modified in the dataset's named yaml.


### Hardware Requirements

The benchmark has been tested with the prefix_cmd set to:

```bash
jbsub -q x86_6h -cores 8+1 -mem 16g
```

This setup assumes GPU availability. Users may need to modify `prefix_cmd` to specify a GPU queue if necessary.

### Resuming Training

Training cannot be resumed directly from `benchmark_run.sh`, but `session-manager-ccc` can be used via `prefix_cmd` `suffix_cmd` to manage sessions effectively. This will require a little work, creating a custom session manager config that takes the command to run as an override, and may ultimately require a separate version of the script.
For now, we will use


### Expected Runtime

- Prediction runs are fast, 5-10 min.
- Training runs can take several hours, depending on how many epochs are requested. The number of epochs used for fine-tuning is set in `bmfm_targets/evaluation/benchmark_configs/task/train.yaml` where it is interpolated from the `config.yaml` file field `max_finetuning_epochs`.
