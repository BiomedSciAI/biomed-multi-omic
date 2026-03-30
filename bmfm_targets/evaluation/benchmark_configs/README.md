# Automated benchmarking of pretrained model ckpts

This directory contains yamls for datasets that are used for benchmarking, and scripts for running them.

## Installation

Ensure that you have installed the package before running the commands. It is assumed that the benchmarks will be run on the cluster.

## Usage

### Single Checkpoint Run

Run benchmarks for a single checkpoint with command-line arguments:

```bash
# Basic usage
bash bmfm_targets/evaluation/benchmark_configs/benchmark_run.sh \
  --name=my_model \
  --ckpt=/path/to/checkpoint.ckpt

# With options
bash bmfm_targets/evaluation/benchmark_configs/benchmark_run.sh \
  --name=my_model \
  --ckpt=/path/to/checkpoint.ckpt \
  --rda=true \
  --pooling=[0,1] \
  --tag=custom_run \
  --do_zs=true \
  --do_ft=false \
  --extra_args="data_module.max_length=4096"
```

**Options:**
- `--name`: Model identifier for ClearML tracking
- `--ckpt`: Path to checkpoint file
- `--rda`: Enable RDA transform (true/false, default: false)
- `--pooling`: Pooling method (e.g., 0, 1, [0,1], [0,1,2])
- `--tag`: Tag for ClearML project naming
- `--do_zs`: Run zero-shot evaluation (default: true)
- `--do_ft`: Run fine-tuning (default: false)
- `--extra_args`: Additional bmfm-targets-scbert arguments

### Multiple Checkpoints from CSV

Automated workflow for managing and benchmarking multiple checkpoints:

1. **Update checkpoint tracking**: Scan directories and update `ckpts.csv`:
   ```bash
   python bmfm_targets/evaluation/benchmark_configs/update_ckpts_from_runs.py [OPTIONS]
   ```

   **Options:**
   - `--dry-run`: Preview changes without updating CSV
   - `--preserve-benchmarking-flag`: Keep `run_benchmarking_on_updates` flag for unchanged checkpoints
   - `--csv-path FILE`: Path to ckpts.csv (default: ./ckpts.csv)
   - `--runs-dir DIR [DIR ...]`: Checkpoint directories to scan (can specify multiple)

   This discovers new/updated checkpoints by comparing versions and timestamps, extracts metadata from `llama_train.yaml` configs, and sets `run_benchmarking_on_updates=1` for changed checkpoints.

2. **Run benchmarks**: Process checkpoints marked for benchmarking:
   ```bash
   bash bmfm_targets/evaluation/benchmark_configs/run_benchmarks_from_csv.sh
   ```

   This iterates through `ckpts.csv` and runs benchmarks for entries where `run_benchmarking_on_updates=1`. Provides interactive confirmation ([y]es/[n]o/[a]ll/[q]uit) before running each benchmark.

**CSV Format (`ckpts.csv`):**
- `name`: Checkpoint identifier
- `folder`: Checkpoint folder path
- `ckpt_path`: Relative path to checkpoint file (e.g., data/last-v2.ckpt)
- `ckpt_timestamp`: File modification timestamp
- `config_path`: Path to training config (llama_train.yaml)
- `hidden_size`: Model hidden dimension (384/768)
- `max_length`: Maximum sequence length (1024/4096/8192)
- `has_decode_from_1`: Pooling method flag (0/1)
- `is_rda`: RDA transform flag (0/1)
- `is_logn`: Log normalization flag (0/1)
- `for_benchmarking`: General benchmarking flag (0/1)
- `run_benchmarking_on_updates`: Auto-set to 1 when checkpoint updates (0/1)
- `comments`: Optional notes

The workflow automatically detects checkpoint updates by comparing version numbers (last.ckpt < last-v1.ckpt < last-v2.ckpt) and timestamps. See `update_ckpts_from_runs.py` docstring for detailed logic.

## Instructions

### Single Checkpoint Configuration

When a new checkpoint is obtained, modify the `bmfm_targets/evaluation/benchmark_configs/config.yaml` fields:
- `checkpoint_path` path to ckpt file eg `/dccstor/bmfm-targets/models/omics/transcriptome/scRNA/pretrain/bmfm.targets.slate.bert.110m.scRNA.RDA.v1/last-v3.ckpt`
- `checkpoint_name`: name that will be used on clearml and on the file system for artifacts created, eg `rda_v1`
- `output_directory`: where all the artifacts will be created. Can be reused for many checkpoints or shared by users. New subdirectories will be created for each new `checkpoint_name` Warning: If running with the same checkpoint name the artifacts will be overwritten. eg `/dccstor/bmfm-targets1/users/dmichael/benchmarking/`

In `benchmark_run.sh` choose a way to launch your job by modifying the `PREFIX_CMD` amd `SUFFIX_CMD` commands. When you are ready simply run `bash benchmark_run.sh` and the benchmarking tasks will be launched.

### Key Configuration Details

- The benchmarks run sequence classification (train and test) and embedding generation, using the main entry point for the scRNA foundation model, `bmfm-targets-scbert`.
- For the predict task, model settings are removed (`~model`) so everything is loaded from the checkpoint.
- Tasks will be created in ClearML: `bmfm-targets/evaluation/{checkpoint_name}/{TIMESTAMP}_{TAG}/{DATASET}_ft` and `bmfm-targets/evaluation/{checkpoint_name}/{TIMESTAMP}_{TAG}/{DATASET}_zero_shot`
- If a job fails, check the ClearML dashboard for logs and debugging information.
- Checkpoints can be overwritten; test runs that require checkpoints are run in the same session so users don't need to track them separately.

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
