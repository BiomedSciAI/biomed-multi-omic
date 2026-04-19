#!/usr/bin/bash
set -euo pipefail

# =============================================================================
# benchmark_run.sh - scBERT Model Benchmark Evaluation Script
# =============================================================================
#
# DESCRIPTION:
#   Runs benchmark evaluations for scBERT models on various single-cell datasets.
#   Supports both zero-shot (ZS) prediction and fine-tuning (FT) workflows.
#   Jobs are submitted to an LSF cluster with GPU allocation.
#
# USAGE:
#   ./benchmark_run.sh [OPTIONS]
#
# OPTIONS:
#   --name=MODEL_NAME          Model name for identification (default: "benchmark")
#   --ckpt=CHECKPOINT_PATH     Path to model checkpoint
#   --rda=true|false           Enable RDA transform (default: false)
#   --pooling=STRING           Pooling method, e.g., 0 | 1 | [0,1] | [0,1,2]
#   --tag=RUN_TAG              Tag for ClearML project naming (default: "")
#   --do_zs=true|false         Run zero-shot evaluation (default: true)
#   --do_ft=true|false         Run fine-tuning (default: false)
#   --extra_args=STRING        Additional bmfm-targets-scbert arguments
#   --help                     Display help message
#
# EXAMPLES:
#   # Run zero-shot evaluation only
#   ./benchmark_run.sh --name=my_model --ckpt=/path/to/checkpoint
#
#   # Run both zero-shot and fine-tuning with RDA
#   ./benchmark_run.sh --name=my_model --ckpt=/path/to/checkpoint --rda=true --do_ft=true
#
#   # Run with custom pooling and tag
#   ./benchmark_run.sh --name=my_model --pooling=[0,1] --tag=custom_run
#
# NOTES:
#   - Results are tracked in ClearML under project: bmfm-targets/evaluate/{NAME}/{TIMESTAMP}_{TAG}_4096
#   - LSF job outputs are saved to $HOME/.lsf/cluster/
#   - Modify the 'datasets' array (line 90) to select which datasets to evaluate
#   - Adjust PREFIX_CMD (lines 112-120) to change cluster submission parameters
#
# =============================================================================

# -----------------------
# Defaults
# -----------------------
NAME="benchmark"
CKPT_PATH=""
IS_RDA="false"
POOLING=""
RUN_TAG=""
DO_ZS="true"
DO_FT="false"
EXTRA_ARGS=""

# -----------------------
# Parse long args
# -----------------------
for arg in "$@"; do
  case $arg in
    --name=*)        NAME="${arg#*=}" ;;
    --ckpt=*)        CKPT_PATH="${arg#*=}" ;;
    --rda=*)         IS_RDA="${arg#*=}" ;;
    --pooling=*)     POOLING="${arg#*=}" ;;
    --tag=*)         RUN_TAG="${arg#*=}" ;;
    --do_zs=*)       DO_ZS="${arg#*=}" ;;
    --do_ft=*)       DO_FT="${arg#*=}" ;;
    --extra_args=*)  EXTRA_ARGS="${arg#*=}" ;;
    --help)
      echo "Usage:"
      echo "  --name=MODEL_NAME"
      echo "  --ckpt=CHECKPOINT_PATH"
      echo "  --rda=true|false"
      echo "  --pooling=STRING eg 0 | 1 | [0,1] | [0,1,2]"
      echo "  --tag=RUN_TAG"
      echo "  --do_zs=true|false (default: true)"
      echo "  --do_ft=true|false (default: false)"
      echo "  --extra_args=STRING (additional bmfm-targets-scbert arguments)"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg"
      exit 1
      ;;
  esac
done


# Generate timestamp for project name
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")


# Normalize boolean values
case "$IS_RDA" in
  1|true|True|TRUE|yes) IS_RDA=true ;;
  *) IS_RDA=false ;;
esac

case "$DO_ZS" in
  1|true|True|TRUE|yes) DO_ZS=true ;;
  0|false|False|FALSE|no) DO_ZS=false ;;
  *) DO_ZS=true ;;
esac

case "$DO_FT" in
  1|true|True|TRUE|yes) DO_FT=true ;;
  0|false|False|FALSE|no) DO_FT=false ;;
  *) DO_FT=false ;;
esac

# use these 12 for zerohot
# declare -a datasets=("covid_19" "heart_atlas" "immune_all_human" "lung_atlas"  "pbmc_10k" "cell_lines" "dc" "human_pbmc" "immune_atlas" "pancrm" "multiple_sclerosis"  "myeloid") # "mce" "mhsp")
# "multiple_sclerosis" "myeloid" "zheng68k" "pancreascross" "hbones") # remove those which are not in the 12-zeroshot benchmarking datasets

# use these 2 for fine-tune
declare -a datasets=("multiple_sclerosis"  "myeloid") #

# mouse only
#declare -a datasets=("mca" "mhsp") #



SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# PREFIX_CMD and SUFFIX_CMD enable determining how the jobs will be launched (if at all)
# Examples:
# set PREFIX_CMD to "echo " and the commands will be printed (check that the bash vars are correct or to dump to a file for future running)
# set PREFIX_CMD to "jbsub -q x86_6h -cores 8+1 -mem 16g" or similar to submit on CCC
# set PREFIX_CMD to a session-manager-ccc call with the command as a variable to be parsed
# set SUFFIX_CMD to "--cfg job --resolve" to have the bmfm-targets-scbert print the resolved yaml without running the code
# PREFIX_CMD="jbsub -q x86_6h -cores 8+1 -mem 16g"
# run on v100
# PREFIX_CMD="bsub -J benchmark_job -R \"rusage[ngpus=1,cpu=8,mem=16GB]\" -gpu num=1:mode=exclusive_process:gmodel=TeslaV100_SXM2_32GB -o $HOME/.lsf/cluster/%J.out -e $HOME/.lsf/cluster/%J.err"
# run on a100

# just debug print
# PREFIX_CMD=echo


# zuvela
PREFIX_CMD=(
  bsub
  -o "$HOME/.lsf/cluster/%J.out"
  -e "$HOME/.lsf/cluster/%J.err"
  -n 8
  -R 'select[ngpus>=1] span[hosts=1]'
  -M 128G
  -gpu 'num=1:mode=exclusive_process'
)

# # ccc
# PREFIX_CMD=(
#   bsub
#   -J benchmark_job
#   -R 'rusage[ngpus=1,cpu=8,mem=16GB]'
#   -gpu 'num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB'
#   -o "$HOME/.lsf/cluster/%J.out"
#   -e "$HOME/.lsf/cluster/%J.err"
# )


MODEL_SPECIFIC_ARGS=()
if [ "$IS_RDA" = "true" ]; then
    MODEL_SPECIFIC_ARGS+=(+data_module.rda_transform=auto_align)
fi
if [ -n "$POOLING" ] && [ "$POOLING" != "0" ]; then
  MODEL_SPECIFIC_ARGS+=( "trainer.pooling_method=$POOLING" )
fi
if [ -n "$NAME" ]; then
  MODEL_SPECIFIC_ARGS+=( "checkpoint_name=$NAME" )
fi
if [ -n "$CKPT_PATH" ]; then
  MODEL_SPECIFIC_ARGS+=( "checkpoint_path=$CKPT_PATH" )
fi

SUFFIX_CMD="" #--cfg job --resolve"
[[ -n "$RUN_TAG" ]] && RUN_TAG="_${RUN_TAG}"
echo
echo "clearml project"
echo "bmfm-targets/evaluate/${NAME}/${TIMESTAMP}${RUN_TAG}"
echo $MODEL_SPECIFIC_ARGS
echo $EXTRA_ARGS
echo

# Fine-tuning
if [ "$DO_FT" = "true" ]; then
    echo "Running fine-tuning for all datasets..."
    for DATASET in "${datasets[@]}"; do
        #use this for tag in project name
        "${PREFIX_CMD[@]}" bmfm-targets-scbert -cd $SCRIPT_DIR -cn config data_module=$DATASET task=train track_clearml.task_name=${DATASET}_ft project_name_tag="${TIMESTAMP}${RUN_TAG}" "${MODEL_SPECIFIC_ARGS[@]}"  $EXTRA_ARGS $SUFFIX_CMD ;

        # use this for tag in task name
        #"${PREFIX_CMD[@]}" bmfm-targets-scbert -cd $SCRIPT_DIR -cn config data_module=$DATASET task=train track_clearml.task_name=${DATASET}_ft_${TIMESTAMP}${RUN_TAG}  "${MODEL_SPECIFIC_ARGS[@]}"  $EXTRA_ARGS $SUFFIX_CMD ;
    done
fi

# Zero-shot prediction section
if [ "$DO_ZS" = "true" ]; then
    echo "Running zero-shot prediction for all datasets..."
    for DATASET in "${datasets[@]}"; do
        "${PREFIX_CMD[@]}" bmfm-targets-scbert -cd $SCRIPT_DIR -cn config data_module=$DATASET task=predict ~model ~tokenizer ~trainer.losses ~fields track_clearml.task_name=${DATASET}_zero_shot ++task.output_predictions=False  project_name_tag="${TIMESTAMP}${RUN_TAG}" "${MODEL_SPECIFIC_ARGS[@]}" $EXTRA_ARGS $SUFFIX_CMD ;
    done
fi


# Exit successfully
exit 0



# example - v1 BERT checkpoints (110M parameters)
# bash benchmark_run.sh --ckpt="ibm-research/biomed.rna.bert.110m.wced.v1" --extra_args=" data_module.max_length=4096
# data_module.sequence_order=sorted" --tag="4096_sorted" --do_zs=true --do_ft=false

#bash benchmark_run.sh --ckpt="ibm-research/biomed.rna.bert.110m.wced.v1"
# --extra_args="task.0.freeze_layers=false max_finetuning_epochs=4 trainer.learning_rate=2e-4 data_module.max_length=1024  task.0.accumulate_grad_batches=8 data_module.batch_size=1 data_module.sequence_order=sorted model=scbert fields=genes_expressions_wced"
#--tag="freeze_false_epoch_4_lr2e4_sorted" --do_zs=false --do_ft=true

# example - v2 LLaMA checkpoints
# bash benchmark_run.sh --ckpt="ibm-research/biomed.rna.llama.47m.wced.multitask.v1" --extra_args=" data_module.max_length=4096
# data_module.sequence_order=sorted" --tag="llama_47m_4096_sorted" --do_zs=true --do_ft=false

# bash benchmark_run.sh --ckpt="ibm-research/biomed.rna.llama.32m.mlm.multitask.v1" --extra_args=" data_module.max_length=4096
# data_module.sequence_order=sorted" --tag="llama_32m_4096_sorted" --do_zs=true --do_ft=false
