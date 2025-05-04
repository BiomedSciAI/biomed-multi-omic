declare -a datasets=("covid_19" "heart_atlas" "immune_all_human" "lung_atlas" "mhsp" "pbmc_10k" "perirhinal_cortex" "cell_lines" "dc" "human_pbmc" "immune_atlas" "mca" "pancrm" "pbmc_3k_68k")

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# PREFIX_CMD and SUFFIX_CMD enable determining how the jobs will be launched (if at all)
# Examples:
# set PREFIX_CMD to "echo " and the commands will be printed (check that the bash vars are correct or to dump to a file for future running)
# set PREFIX_CMD to "jbsub -q x86_6h -cores 8+1 -mem 16g" or similar to submit on CCC
# set PREFIX_CMD to a session-manager-ccc call with the command as a variable to be parsed
# set SUFFIX_CMD to "--cfg job --resolve" to have the bmfm-targets-scbert print the resolved yaml without running the code
PREFIX_CMD="jbsub -q x86_6h -cores 8+1 -mem 16g"
SUFFIX_CMD="" #--cfg job --resolve"
for DATASET in "${datasets[@]}"; do
    $PREFIX_CMD bmfm-targets-scbert -cd $SCRIPT_DIR -cn config data_module=$DATASET dataset_name=$DATASET task=train model=scbert track_clearml.task_name=${DATASET}_ft $SUFFIX_CMD ;
done

for DATASET in "${datasets[@]}"; do
    $PREFIX_CMD bmfm-targets-scbert -cd $SCRIPT_DIR -cn config data_module=$DATASET dataset_name=$DATASET data_module.collation_strategy=multitask task=predict ~model  trainer={} ~fields ~tokenizer track_clearml.task_name=${DATASET}_zero_shot $SUFFIX_CMD ;
done
