declare -a datasets=("covid_19" "heart_atlas" "immune_all_human" "lung_atlas" "mhsp" "pbmc_10k" "cell_lines" "dc" "human_pbmc" "immune_atlas" "mca" "pancrm" "multiple_sclerosis" "myeloid" "zheng68k" "pancreascross" "hbones")


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
PREFIX_CMD="bsub -J benchmark_job -R \"rusage[ngpus=1,cpu=8,mem=16GB]\" -gpu num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB -o $HOME/.lsf/cluster/%J.out -e $HOME/.lsf/cluster/%J.err"
SUFFIX_CMD="" #--cfg job --resolve"
for DATASET in "${datasets[@]}"; do
    $PREFIX_CMD bmfm-targets-scbert -cd $SCRIPT_DIR -cn config data_module=$DATASET task=train model=scbert $SUFFIX_CMD ;
done

for DATASET in "${datasets[@]}"; do
$PREFIX_CMD bmfm-targets-scbert -cd $SCRIPT_DIR -cn config data_module=$DATASET data_module.collation_strategy=language_modeling task=predict ~model ~fields track_clearml.task_name=${DATASET}_zero_shot $SUFFIX_CMD ;
done
