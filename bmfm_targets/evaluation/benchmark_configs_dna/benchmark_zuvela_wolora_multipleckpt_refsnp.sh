declare -a datasets=("tf" "coreprom" "covid" "splice" "promoter_dnabert2") # "mpra" "snv_TeWhey_split_by_pairs")
#declare -a datasets=("mpra")
declare -a label_column_names=("label" "label" "label" "label" "label" "mean_value")
#declare -a label_column_names=("mean_value")

EST_TIME=$(TZ="America/New_York" date +"%Y%m%d_%H%M")
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
TOKENIZER="snp2vec"
BS=32
LEARNING_RATE=0.0001
MODEL_PE=128
MODEL_WD=0.01
MODEL="modernbert" # Makesure it is passed on config.yaml as MODEL
MODEL_NAME="modernbert_wo_lora"
CHKPT_NAME="hic_3celllines_multipleckpts"

CHKPT_REFS=(
  "\'/proj/bmfm/users/hongyang/training_runs/ref_snp_hic_3celllines/backup_ckpt/epoch=0-step=19890-val_loss=4.73.ckpt\'"
  "\'/proj/bmfm/users/hongyang/training_runs/ref_snp_hic_3celllines/backup_ckpt/epoch=1-step=39780-val_loss=4.57.ckpt\'"
  "\'/proj/bmfm/users/hongyang/training_runs/ref_snp_hic_3celllines/backup_ckpt/epoch=2-step=59670-val_loss=4.51.ckpt\'"
  "\'/proj/bmfm/users/hongyang/training_runs/ref_snp_hic_3celllines/backup_ckpt/epoch=3-step=79560-val_loss=4.47.ckpt\'"
  "\'/proj/bmfm/users/hongyang/training_runs/ref_snp_hic_3celllines/backup_ckpt/epoch=4-step=99450-val_loss=4.45.ckpt\'"
  "\'/proj/bmfm/users/hongyang/training_runs/ref_snp_hic_3celllines/backup_ckpt/epoch=5-step=119340-val_loss=4.43.ckpt\'"
  "\'/proj/bmfm/users/hongyang/training_runs/ref_snp_hic_3celllines/backup_ckpt/epoch=6-step=139230-val_loss=4.42.ckpt\'"
  "\'/proj/bmfm/users/hongyang/training_runs/ref_snp_hic_3celllines/backup_ckpt/epoch=7-step=159120-val_loss=4.41.ckpt\'"
  "\'/proj/bmfm/users/hongyang/training_runs/ref_snp_hic_3celllines/backup_ckpt/epoch=8-step=179010-val_loss=4.40.ckpt\'"
  "\'/proj/bmfm/users/hongyang/training_runs/ref_snp_hic_3celllines/backup_ckpt/epoch=9-step=198900-val_loss=4.39.ckpt\'"
  "\'/proj/bmfm/users/hongyang/training_runs/ref_snp_hic_3celllines/backup_ckpt/epoch=10-step=218790-val_loss=4.39.ckpt\'"
  "\'/proj/bmfm/users/hongyang/training_runs/ref_snp_hic_3celllines/backup_ckpt/epoch=11-step=238680-val_loss=4.38.ckpt\'"
  "\'/proj/bmfm/users/hongyang/training_runs/ref_snp_hic_3celllines/backup_ckpt/epoch=12-step=258570-val_loss=4.37.ckpt\'"
  "\'/proj/bmfm/users/hongyang/training_runs/ref_snp_hic_3celllines/backup_ckpt/epoch=13-step=278460-val_loss=4.37.ckpt\'"
  "\'/proj/bmfm/users/hongyang/training_runs/ref_snp_hic_3celllines/backup_ckpt/epoch=14-step=298350-val_loss=4.37.ckpt\'"
)

# CHKPT_NAME="human_genome_multipleckpts"
# CHKPT_REFS=(
#     "\'/proj/bmfm/users/hongyang/training_runs/ref_snp_rc_1kb_10kb_10x/backup_ckpt/epoch=0-step=21843-val_loss=4.61.ckpt\'"
#     "\'/proj/bmfm/users/hongyang/training_runs/ref_snp_rc_1kb_10kb_10x/backup_ckpt/epoch=1-step=43686-val_loss=4.48.ckpt\'"
#     "\'/proj/bmfm/users/hongyang/training_runs/ref_snp_rc_1kb_10kb_10x/backup_ckpt/epoch=2-step=65529-val_loss=4.42.ckpt\'"
#     "\'/proj/bmfm/users/hongyang/training_runs/ref_snp_rc_1kb_10kb_10x/backup_ckpt/epoch=3-step=87372-val_loss=4.40.ckpt\'"
#     "\'/proj/bmfm/users/hongyang/training_runs/ref_snp_rc_1kb_10kb_10x/backup_ckpt/epoch=4-step=109215-val_loss=4.38.ckpt\'"
#     "\'/proj/bmfm/users/hongyang/training_runs/ref_snp_rc_1kb_10kb_10x/backup_ckpt/epoch=5-step=131058-val_loss=4.37.ckpt\'"
#     "\'/proj/bmfm/users/hongyang/training_runs/ref_snp_rc_1kb_10kb_10x/backup_ckpt/epoch=6-step=152901-val_loss=4.36.ckpt\'"
#     "\'/proj/bmfm/users/hongyang/training_runs/ref_snp_rc_1kb_10kb_10x/backup_ckpt/epoch=7-step=174744-val_loss=4.35.ckpt\'"
#     "\'/proj/bmfm/users/hongyang/training_runs/ref_snp_rc_1kb_10kb_10x/backup_ckpt/epoch=8-step=196587-val_loss=4.34.ckpt\'"
# )

OUTPUT_DIR="/proj/bmfm/users/sanjoy/benchmarking/${CHKPT_NAME}"
EXTRA_TAG="batch${BS}_lr${LEARNING_RATE}_pe${MODEL_PE}_wd${MODEL_WD}_batch_dump" # This can be used for saving benchmarking and also clearml logging

# project_name: "bmfm-targets/evaluate_dna/${model_name}_${CHKPT_NAME}${extra_tag}"
# default_root_dir: "${output_directory}/${model_name}_${CHKPT_NAME}${extra_tag}/${dataset_name}"

# PREFIX_CMD and SUFFIX_CMD enable determining how the jobs will be launched (if at all)
# Examples:
# set PREFIX_CMD to "echo " and the commands will be printed (check that the bash vars are correct or to dump to a file for future running)
# set PREFIX_CMD to "jbsub -q x86_6h -cores 8+1 -mem 50g" or similar to submit on CCC
# set PREFIX_CMD to a session-manager-ccc call with the command as a variable to be parsed
# set SUFFIX_CMD to "--cfg job --resolve" to have the bmfm-targets-run print the resolved yaml without running the code
PREFIX_CMD="bsub -M 50G -n 8 -W 6:00 -gpu num=1:mode=exclusive_process "
SUFFIX_CMD="" #  +trainer.lora_config=default" #"--cfg job --resolve"
for ind in "${!CHKPT_REFS[@]}"; do
    CHKPT_REF="${CHKPT_REFS[$ind]}"
    NEW_CHKPT_NAME="${CHKPT_NAME}_epoch${ind}"
    NEW_EXTRA_TAG="${EXTRA_TAG}/epoch${ind}"
    echo "index=$ind  value=$CHKPT_REF"
    for i in "${!datasets[@]}"; do
        DATASET=${datasets[i]}
        LABEL_COLUMN_NAME=${label_column_names[i]}
        echo $DATASET
        # Set Dataset_name to default dataset
        DATASET_NAME=$DATASET
        if [[ "$DATASET" == 'tf' || "$DATASET" == 'promoter' ]]; then
            for fold in "fold1" "fold2" "fold3" "fold4" "fold5"; do
                for version in "ref_genome" "snp_genome"; do
                    DATASET_NAME="${DATASET}_${fold}_${version}"
                    #rm -rf $OUTPUT_DIR/${MODEL_NAME}_${CHKPT_NAME}/${DATASET_NAME}
                    mkdir -p ../output_logs/${MODEL_NAME}_${CHKPT_NAME}/${DATASET_NAME}
                    $PREFIX_CMD -o ../output_logs/${MODEL_NAME}_${CHKPT_NAME}/$DATASET_NAME/trian$EST_TIME.out \
                        -e ../output_logs/${MODEL_NAME}_${CHKPT_NAME}/$DATASET_NAME/train$EST_TIME.err \
                        "bash -c \"bmfm-targets-run --config-path $SCRIPT_DIR -cn config \
                        batch_size=$BS \
                        tokenizer=$TOKENIZER \
                        data_module=$DATASET task=train model=$MODEL \
                        dataset_name=$DATASET_NAME fold="${fold}/${version}" label_column_name=$LABEL_COLUMN_NAME \
                        model_name=$MODEL_NAME \
                        model_pe=$MODEL_PE \
                        model_wd=$MODEL_WD \
                        checkpoint_path=$CHKPT_REF \
                        checkpoint_name=$CHKPT_NAME \
                        learning_rate=$LEARNING_RATE \
                        output_directory=$OUTPUT_DIR \
                        extra_tag=$EXTRA_TAG \
                        max_finetuning_epochs=5 \
                        $SUFFIX_CMD\"" ;
                    #$PREFIX_CMD bmfm-targets-run --config-path $SCRIPT_DIR -cn config data_module=$DATASET dataset_name=$DATASET_WITH_FOLD fold=$fold label_column_name=$LABEL_COLUMN_NAME task=predict ~model track_clearml.task_name=${DATASET}_zero_shot $SUFFIX_CMD ;
                done
            done
        elif [ "$DATASET" == "mpra" ]; then
            for fold in "K562_original_trimmed" "HepG2_original_trimmed" "WTC11_original_trimmed"; do
                DATASET_NAME=${DATASET}_${fold}
                mkdir -p ../output_logs/${MODEL_NAME}_${CHKPT_NAME}/${DATASET_NAME}
                $PREFIX_CMD -o ../output_logs/${MODEL_NAME}_${CHKPT_NAME}/$DATASET_NAME/train$EST_TIME.out \
                    -e ../output_logs/${MODEL_NAME}_${CHKPT_NAME}/$DATASET_NAME/train$EST_TIME.err \
                    "bash -c \"bmfm-targets-run --config-path $SCRIPT_DIR -cn config \
                    label_columns=$DATASET \
                    batch_size=$BS \
                    tokenizer=$TOKENIZER \
                    data_module=$DATASET  trainer=regression task=train model=$MODEL\
                    max_finetuning_epochs=30 \
                    dataset_name=${DATASET_NAME} fold=$fold label_column_name=$LABEL_COLUMN_NAME \
                    model_name=$MODEL_NAME \
                    model_pe=$MODEL_PE \
                    model_wd=$MODEL_WD \
                    checkpoint_path=$CHKPT_REF \
                    checkpoint_name=$CHKPT_NAME \
                    learning_rate=$LEARNING_RATE \
                    output_directory=$OUTPUT_DIR \
                    extra_tag=$EXTRA_TAG \
                    $SUFFIX_CMD\"" ;
                #$PREFIX_CMD bmfm-targets-run --config-path $SCRIPT_DIR -cn config data_module=$DATASET label_columns=$DATASET trainer=regression_drosophila_enhancer dataset_name=$DATASET label_column_name=$LABEL_COLUMN_NAME task=predict ~model track_clearml.task_name=${DATASET}_zero_shot $SUFFIX_CMD ;
            done
        elif [ "$DATASET" == "drosophila_enhancer" ]; then
            mkdir -p ../output_logs/${MODEL_NAME}_${CHKPT_NAME}/${DATASET_NAME}
            $PREFIX_CMD -o ../output_logs/${MODEL_NAME}_${CHKPT_NAME}/$DATASET_NAME/train$EST_TIME.out \
                -e ../output_logs/${MODEL_NAME}_${CHKPT_NAME}/$DATASET_NAME/train$EST_TIME.err \
                "bash -c \"bmfm-targets-run --config-path $SCRIPT_DIR -cn config \
                label_columns=$DATASET \
                batch_size=$BS \
                tokenizer=$TOKENIZER \
                data_module=$DATASET trainer=regression_drosophila_enhancer task=train model=$MODEL\
                dataset_name=$DATASET_NAME label_column_name=$LABEL_COLUMN_NAME \
                model_name=$MODEL_NAME \
                model_pe=$MODEL_PE \
                model_wd=$MODEL_WD \
                checkpoint_path=$CHKPT_REF \
                checkpoint_name=$CHKPT_NAME \
                learning_rate=$LEARNING_RATE \
                output_directory=$OUTPUT_DIR \
                extra_tag=$EXTRA_TAG \
                $SUFFIX_CMD\"";
            #$PREFIX_CMD bmfm-targets-run --config-path $SCRIPT_DIR -cn config data_module=$DATASET label_columns=$DATASET trainer=regression_drosophila_enhancer dataset_name=$DATASET label_column_name=$LABEL_COLUMN_NAME task=predict ~model track_clearml.task_name=${DATASET}_zero_shot $SUFFIX_CMD ;
        elif [[ "$DATASET" == "promoter_dnabert2" || "$DATASET" == "coreprom" || "$DATASET" == "splice"  ]]; then
            for version in "snpified_v3" ; do
                for type in "ref_genome" "snp_genome" "refsnp_genome"; do
                    DATASET_NAME="${DATASET}_${version}_${type}"
                    mkdir -p ../output_logs/${MODEL_NAME}_${NEW_CHKPT_NAME}/${DATASET_NAME}
                    $PREFIX_CMD -o ../output_logs/${MODEL_NAME}_${NEW_CHKPT_NAME}/$DATASET_NAME/train$EST_TIME.out \
                        -e ../output_logs/${MODEL_NAME}_${NEW_CHKPT_NAME}/$DATASET_NAME/train$EST_TIME.err \
                        "bash -c \"bmfm-targets-run --config-path $SCRIPT_DIR -cn config \
                        tokenizer=$TOKENIZER \
                        batch_size=$BS \
                        data_module=$DATASET task=train model=$MODEL\
                        dataset_name=${DATASET_NAME} fold="${version}/${type}" label_column_name=$LABEL_COLUMN_NAME \
                        model_name=$MODEL_NAME \
                        model_pe=$MODEL_PE \
                        model_wd=$MODEL_WD \
                        checkpoint_path=$CHKPT_REF \
                        checkpoint_name=$CHKPT_NAME \
                        learning_rate=$LEARNING_RATE \
                        output_directory=$OUTPUT_DIR \
                        extra_tag=$NEW_EXTRA_TAG \
                        max_finetuning_epochs=5 \
                        $SUFFIX_CMD\"" ;
                done
            done
        elif [ "$DATASET" == "snv_TeWhey_split_by_pairs" ]; then
            for fold in "K562" "HepG2"; do
                DATASET_NAME=${DATASET}_${fold}
                LABEL_COLUMN_NAME="${fold}_label"
                mkdir -p ../output_logs/${MODEL_NAME}_${CHKPT_NAME}/${DATASET_NAME}
                $PREFIX_CMD -o ../output_logs/${MODEL_NAME}_${CHKPT_NAME}/$DATASET_NAME/train$EST_TIME.out \
                    -e ../output_logs/${MODEL_NAME}_${CHKPT_NAME}/$DATASET_NAME/train$EST_TIME.err \
                    "bash -c \"bmfm-targets-run --config-path $SCRIPT_DIR -cn config \
                    label_columns=mpra \
                    batch_size=$BS \
                    tokenizer=$TOKENIZER \
                    data_module=$DATASET  trainer=regression task=train model=$MODEL\
                    max_finetuning_epochs=15 \
                    dataset_name=${DATASET_NAME} fold=$fold label_column_name=$LABEL_COLUMN_NAME \
                    model_name=$MODEL_NAME \
                    model_pe=$MODEL_PE \
                    model_wd=$MODEL_WD \
                    checkpoint_path=$CHKPT_REF \
                    checkpoint_name=$CHKPT_NAME \
                    learning_rate=$LEARNING_RATE \
                    output_directory=$OUTPUT_DIR \
                    extra_tag=$EXTRA_TAG \
                    $SUFFIX_CMD\"" ;
                #$PREFIX_CMD bmfm-targets-run --config-path $SCRIPT_DIR -cn config data_module=$DATASET label_columns=$DATASET trainer=regression_drosophila_enhancer dataset_name=$DATASET label_column_name=$LABEL_COLUMN_NAME task=predict ~model track_clearml.task_name=${DATASET}_zero_shot $SUFFIX_CMD ;
            done
        else
            mkdir -p ../output_logs/${MODEL_NAME}_${CHKPT_NAME}/${DATASET_NAME}
            $PREFIX_CMD -o ../output_logs/${MODEL_NAME}_${CHKPT_NAME}/$DATASET_NAME/train$EST_TIME.out \
                -e ../output_logs/${MODEL_NAME}_${CHKPT_NAME}/$DATASET_NAME/train$EST_TIME.err \
                "bash -c \"bmfm-targets-run --config-path $SCRIPT_DIR -cn config \
                tokenizer=$TOKENIZER \
                batch_size=$BS \
                data_module=$DATASET task=train model=$MODEL\
                dataset_name=${DATASET_NAME} label_column_name=$LABEL_COLUMN_NAME \
                model_name=$MODEL_NAME \
                model_pe=$MODEL_PE \
                model_wd=$MODEL_WD \
                checkpoint_path=$CHKPT_REF \
                checkpoint_name=$CHKPT_NAME \
                learning_rate=$LEARNING_RATE \
                output_directory=$OUTPUT_DIR \
                extra_tag=$EXTRA_TAG \
                max_finetuning_epochs=5 \
                $SUFFIX_CMD\"";
            #$PREFIX_CMD bmfm-targets-run --config-path $SCRIPT_DIR -cn config data_module=$DATASET dataset_name=$DATASET label_column_name=$LABEL_COLUMN_NAME task=predict ~model track_clearml.task_name=${DATASET}_zero_shot $SUFFIX_CMD ;
        fi
    done

done
