declare -a datasets=("tf" "coreprom" "covid" "splice"  "promoter_dnabert2" "mpra" )
#declare -a datasets=("coreprom")
declare -a label_column_names=("label" "label" "label" "label" "label" "mean_value")
#declare -a label_column_names=("label" "label" "label")

EST_TIME=$(TZ="America/New_York" date +"%Y%m%d_%H%M")
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
TOKENIZER="snp2vec"
BS=32
LEARNING_RATE=0.0001
MODEL_PE=128
MODEL_WD=0.01
MODEL="modernbert" # Makesure it is passed on config.yaml as MODEL
MODEL_NAME="modernbert_wo_lora"
CHKPT_NAME="chkpt_no"

CHKPT_REF=null
#"\'/proj/bmfm/users/hongyang/training_runs/ref_snp_rc_1kb_10kb_10x_modernbert_v3/backup_ckpt/epoch=7-step=174744-val_loss=4.34.ckpt\'"
#"\'/proj/bmfm/users/hongyang/training_runs/ref_rc_1kb_10kb_10x_modernbert_v3/backup_ckpt/epoch=17-step=131004-val_loss=4.40.ckpt\'"
#"\'/proj/bmfm/users/hongyang/training_runs/ref_snp_rc_1kb_10kb_10x_modernbert_v3/backup_ckpt/epoch=3-step=87372-val_loss=4.40.ckpt\'"
#"\'/proj/bmfm/users/hongyang/training_runs/snp_rc_1kb_10kb_10x_modernbert_v3/backup_ckpt/epoch=9-step=72810-val_loss=4.40.ckpt\'"
#"\'/proj/bmfm/users/hongyang/training_runs/ref_rc_1kb_10kb_10x_modernbert_v3/backup_ckpt/epoch=7-step=58224-val_loss=4.46.ckpt\'"
#"\'/proj/bmfm/users/sanjoy/benchmarking/ckpts/scbert/scbert_ref_rc_1kb_10kb_10x_change0.3/epoch=8-step=895320-val_loss=5.13.ckpt\'"
#"\'/proj/bmfm/users/hongyang/training_runs/ref_rc_1kb_10kb_10x_mb_1g/epoch=1-step=78612-val_loss=5.23.ckpt\'"
#CHKPT_MODERNBERT_REF="/proj/bmfm/users/hongyang/training_runs/ref_rc_1kb_10kb_10x_mb_1g/epoch=1-step=69877-val_loss=5.27.ckpt"
OUTPUT_DIR="/proj/bmfm/users/sanjoy/benchmarking/"
EXTRA_TAG="batch${BS}_lr${LEARNING_RATE}_pe${MODEL_PE}_wd${MODEL_WD}_val_chk0.1_batch_dump" # This can be used for saving benchmarking and also clearml logging

# project_name: "bmfm-targets/evaluate_dna/${model_name}_${CHKPT_NAME}${extra_tag}"
# default_root_dir: "${output_directory}/${model_name}_${CHKPT_NAME}${extra_tag}/${dataset_name}"

# PREFIX_CMD and SUFFIX_CMD enable determining how the jobs will be launched (if at all)
# Examples:
# set PREFIX_CMD to "echo " and the commands will be printed (check that the bash vars are correct or to dump to a file for future running)
# set PREFIX_CMD to "jbsub -q x86_6h -cores 8+1 -mem 50g" or similar to submit on CCC
# set PREFIX_CMD to a session-manager-ccc call with the command as a variable to be parsed
# set SUFFIX_CMD to "--cfg job --resolve" to have the bmfm-targets-scbert print the resolved yaml without running the code
PREFIX_CMD="bsub -M 30G -n 16 -W 6:00 -gpu num=1:mode=exclusive_process "
SUFFIX_CMD="" #  +trainer.lora_config=default" #"--cfg job --resolve"
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
                    "bash -c \"bmfm-targets-scbert -cd $SCRIPT_DIR -cn config \
                    batch_size=$BS \
                    tokenizer=$TOKENIZER \
                    data_module=$DATASET task=train model=$MODEL\
                    dataset_name=$DATASET_NAME fold="${fold}/${version}" label_column_name=$LABEL_COLUMN_NAME \
                    model_name=$MODEL_NAME \
                    model_pe=$MODEL_PE \
                    model_wd=$MODEL_WD \
                    checkpoint_path=$CHKPT_REF \
                    checkpoint_name=$CHKPT_NAME \
                    learning_rate=$LEARNING_RATE \
                    output_directory=$OUTPUT_DIR \
                    extra_tag=$EXTRA_TAG \
                    $SUFFIX_CMD\"" ;
                #$PREFIX_CMD bmfm-targets-scbert -cd $SCRIPT_DIR -cn config data_module=$DATASET dataset_name=$DATASET_WITH_FOLD fold=$fold label_column_name=$LABEL_COLUMN_NAME task=predict ~model track_clearml.task_name=${DATASET}_zero_shot $SUFFIX_CMD ;
            done
        done
    elif [ "$DATASET" == "mpra" ]; then
        for fold in "K562_original" "K562_biallelic_200"; do
            DATASET_NAME=${DATASET}_${fold}
            mkdir -p ../output_logs/${MODEL_NAME}_${CHKPT_NAME}/${DATASET_NAME}
            $PREFIX_CMD -o ../output_logs/${MODEL_NAME}_${CHKPT_NAME}/$DATASET_NAME/train$EST_TIME.out \
                -e ../output_logs/${MODEL_NAME}_${CHKPT_NAME}/$DATASET_NAME/train$EST_TIME.err \
                "bash -c \"bmfm-targets-scbert -cd $SCRIPT_DIR -cn config \
                label_columns=$DATASET \
                batch_size=$BS \
                tokenizer=$TOKENIZER \
                data_module=$DATASET  trainer=regression task=train model=$MODEL\
                max_finetuning_epochs=25 \
                dataset_name=${DATASET_NAME} fold=$fold label_column_name=$LABEL_COLUMN_NAME \
                model_name=$MODEL_NAME \
                model_pe=$MODEL_PE \
                model_wd=$MODEL_WD \
                checkpoint_path=$CHKPT_REF \
                checkpoint_name=$CHKPT_NAME \
                learning_rate=$LEARNING_RATE \
                output_directory=$OUTPUT_DIR \
                extra_tag=$EXTRA_TAG \
                max_finetuning_epochs=20 \
                $SUFFIX_CMD\"" ;
            #$PREFIX_CMD bmfm-targets-scbert -cd $SCRIPT_DIR -cn config data_module=$DATASET label_columns=$DATASET trainer=regression_drosophila_enhancer dataset_name=$DATASET label_column_name=$LABEL_COLUMN_NAME task=predict ~model track_clearml.task_name=${DATASET}_zero_shot $SUFFIX_CMD ;
        done
    elif [ "$DATASET" == "drosophila_enhancer" ]; then
        mkdir -p ../output_logs/${MODEL_NAME}_${CHKPT_NAME}/${DATASET_NAME}
        $PREFIX_CMD -o ../output_logs/${MODEL_NAME}_${CHKPT_NAME}/$DATASET_NAME/train$EST_TIME.out \
            -e ../output_logs/${MODEL_NAME}_${CHKPT_NAME}/$DATASET_NAME/train$EST_TIME.err \
            "bash -c \"bmfm-targets-scbert -cd $SCRIPT_DIR -cn config \
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
        #$PREFIX_CMD bmfm-targets-scbert -cd $SCRIPT_DIR -cn config data_module=$DATASET label_columns=$DATASET trainer=regression_drosophila_enhancer dataset_name=$DATASET label_column_name=$LABEL_COLUMN_NAME task=predict ~model track_clearml.task_name=${DATASET}_zero_shot $SUFFIX_CMD ;
    elif [[ "$DATASET" == "promoter_dnabert2" || "$DATASET" == "coreprom" || "$DATASET" == "splice"  ]]; then
        for version in  "snpified_v1" ; do
            for type in  "snp_genome"; do
                DATASET_NAME="${DATASET}_${version}_${type}"
                mkdir -p ../output_logs/${MODEL_NAME}_${CHKPT_NAME}/${DATASET_NAME}
                $PREFIX_CMD -o ../output_logs/${MODEL_NAME}_${CHKPT_NAME}/$DATASET_NAME/train$EST_TIME.out \
                    -e ../output_logs/${MODEL_NAME}_${CHKPT_NAME}/$DATASET_NAME/train$EST_TIME.err \
                    "bash -c \"bmfm-targets-scbert -cd $SCRIPT_DIR -cn config \
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
                    extra_tag=$EXTRA_TAG \
                    max_finetuning_epochs=1 \
                    $SUFFIX_CMD\"" ;
            done
        done

    else
        mkdir -p ../output_logs/${MODEL_NAME}_${CHKPT_NAME}/${DATASET_NAME}
        $PREFIX_CMD -o ../output_logs/${MODEL_NAME}_${CHKPT_NAME}/$DATASET_NAME/train$EST_TIME.out \
            -e ../output_logs/${MODEL_NAME}_${CHKPT_NAME}/$DATASET_NAME/train$EST_TIME.err \
            "bash -c \"bmfm-targets-scbert -cd $SCRIPT_DIR -cn config \
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
            $SUFFIX_CMD\"";
        #$PREFIX_CMD bmfm-targets-scbert -cd $SCRIPT_DIR -cn config data_module=$DATASET dataset_name=$DATASET label_column_name=$LABEL_COLUMN_NAME task=predict ~model track_clearml.task_name=${DATASET}_zero_shot $SUFFIX_CMD ;
    fi
done
