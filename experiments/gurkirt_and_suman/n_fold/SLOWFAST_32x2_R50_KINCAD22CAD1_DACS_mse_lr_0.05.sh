#!/bin/bash
####################################################################################################
###########  Please check lines between #####, they need to be edited before experiment  ###########
####################################################################################################

###################################################

module load gcc/8.2.0 python_gpu/3.9.9 libjpeg-turbo eth_proxy; unset PYTHONPATH
source ~/envirs/slowfast110/bin/activate


## copy cad1
source experiments/gurkirt_and_suman/data_copy/movCAD1.sh
## copy kinetics+cad2
source experiments/gurkirt_and_suman/data_copy/movKIN8CAD2.sh

###################################################

TRAIN_THRESH=0.8
EVAL_THRESH=0.8
# Training
echo "#################### Training on KIN+CAD2 #####################"


###################################################

COUT_DIR=/cluster/work/cvl/susaha/experiments/experiments_yifan/n_fold_kincad22cad1_daaim_mse_lr0.05/

###################################################
init_method="tcp://localhost:8013"

CFG_FILE=configs/Gurkirt_and_Suman/KINCAD22CAD1/SLOWFAST_32x2_R50_DA_DACS.yaml
python tools/run_net.py --cfg ${CFG_FILE} \
        --init_method ${init_method} \
        AVA.DETECTION_SCORE_THRESH ${TRAIN_THRESH} \
        OUTPUT_DIR ${COUT_DIR} \
        AVA.FRAME_DIR ${TARGET_DIR} \
        AUX.FRAME_DIR ${TARGET_DIR} \
        TRAIN.BATCH_SIZE 24 \
        DATA_LOADER.NUM_WORKERS 2 \
        SOLVER.BASE_LR 0.05 \
        SOLVER.WARMUP_START_LR 0.005 \
        SOLVER.COSINE_END_LR 0.0005 \
        SOLVER.WARMUP_EPOCHS 2.0 \
        SOLVER.MAX_EPOCH 10 \
        DACS.AUGMENTATION_ENABLE True \
        DACS.PSEUDO_LABEL_ENABLE True \
        DACS.RESIZE_ENABLE True \
        DACS.CONSISTENCY_LOSS 'mse_weighted' \
        DACS.PSEUDO_TARGETS 'logits' \
        DACS.CONSISTENCY_LOSS_WEIGHT 'threshold_uniform' \
        DACS.THRESHOLDS '[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]' \
        DACS.THRESHOLD 0.5 \
        AUX.FRAME_LIST_DIR "/cluster/work/cvl/susaha/dataset/action-dataset/datasets_yifan/cad1/frame_lists/" \
        AUX.ANNOTATION_DIR "/cluster/work/cvl/susaha/dataset/action-dataset/datasets_yifan/cad1/annotations/" \
        AUX.LABEL_MAP_FILE "cad1_action_list.pbtxt" \
        AUX.EXCLUSION_FILE "cad1_val_excluded_timestamps.csv" \
        AUX.TRAIN_LISTS ["train_new.csv"] \
        AUX.TEST_LISTS ["val_new.csv"] \
        AUX.TRAIN_GT_BOX_LISTS ["cad1_train_new.csv"] \
        AUX.GROUNDTRUTH_FILE "cad1_val_new.csv"  \
        AUX.TEST_PREDICT_BOX_LISTS ["cad1_val_new.csv"] \

now=$(date +"%T")
echo "time after DA training to CAD1 with Kinetics+CAD2 Source: $now"
echo ${COUT_DIR}

# Evaluation
echo "#################### Evaluation on CAD1 #####################"
CFG_FILE=configs/Gurkirt_and_Suman/CAD1/SLOWFAST_32x2_R50.yaml
python tools/run_net.py --cfg ${CFG_FILE} \
    --init_method ${init_method} \
    TRAIN.ENABLE False \
    TENSORBOARD.ENABLE False \
    AVA.DETECTION_SCORE_THRESH ${EVAL_THRESH} \
    AUX.FRAME_DIR ${TARGET_DIR} \
    OUTPUT_DIR ${COUT_DIR} \
    AVA.FRAME_DIR ${TARGET_DIR}

now=$(date +"%T")
echo "time after testing DA to CAD1 Supervised Kinetics+CAD2 on CAD1: $now"
echo ${COUT_DIR}

# rsync -rav eu://cluster/work/cvl/susaha/experiments/experiments_yifan/ experiments_yifan_euler/