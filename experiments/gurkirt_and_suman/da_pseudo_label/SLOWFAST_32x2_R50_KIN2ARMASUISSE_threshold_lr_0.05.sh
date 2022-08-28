#!/bin/bash
####################################################################################################
###########  Please check lines between #####, they need to be edited before experiment  ###########
####################################################################################################

###################################################

module load gcc/8.2.0 python_gpu/3.9.9 libjpeg-turbo eth_proxy; unset PYTHONPATH
source ~/envirs/slowfast110/bin/activate


## copy armasuisse 3
source experiments/gurkirt_and_suman/data_copy/movAS3.sh
## copy kinetics 3
source experiments/gurkirt_and_suman/data_copy/movKIN3.sh

###################################################

TRAIN_THRESH=0.8
EVAL_THRESH=0.8
# Training
echo "#################### Training on KIN #####################"


###################################################

COUT_DIR=/cluster/work/cvl/susaha/experiments/experiments_yifan/kin2as_pseudolabel_threshold_lr0.05/

###################################################
init_method="tcp://localhost:7416"

CFG_FILE=configs/Gurkirt_and_Suman/KIN2ARMASUISSE/SLOWFAST_32x2_R50_DA_DACS.yaml
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
        SOLVER.WARMUP_EPOCHS 1.0 \
        SOLVER.MAX_EPOCH 6 \
        DACS.AUGMENTATION_ENABLE False \
        DACS.PSEUDO_LABEL_ENABLE True \
        DACS.CONSISTENCY_LOSS 'ce_weighted' \
        DACS.PSEUDO_TARGETS 'binary' \
        DACS.CONSISTENCY_LOSS_WEIGHT 'ce_threshold' \
        DACS.THRESHOLDS '[0.9, 0.9, 0.9]' \
        DACS.THRESHOLD 0.9 \

now=$(date +"%T")
echo "time after DA training to AS with Kinetics Source: $now"
echo ${COUT_DIR}

# Evaluation
echo "#################### Evaluation on AS #####################"
CFG_FILE=configs/Gurkirt_and_Suman/ARMASUISSE/SLOWFAST_32x2_R50.yaml
python tools/run_net.py --cfg ${CFG_FILE} \
    --init_method ${init_method} \
    TRAIN.ENABLE False \
    TENSORBOARD.ENABLE False \
    AVA.DETECTION_SCORE_THRESH ${EVAL_THRESH} \
    AUX.FRAME_DIR ${TARGET_DIR} \
    OUTPUT_DIR ${COUT_DIR} \
    AVA.FRAME_DIR ${TARGET_DIR}

now=$(date +"%T")
echo "time after testing DA to AS Supervised Kinetics on AS: $now"
echo ${COUT_DIR}

# rsync -rav eu://cluster/work/cvl/susaha/experiments/experiments_yifan/ experiments_yifan_euler/