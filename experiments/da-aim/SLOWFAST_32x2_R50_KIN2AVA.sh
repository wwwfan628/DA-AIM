#!/bin/bash
####################################################################################################
########## Please comment this block if experiments are not executed on Euler or Leonhard ##########
module load gcc/8.2.0 python_gpu/3.9.9 libjpeg-turbo eth_proxy; unset PYTHONPATH                   #
source ~/envirs/slowfast110/bin/activate                                                           #
## copy ava 6                                                                                      #
source experiments/data_copy/movAVA6.sh                                                            #
## copy kinetics 6                                                                                 #
source experiments/data_copy/movKIN6.sh                                                            #
####################################################################################################

TRAIN_THRESH=0.8
EVAL_THRESH=0.8
COUT_DIR=/cluster/work/cvl/susaha/experiments/experiments_yifan/kin2ava_daaim/
init_method="tcp://localhost:8885"

# Training
echo "#################### Training on KIN #####################"
CFG_FILE=configs/DA-AIM/KIN2AVA/SLOWFAST_32x2_R50_DA_AIM.yaml
python tools/run_net.py --cfg ${CFG_FILE} \
        --init_method ${init_method} \
        AVA.DETECTION_SCORE_THRESH ${TRAIN_THRESH} \
        OUTPUT_DIR ${COUT_DIR} \
        AVA.FRAME_DIR ${TARGET_DIR} \
        AVA.FRAME_LIST_DIR "/cluster/work/cvl/susaha/dataset/action-dataset/datasets_yifan/kinetics_6_5000_all/frame_lists/" \
        AVA.ANNOTATION_DIR "/cluster/work/cvl/susaha/dataset/action-dataset/datasets_yifan/kinetics_6_5000_all/annotations/" \
        AUX.FRAME_DIR ${TARGET_DIR} \
        AUX.FRAME_LIST_DIR "/cluster/work/cvl/susaha/dataset/action-dataset/datasets_yifan/ava_6_5000_all/frame_lists/" \
        AUX.ANNOTATION_DIR "/cluster/work/cvl/susaha/dataset/action-dataset/datasets_yifan/ava_6_5000_all/annotations/" \
        TRAIN.BATCH_SIZE 24 \
        TRAIN.CHECKPOINT_FILE_PATH "cluster/work/cvl/susaha/dataset/action-dataset/pretrained_model/SLOWFAST_8x8_R50.pyth" \
        DATA_LOADER.NUM_WORKERS 2 \
        SOLVER.BASE_LR 0.01 \
        SOLVER.WARMUP_START_LR 0.001 \
        SOLVER.COSINE_END_LR 0.0001 \
        SOLVER.WARMUP_EPOCHS 1.0 \
        SOLVER.MAX_EPOCH 6 \
        DAAIM.AUGMENTATION_ENABLE True \
        DAAIM.PSEUDO_LABEL_ENABLE True \
        DAAIM.CONSISTENCY_LOSS 'ce_weighted' \
        DAAIM.PSEUDO_TARGETS 'binary' \
        DAAIM.CONSISTENCY_LOSS_WEIGHT 'ce_threshold_uniform' \
        DAAIM.THRESHOLDS '[0.9, 0.9, 0.9, 0.9, 0.9, 0.9]' \
        DAAIM.THRESHOLD 0.9 \
        DAAIM.RESIZE_ENABLE True

now=$(date +"%T")
echo "time after DA training to AVA with Kinetics Source: $now"
echo ${COUT_DIR}

# Evaluation
echo "#################### Evaluation on AVA #####################"
CFG_FILE=configs/DA-AIM/AVA/SLOWFAST_32x2_R50.yaml
python tools/run_net.py --cfg ${CFG_FILE} \
    --init_method ${init_method} \
    TRAIN.ENABLE False \
    TRAIN.AUTO_RESUME True \
    TENSORBOARD.ENABLE False \
    AVA.DETECTION_SCORE_THRESH ${EVAL_THRESH} \
    OUTPUT_DIR ${COUT_DIR} \
    AVA.FRAME_DIR ${TARGET_DIR} \
    AVA.FRAME_LIST_DIR "/cluster/work/cvl/susaha/dataset/action-dataset/datasets_yifan/ava_6_5000_all/frame_lists/" \
    AVA.ANNOTATION_DIR "/cluster/work/cvl/susaha/dataset/action-dataset/datasets_yifan/ava_6_5000_all/annotations/"

now=$(date +"%T")
echo "time after testing DA to AVA Supervised Kinetics on AVA: $now"
echo ${COUT_DIR}

# rsync -rav eu://cluster/work/cvl/susaha/experiments/experiments_yifan/ experiments_yifan_euler/