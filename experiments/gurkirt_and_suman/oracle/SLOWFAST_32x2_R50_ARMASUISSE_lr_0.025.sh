#!/bin/bash
####################################################################################################
###########  Please check lines between #####, they need to be edited before experiment  ###########
####################################################################################################

###################################################

module load gcc/8.2.0 python_gpu/3.9.9 libjpeg-turbo eth_proxy; unset PYTHONPATH
source ~/envirs/slowfast110/bin/activate


## copy armasuisse 3
source experiments/gurkirt_and_suman/data_copy/movAS3.sh
###################################################

TRAIN_THRESH=0.8
EVAL_THRESH=0.8
# Training
echo "#################### Training on ARMASUISSE #####################"


###################################################

COUT_DIR=/cluster/work/cvl/susaha/experiments/experiments_yifan/armasuisse_oracle_lr0.025/

###################################################
init_method="tcp://localhost:6690"

CFG_FILE=configs/Gurkirt_and_Suman/ARMASUISSE/SLOWFAST_32x2_R50.yaml
python tools/run_net.py --cfg ${CFG_FILE} \
        --init_method ${init_method} \
        AVA.DETECTION_SCORE_THRESH ${TRAIN_THRESH} \
        OUTPUT_DIR ${COUT_DIR} \
        AVA.FRAME_DIR ${TARGET_DIR} \
        AVA.FRAME_LIST_DIR "/cluster/work/cvl/susaha/dataset/action-dataset/datasets_yifan/armasuisse_3_5000_all/frame_lists/" \
        AVA.ANNOTATION_DIR "/cluster/work/cvl/susaha/dataset/action-dataset/datasets_yifan/armasuisse_3_5000_all/annotations/" \
        MODEL.NUM_CLASSES 3 \
        TRAIN.BATCH_SIZE 24 \
        DATA_LOADER.NUM_WORKERS 2 \
        SOLVER.BASE_LR 0.025 \
        SOLVER.WARMUP_START_LR 0.0025 \
        SOLVER.COSINE_END_LR 0.00025 \
        SOLVER.WARMUP_EPOCHS 1.0 \
        SOLVER.MAX_EPOCH 4 \

now=$(date +"%T")
echo "time after training on ARMASUISSE: $now"
echo ${COUT_DIR}

# rsync -rav eu://cluster/work/cvl/susaha/experiments/experiments_yifan/ experiments_yifan_euler/