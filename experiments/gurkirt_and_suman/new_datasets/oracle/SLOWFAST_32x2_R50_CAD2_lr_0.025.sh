#!/bin/bash
####################################################################################################
###########  Please check lines between #####, they need to be edited before experiment  ###########
####################################################################################################

###################################################

module load gcc/8.2.0 python_gpu/3.9.9 libjpeg-turbo eth_proxy; unset PYTHONPATH
source ~/envirs/slowfast110/bin/activate


## copy cad2
source experiments/gurkirt_and_suman/data_copy/movCAD2.sh
###################################################

TRAIN_THRESH=0.8
EVAL_THRESH=0.8
# Training
echo "#################### Training on CAD2 #####################"


###################################################

COUT_DIR=/cluster/work/cvl/susaha/experiments/experiments_yifan/cad2_oracle_lr0.025/

###################################################
init_method="tcp://localhost:3574"

CFG_FILE=configs/Gurkirt_and_Suman/CAD2/SLOWFAST_32x2_R50.yaml
python tools/run_net.py --cfg ${CFG_FILE} \
        --init_method ${init_method} \
        AVA.DETECTION_SCORE_THRESH ${TRAIN_THRESH} \
        OUTPUT_DIR ${COUT_DIR} \
        AVA.FRAME_DIR ${TARGET_DIR} \
        AVA.FRAME_LIST_DIR "/cluster/work/cvl/susaha/dataset/action-dataset/datasets_yifan/cad2/frame_lists/" \
        AVA.ANNOTATION_DIR "/cluster/work/cvl/susaha/dataset/action-dataset/datasets_yifan/cad2/annotations/" \
        MODEL.NUM_CLASSES 8 \
        TRAIN.BATCH_SIZE 24 \
        DATA_LOADER.NUM_WORKERS 2 \
        SOLVER.BASE_LR 0.025 \
        SOLVER.WARMUP_START_LR 0.0025 \
        SOLVER.COSINE_END_LR 0.00025 \
        SOLVER.WARMUP_EPOCHS 2.0 \
        SOLVER.MAX_EPOCH 10 \

now=$(date +"%T")
echo "time after training on CAD2: $now"
echo ${COUT_DIR}

# rsync -rav eu://cluster/work/cvl/susaha/experiments/experiments_yifan/ experiments_yifan_euler/