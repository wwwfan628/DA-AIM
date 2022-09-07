#!/bin/bash
#SBATCH --mail-type=ALL                                                         # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/home/yiflu/Desktop/experiments/KIN2AVA/%j.out                 # where to store the output ( %j is the JOBID )
#SBATCH --error=/home/yiflu/Desktop/experiments/KIN2AVA/%j.err                  # where to store error messages
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G

source /itet-stor/yiflu/net_scratch/conda/bin/activate py39
export PYTHONPATH=/home/yiflu/Desktop/DA-AIM/slowfast:$PYTHONPATH

TRAIN_THRESH=0.8
EVAL_THRESH=0.8
COUT_DIR=/home/yiflu/Desktop/experiments/KIN2AVA/
# Training
echo "#################### Training on KIN #####################"
cd ..
CFG_FILE=configs/DA-AIM/KIN2AVA/SLOWFAST_32x2_R50_DA_AIM.yaml
python tools/run_net.py --cfg ${CFG_FILE} \
        AVA.DETECTION_SCORE_THRESH ${TRAIN_THRESH} \
        OUTPUT_DIR ${COUT_DIR} \
        AVA.FRAME_DIR "/srv/beegfs02/scratch/da_action/data/kinetics/images" \
        AVA.FRAME_LIST_DIR "/srv/beegfs02/scratch/da_action/data/datasets_yifan/kinetics_6_5000_all/frame_lists/" \
        AVA.ANNOTATION_DIR "/srv/beegfs02/scratch/da_action/data/datasets_yifan/kinetics_6_5000_all/annotations/" \
        AUX.FRAME_DIR "/srv/beegfs02/scratch/da_action/data/ava/frames/" \
        AUX.FRAME_LIST_DIR "/srv/beegfs02/scratch/da_action/data/datasets_yifan/ava_6_5000_all/frame_lists/" \
        AUX.ANNOTATION_DIR "/srv/beegfs02/scratch/da_action/data/datasets_yifan/ava_6_5000_all/annotations/" \
        TRAIN.BATCH_SIZE 8 \
        TRAIN.CHECKPOINT_FILE_PATH "/srv/beegfs-benderdata/scratch/da_action/data/ava-kinetics/SLOWFAST_8x8_R50.pyth" \
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
    AVA.FRAME_DIR "/srv/beegfs02/scratch/da_action/data/kinetics/images" \
    AVA.FRAME_LIST_DIR "/srv/beegfs02/scratch/da_action/data/datasets_yifan/kinetics_6_5000_all/frame_lists/" \
    AVA.ANNOTATION_DIR "/srv/beegfs02/scratch/da_action/data/datasets_yifan/kinetics_6_5000_all/annotations/"
now=$(date +"%T")
echo "time after testing DA to AVA Supervised Kinetics on AVA: $now"

