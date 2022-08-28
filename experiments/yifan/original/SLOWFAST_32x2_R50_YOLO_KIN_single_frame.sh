#!/bin/bash
#SBATCH --mail-type=ALL                                                         # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/home/yiflu/Desktop/experiments/KIN/%j.out                 # where to store the output ( %j is the JOBID )
#SBATCH --error=/home/yiflu/Desktop/experiments/KIN/%j.err                  # where to store error messages
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G

# source /itet-stor/yiflu/net_scratch/conda/bin/activate py39
# export PYTHONPATH=/home/yiflu/Desktop/domain-adaptive-slowfast/slowfast:$PYTHONPATH

TRAIN_THRESH=0.8
EVAL_THRESH=0.8
# Training
echo "#################### Training on KIN #####################"
# cd ../../..
# COUT_DIR=/home/yiflu/Desktop/experiments/KIN/
MODEL_PATH=/raid/susaha/datasets/ava-kinetics/pretrained_models/slowfast-kinetics/c2d_baseline_8x8_IN_pretrain_400k.pkl
## Download from https://dl.fbaipublicfiles.com/video-nonlocal/c2d_baseline_8x8_IN_pretrain_400k.pkl
COUT_DIR=/raid/susaha/datasets/ava-kinetics/yifan-run/
CFG_FILE=configs/Yifan/KIN/SLOWFAST_32x2_R50_YOLO_single_frame.yaml
python tools/run_net.py --cfg ${CFG_FILE} \
    TRAIN.AUTO_RESUME False AVA.DETECTION_SCORE_THRESH ${TRAIN_THRESH} \
    OUTPUT_DIR ${COUT_DIR} TRAIN.CHECKPOINT_FILE_PATH ${MODEL_PATH} \
    TRAIN.CHECKPOINT_TYPE 'caffe2' \
    TRAIN.CHECKPOINT_INFLATE True 


now=$(date +"%T")
echo "time after training with KIN: $now"