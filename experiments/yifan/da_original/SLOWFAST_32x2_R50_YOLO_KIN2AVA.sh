#!/bin/bash
#SBATCH --mail-type=ALL                                                         # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/home/yiflu/Desktop/experiments/KIN/%j.out                 # where to store the output ( %j is the JOBID )
#SBATCH --error=/home/yiflu/Desktop/experiments/KIN/%j.err                  # where to store error messages
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G

source /itet-stor/yiflu/net_scratch/conda/bin/activate py39
export PYTHONPATH=/home/yiflu/Desktop/domain-adaptive-slowfast/slowfast:$PYTHONPATH

TRAIN_THRESH=0.8
EVAL_THRESH=0.8
# Training
echo "#################### Training on KIN #####################"
cd ../../..
COUT_DIR=/home/yiflu/Desktop/experiments/KIN/
CFG_FILE=configs/Yifan/KIN/SLOWFAST_32x2_R50_YOLO.yaml
python tools/run_net.py --cfg ${CFG_FILE} TRAIN.AUTO_RESUME False AVA.DETECTION_SCORE_THRESH ${TRAIN_THRESH} OUTPUT_DIR ${COUT_DIR}
now=$(date +"%T")
echo "time after DA training to AVA with Kinetics Source: $now"

# Evaluation
echo "#################### Evaluation on AVA #####################"
CFG_FILE=configs/Yifan/AVA/SLOWFAST_32x2_R50_YOLO.yaml
python tools/run_net.py --cfg ${CFG_FILE} TRAIN.ENABLE False TENSORBOARD.ENABLE False AVA.DETECTION_SCORE_THRESH ${EVAL_THRESH} OUTPUT_DIR ${COUT_DIR}
now=$(date +"%T")
echo "time after testing DA to AVA Supervised Kinetics on AVA: $now"