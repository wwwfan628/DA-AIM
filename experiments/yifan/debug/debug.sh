#!/bin/bash
#SBATCH --mail-type=ALL                                                         # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/home/yiflu/Desktop/experiments/DEBUG/%j.out                 # where to store the output ( %j is the JOBID )
#SBATCH --error=/home/yiflu/Desktop/experiments/DEBUG/%j.err                  # where to store error messages
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G

source /itet-stor/yiflu/net_scratch/conda/bin/activate py39
export PYTHONPATH=/home/yiflu/Desktop/domain-adaptive-slowfast/slowfast:$PYTHONPATH

TRAIN_THRESH=0.8
EVAL_THRESH=0.8
# Training
echo "#################### Training #####################"
cd ../../..
COUT_DIR=/home/yiflu/Desktop/experiments/DEBUG/
# MODEL_PATH=/raid/susaha/datasets/ava-kinetics/pretrained_models/slowfast-kinetics/c2d_baseline_8x8_IN_pretrain_400k.pkl
## Download from https://dl.fbaipublicfiles.com/video-nonlocal/c2d_baseline_8x8_IN_pretrain_400k.pkl
CFG_FILE=configs/Yifan/DEBUG/SLOWFAST_32x2_R50_YOLO.yaml

python tools/run_net.py --cfg ${CFG_FILE} \
    TRAIN.AUTO_RESUME False AVA.DETECTION_SCORE_THRESH ${TRAIN_THRESH} \
    OUTPUT_DIR ${COUT_DIR}


now=$(date +"%T")
echo "time after training: $now"

