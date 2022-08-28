#!/bin/bash
#SBATCH --mail-type=ALL                                                     # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/home/yiflu/Desktop/experiments/TSNE/%j.out                 # where to store the output ( %j is the JOBID )
#SBATCH --error=/home/yiflu/Desktop/experiments/TSNE/%j.err                  # where to store error messages
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G

source /itet-stor/yiflu/net_scratch/conda/bin/activate py39
export PYTHONPATH=/home/yiflu/Desktop/domain-adaptive-slowfast/slowfast:$PYTHONPATH

EVAL_THRESH=0.8

# Evaluation
echo "#################### Evaluation on CAD1 #####################"
cd ../../..
COUT_DIR=/home/yiflu/Desktop/experiments/TSNE/
# CFG_FILE=configs/Yifan/TSNE/SLOWFAST_32x2_R50_CKP_KIN2CAD1_CAD1.yaml
# CFG_FILE=configs/Yifan/TSNE/SLOWFAST_32x2_R50_CKP_KIN2CAD1_KIN.yaml
# CFG_FILE=configs/Yifan/TSNE/SLOWFAST_32x2_R50_CKP_KINCAD22CAD1_CAD1.yaml
# CFG_FILE=configs/Yifan/TSNE/SLOWFAST_32x2_R50_CKP_KINCAD22CAD1_KINCAD2.yaml
CFG_FILE=configs/Yifan/TSNE/SLOWFAST_32x2_R50_CKP_KIN2CAD1_NO_UDA_CAD1.yaml
# CFG_FILE=configs/Yifan/TSNE/SLOWFAST_32x2_R50_CKP_KINCAD22CAD1_NO_UDA_CAD1.yaml
python tools/run_net.py --cfg ${CFG_FILE} TRAIN.ENABLE False TENSORBOARD.ENABLE True AVA.DETECTION_SCORE_THRESH ${EVAL_THRESH} OUTPUT_DIR ${COUT_DIR}
now=$(date +"%T")
echo "time after testing DA to CAD1 Supervised KIN on CAD1: $now"