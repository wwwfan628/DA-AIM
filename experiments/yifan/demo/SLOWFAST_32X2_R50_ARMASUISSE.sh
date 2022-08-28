#!/bin/bash
#SBATCH --mail-type=ALL                                                         # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/home/yiflu/Desktop/experiments/ARMASUISSE_DEMO/%j.out                 # where to store the output ( %j is the JOBID )
#SBATCH --error=/home/yiflu/Desktop/experiments/ARMASUISSE_DEMO/%j.err                  # where to store error messages
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G

source /itet-stor/yiflu/net_scratch/conda/bin/activate py39
export PYTHONPATH=/home/yiflu/Desktop/domain-adaptive-slowfast/slowfast:$PYTHONPATH

EVAL_THRESH=0.8
# DEMO
echo "#################### Demo on ARMASUISSE #####################"
cd ../../..
COUT_DIR=/home/yiflu/Desktop/experiments/ARMASUISSE_DEMO/
CFG_FILE=configs/Yifan/DEMO/ARMASUISSE/SLOWFAST_32x2_R50.yaml
python tools/run_net.py --cfg ${CFG_FILE} AVA.DETECTION_SCORE_THRESH ${EVAL_THRESH} OUTPUT_DIR ${COUT_DIR}
now=$(date +"%T")
echo "time after Demo on Armasuisse: $now"

