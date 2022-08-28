#!/bin/bash
#SBATCH --mail-type=ALL                                                         # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/home/yiflu/Desktop/experiments/AVA_DEMO_1/%j.out                 # where to store the output ( %j is the JOBID )
#SBATCH --error=/home/yiflu/Desktop/experiments/AVA_DEMO_1/%j.err                  # where to store error messages
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G

source /itet-stor/yiflu/net_scratch/conda/bin/activate py39
export PYTHONPATH=/home/yiflu/Desktop/domain-adaptive-slowfast/slowfast:$PYTHONPATH

EVAL_THRESH=0.8
# DEMO
echo "#################### Demo on AVA #####################"
cd ../../..
COUT_DIR=/home/yiflu/Desktop/experiments/AVA_DEMO_1/
CFG_FILE=configs/Yifan/DEMO/AVA/SLOWFAST_32x2_R50_1.yaml
python tools/run_net.py --cfg ${CFG_FILE} AVA.DETECTION_SCORE_THRESH ${EVAL_THRESH} OUTPUT_DIR ${COUT_DIR}
now=$(date +"%T")
echo "time after Demo on AVA: $now"