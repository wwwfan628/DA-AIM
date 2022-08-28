#!/bin/bash
#SBATCH --mail-type=ALL                                                     # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/home/yiflu/Desktop/experiments/KIN2CAD1_DACS_PLOT_CM/%j.out                 # where to store the output ( %j is the JOBID )
#SBATCH --error=/home/yiflu/Desktop/experiments/KIN2CAD1_DACS_PLOT_CM/%j.err                  # where to store error messages
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G

source /itet-stor/yiflu/net_scratch/conda/bin/activate py39
export PYTHONPATH=/home/yiflu/Desktop/domain-adaptive-slowfast/slowfast:$PYTHONPATH

TRAIN_THRESH=0.8
EVAL_THRESH=0.8
# Training
echo "#################### Training on KIN #####################"
cd ../../..
COUT_DIR=/home/yiflu/Desktop/experiments/KIN2CAD1_DACS_PLOT_CM/
CFG_FILE=configs/Yifan/KIN2CAD1/SLOWFAST_32x2_R50_DA_DACS_PLOT_CM.yaml
python tools/run_net.py --cfg ${CFG_FILE} TRAIN.AUTO_RESUME False AVA.DETECTION_SCORE_THRESH ${TRAIN_THRESH} OUTPUT_DIR ${COUT_DIR}
now=$(date +"%T")
echo "time after training with KIN: $now"
