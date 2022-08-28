source ~/slowfast/bin/activate

echo "temp directory for the job" $TMPDIR
TARGET_DIR=${TMPDIR}/data/
mkdir -p $TARGET_DIR
echo  "Data directory for the job is :: " $TARGET_DIR
now=$(date +"%T")
echo "time before unpacking : $now"


### Copy AVA Train data
SOURCE_DIR=/cluster/work/cvl/gusingh/data/ava-kinetics/ava/train-images-tars/
time python data_scripts/unpack_dataset.py ${TARGET_DIR} ${SOURCE_DIR} --num_jobs=16
now=$(date +"%T")
echo "time after unpacking AVA train: $now"


### Copy AVA Val data
SOURCE_DIR=/cluster/work/cvl/gusingh/data/ava-kinetics/ava/val-images-tars/
time python data_scripts/unpack_dataset.py ${TARGET_DIR} ${SOURCE_DIR} --num_jobs=16
now=$(date +"%T")
echo "time after unpacking AVA val: $now"


## Copy Kinetics Train data
SOURCE_DIR=/cluster/work/cvl/gusingh/data/ava-kinetics/kinetics/train-images-tars/
time python data_scripts/unpack_dataset.py ${TARGET_DIR} ${SOURCE_DIR} --num_jobs=16
echo "time after unpacking Kinetics train: $now"


## Copy Kinetics Val data
SOURCE_DIR=/cluster/work/cvl/gusingh/data/ava-kinetics/kinetics/val-images-tars/
time python data_scripts/unpack_dataset.py ${TARGET_DIR} ${SOURCE_DIR} --num_jobs=16
now=$(date +"%T")
echo "time after Kinetics val unpacking: $now"

## list videos and check space
echo "number of videos extracted are:: "
ls ${TARGET_DIR} | wc -l
echo "Space consumption is:: "
du -hs ${TARGET_DIR}
now=$(date +"%T")
echo "time after checking space : $now"

TRAIN_THRESH=0.1
EVAL_THRESH=0.4

echo ${TARGET_DIR}
COUT_DIR=/cluster/work/cvl/gusingh/data/ava-kinetics/ava/experiments/SLOWFAST_32x2_R50_MIT_YOLO_AUX_KIN_CL/
CFG_FILE=configs/AVA-KIN/AVA/SLOWFAST_32x2_R50_YOLO_AUX.yaml
python tools/run_net.py --cfg ${CFG_FILE} AVA.FRAME_DIR ${TARGET_DIR} AUX.FRAME_DIR ${TARGET_DIR} AVA.DETECTION_SCORE_THRESH ${TRAIN_THRESH} OUTPUT_DIR ${COUT_DIR}
now=$(date +"%T")
echo "time after AUX training on Kinetics: $now"

MODEL_PATH=${COUT_DIR}checkpoints/checkpoint_epoch_00005.pyth
COUT_DIR=/cluster/work/cvl/gusingh/data/ava-kinetics/ava/experiments/SLOWFAST_32x2_R50_MIT_YOLO_AUX_KIN_CL_SUP_AVA/
CFG_FILE=configs/AVA-KIN/AVA/SLOWFAST_32x2_R50_YOLO.yaml
python tools/run_net.py --cfg ${CFG_FILE} AVA.FRAME_DIR ${TARGET_DIR} AVA.DETECTION_SCORE_THRESH ${TRAIN_THRESH} OUTPUT_DIR ${COUT_DIR} TRAIN.CHECKPOINT_FILE_PATH ${MODEL_PATH} TRAIN.CHECKPOINT_EPOCH_RESET True
now=$(date +"%T")
echo "time after supervised  training on AVA: $now"

CFG_FILE=configs/AVA-KIN/AVA/SLOWFAST_32x2_R50_YOLO.yaml
python tools/run_net.py --cfg ${CFG_FILE}  AVA.FRAME_DIR ${TARGET_DIR} TRAIN.ENABLE False TENSORBOARD.ENABLE False AVA.DETECTION_SCORE_THRESH ${EVAL_THRESH} OUTPUT_DIR ${COUT_DIR}
now=$(date +"%T")
echo "time after testing AUX Kinetics Supervised AVA on AVA: $now"
echo ${TARGET_DIR}

CFG_FILE=configs/AVA-KIN/KIN/SLOWFAST_32x2_R50_YOLO.yaml
python tools/run_net.py --cfg ${CFG_FILE}  AVA.FRAME_DIR ${TARGET_DIR} TRAIN.ENABLE False TENSORBOARD.ENABLE False AVA.DETECTION_SCORE_THRESH ${EVAL_THRESH} OUTPUT_DIR ${COUT_DIR}
now=$(date +"%T")
echo "time after testing AUX Kinetics Supervised AVA on KIN: $now"
echo ${TARGET_DIR}

CFG_FILE=configs/AVA-KIN/SLOWFAST_32x2_R50_YOLO.yaml
echo ${TARGET_DIR}
python tools/run_net.py --cfg ${CFG_FILE}  AVA.FRAME_DIR ${TARGET_DIR} TRAIN.ENABLE False TENSORBOARD.ENABLE False AVA.DETECTION_SCORE_THRESH ${EVAL_THRESH} OUTPUT_DIR ${COUT_DIR}
now=$(date +"%T")
echo "time after testing AUX Kinetics Supervised AVA on AVA-KIN: $now"
echo ${TARGET_DIR}
