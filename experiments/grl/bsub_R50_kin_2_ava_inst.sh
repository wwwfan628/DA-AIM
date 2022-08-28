
source ~/slowfast/bin/activate

echo "temp directory for the job" $TMPDIR
TARGET_DIR=${TMPDIR}/data/
mkdir -p $TARGET_DIR
echo  "Data directory for the job is :: " $TARGET_DIR
now=$(date +"%T")
echo "time before unpacking : $now"


### Copy AVA train data
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
EVAL_THRESH=0.4
echo ${TARGET_DIR}

TRAIN_THRESH=0.1
EVAL_THRESH=0.4
GRL_LAMBDA=0.9


init_method="tcp://localhost:9905"
### Training 
COUT_DIR=/cluster/work/cvl/gusingh/data/ava-kinetics/kinetics/experiments/SLOWFAST_32x2_R50_MIT_YOLO_GRL_INST_09_wGT/
CFG_FILE=configs/AVA-KIN/KIN/SLOWFAST_32x2_R50_YOLO_GRL.yaml
python tools/run_net.py --cfg ${CFG_FILE} --init_method ${init_method} \
        AVA.FRAME_DIR ${TARGET_DIR} \
        AUX.FRAME_DIR ${TARGET_DIR} AVA.DETECTION_SCORE_THRESH ${TRAIN_THRESH} \
        OUTPUT_DIR ${COUT_DIR} GRL.LAMBDA ${GRL_LAMBDA} \
        NUM_GPUS 4 TEST.BATCH_SIZE 4

now=$(date +"%T")
echo "Setup:: GRL Kinetics source AVA target, time after training: $now"


## Evalution 
CFG_FILE=configs/AVA-KIN/AVA/SLOWFAST_32x2_R50_YOLO.yaml
python tools/run_net.py --cfg ${CFG_FILE}  --init_method ${init_method} \
        AVA.FRAME_DIR ${TARGET_DIR} \
        TRAIN.ENABLE False TENSORBOARD.ENABLE False \
        AVA.DETECTION_SCORE_THRESH ${EVAL_THRESH} \
        OUTPUT_DIR ${COUT_DIR} \
        NUM_GPUS 4 TEST.BATCH_SIZE 4

now=$(date +"%T")
echo "Setup:: GRL Kinetics source AVA target time after testing on AVA: $now"
echo ${TARGET_DIR}


CFG_FILE=configs/AVA-KIN/KIN/SLOWFAST_32x2_R50_YOLO.yaml
python tools/run_net.py --cfg ${CFG_FILE}  --init_method ${init_method} \
        AVA.FRAME_DIR ${TARGET_DIR} \
        TRAIN.ENABLE False TENSORBOARD.ENABLE False \
        AVA.DETECTION_SCORE_THRESH ${EVAL_THRESH} \
        OUTPUT_DIR ${COUT_DIR} \
        NUM_GPUS 4 TEST.BATCH_SIZE 4
now=$(date +"%T")
echo "Setup:: GRL Kinetics source AVA target time after testing on KIN: $now"
echo ${TARGET_DIR}


CFG_FILE=configs/AVA-KIN/SLOWFAST_32x2_R50_YOLO.yaml
echo ${TARGET_DIR}
python tools/run_net.py --cfg ${CFG_FILE} --init_method ${init_method} \
        AVA.FRAME_DIR ${TARGET_DIR} \
        TRAIN.ENABLE False TENSORBOARD.ENABLE False \
        AVA.DETECTION_SCORE_THRESH ${EVAL_THRESH} \
        OUTPUT_DIR ${COUT_DIR} \
        NUM_GPUS 4 TEST.BATCH_SIZE 4

now=$(date +"%T")
echo "Setup:: GRL Kinetics source AVA target time after testing on AVA-KIN: $now"
echo ${TARGET_DIR}