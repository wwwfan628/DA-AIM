
source ~/slowfast/bin/activate

echo "temp directory for the job" $TMPDIR
TARGET_DIR=${TMPDIR}/data/
mkdir -p $TARGET_DIR
echo  "Data directory for the job is :: " $TARGET_DIR
now=$(date +"%T")
echo "time before unpacking : $now"


### Copy AVA train data
SOURCE_DIR=/cluster/work/cvl/gusingh/data/ava-kinetics/ava/train-images-tars/
time python unpack_dataset.py ${TARGET_DIR} ${SOURCE_DIR} --num_jobs=16
now=$(date +"%T")
echo "time after unpacking AVA train: $now"


### Copy AVA Val data
SOURCE_DIR=/cluster/work/cvl/gusingh/data/ava-kinetics/ava/val-images-tars/
time python unpack_dataset.py ${TARGET_DIR} ${SOURCE_DIR} --num_jobs=16
now=$(date +"%T")
echo "time after unpacking AVA val: $now"


## Copy Kinetics Train data
SOURCE_DIR=/cluster/work/cvl/gusingh/data/ava-kinetics/kinetics/train-images-tars/
time python unpack_dataset.py ${TARGET_DIR} ${SOURCE_DIR} --num_jobs=16
now=$(date +"%T")
echo "time after unpacking Kinetics train: $now"


## Copy Kinetics Val data
SOURCE_DIR=/cluster/work/cvl/gusingh/data/ava-kinetics/kinetics/val-images-tars/
time python unpack_dataset.py ${TARGET_DIR} ${SOURCE_DIR} --num_jobs=16
now=$(date +"%T")
echo "time after Kinetics val unpacking: $now"


## list videos and check space
echo "number of videos extracted are:: "
ls ${TARGET_DIR} | wc -l
echo "Space consumption is:: "
du -hs ${TARGET_DIR}
now=$(date +"%T")
echo "time after checking space : $now"

echo ${TARGET_DIR}
COUT_DIR=/cluster/work/cvl/gusingh/data/ava-kinetics/experiments/SLOWFAST_32x2_R50_MIT_YOLO_01/
CFG_FILE=configs/AVA-KIN/SLOWFAST_32x2_R50_YOLO.yaml
TRAIN_THRESH=0.1
EVAL_THRESH=0.4
python tools/run_net.py --cfg ${CFG_FILE} AVA.FRAME_DIR ${TARGET_DIR} AVA.DETECTION_SCORE_THRESH ${TRAIN_THRESH} OUTPUT_DIR ${COUT_DIR}
now=$(date +"%T")
echo "time after training : $now"

CFG_FILE=configs/AVA-KIN/SLOWFAST_32x2_R50_YOLO.yaml
echo ${TARGET_DIR}
python tools/run_net.py --cfg ${CFG_FILE}  AVA.FRAME_DIR ${TARGET_DIR} TRAIN.ENABLE False TENSORBOARD.ENABLE False AVA.DETECTION_SCORE_THRESH ${EVAL_THRESH} OUTPUT_DIR ${COUT_DIR}
now=$(date +"%T")
echo "time after testing AVA-KIN on AVA-KIN: $now"
echo ${TARGET_DIR}

CFG_FILE=configs/AVA-KIN/AVA/SLOWFAST_32x2_R50_YOLO.yaml
python tools/run_net.py --cfg ${CFG_FILE}  AVA.FRAME_DIR ${TARGET_DIR} TRAIN.ENABLE False TENSORBOARD.ENABLE False AVA.DETECTION_SCORE_THRESH ${EVAL_THRESH} OUTPUT_DIR ${COUT_DIR}
now=$(date +"%T")
echo "time after testing AVA-KIN on AVA: $now"
echo ${TARGET_DIR}

CFG_FILE=configs/AVA-KIN/KIN/SLOWFAST_32x2_R50_YOLO.yaml
python tools/run_net.py --cfg ${CFG_FILE}  AVA.FRAME_DIR ${TARGET_DIR} TRAIN.ENABLE False TENSORBOARD.ENABLE False AVA.DETECTION_SCORE_THRESH ${EVAL_THRESH} OUTPUT_DIR ${COUT_DIR}
now=$(date +"%T")
echo "time after testing AVA-KIN on KIN: $now"
echo ${TARGET_DIR}

# echo ${TARGET_DIR}
# python tools/run_net.py --cfg ${CFG_FILE}  AVA.FRAME_DIR ${TARGET_DIR} TRAIN.ENABLE False AVA.DETECTION_SCORE_THRESH 0.5 OUTPUT_DIR ${COUT_DIR}
# now=$(date +"%T")
# echo "time after testing 0.5 : $now"
# echo ${TARGET_DIR}
# python tools/run_net.py --cfg ${CFG_FILE}  AVA.FRAME_DIR ${TARGET_DIR} TRAIN.ENABLE False AVA.DETECTION_SCORE_THRESH 0.45 OUTPUT_DIR ${COUT_DIR}
# now=$(date +"%T")
# echo "time after testing 0.45 : $now"
# echo ${TARGET_DIR}
# python tools/run_net.py --cfg ${CFG_FILE}  AVA.FRAME_DIR ${TARGET_DIR} TRAIN.ENABLE False AVA.DETECTION_SCORE_THRESH 0.4 OUTPUT_DIR ${COUT_DIR}
# now=$(date +"%T")
# echo "time after testing 0.4 : $now"
# echo ${TARGET_DIR}
# python tools/run_net.py --cfg ${CFG_FILE} AVA.FRAME_DIR ${TARGET_DIR} TRAIN.ENABLE False AVA.DETECTION_SCORE_THRESH 0.35 OUTPUT_DIR ${COUT_DIR}
# now=$(date +"%T")
# echo "time after testing 0.35 : $now"
# echo ${TARGET_DIR}
# python tools/run_net.py --cfg ${CFG_FILE} AVA.FRAME_DIR ${TARGET_DIR} TRAIN.ENABLE False AVA.DETECTION_SCORE_THRESH 0.3 OUTPUT_DIR ${COUT_DIR}
# now=$(date +"%T")
# echo "time after testing 0.3 : $now"
# echo ${TARGET_DIR}
# python tools/run_net.py --cfg ${CFG_FILE}  AVA.FRAME_DIR ${TARGET_DIR} TRAIN.ENABLE False AVA.DETECTION_SCORE_THRESH 0.2 OUTPUT_DIR ${COUT_DIR}
# now=$(date +"%T")
# echo "time after testing 0.2 : $now"
# echo ${TARGET_DIR}
# python tools/run_net.py --cfg ${CFG_FILE} AVA.FRAME_DIR ${TARGET_DIR} TRAIN.ENABLE False AVA.DETECTION_SCORE_THRESH 0.1 OUTPUT_DIR ${COUT_DIR}
# now=$(date +"%T")
# echo "time after testing 0.1 : $now"
# echo ${TARGET_DIR}
