source ~/slowfast/bin/activate
echo "temp directory for the job" $TMPDIR
TARGET_DIR=${TMPDIR}/data/
mkdir -p $TARGET_DIR
echo  $TARGET_DIR
now=$(date +"%T")
echo "time before unpacking : $now"
SOURCE_DIR=/cluster/work/cvl/gusingh/data/ava-kinetics/ava/frames-tarballs/
time python unpack_dataset.py ${TARGET_DIR} ${SOURCE_DIR} --num_jobs=16
now=$(date +"%T")
echo "time after unpacking: $now"
ls -lh ${TARGET_DIR}
# du -hs ${TARGET_DIR}
now=$(date +"%T")
echo "time after checking space : $now"
echo ${TARGET_DIR}
COUT_DIR=/cluster/work/cvl/gusingh/data/ava-kinetics/ava/experiments/SLOWFAST_32x2_R50_YOLO_07/
CFG_FILE=configs/AVA-KIN/AVA/SLOWFAST_32x2_R50_YOLO.yaml
TRAIN_THRESH=0.5
#python tools/run_net.py --cfg configs/AVA-KIN/AVA/SLOWFAST_32x2_R50_YOLO.yaml AVA.FRAME_DIR ${TARGET_DIR} TRAIN.BATCH_SIZE 48 SOLVER.BASE_LR 0.075 AVA.DETECTION_SCORE_THRESH ${TRAIN_THRESH} OUTPUT_DIR ${COUT_DIR}
now=$(date +"%T")
echo "time after checking space : $now"
echo ${TARGET_DIR}
python tools/run_net.py --cfg ${CFG_FILE} AVA.FRAME_DIR ${TARGET_DIR} TRAIN.BATCH_SIZE 48 SOLVER.BASE_LR 0.075 TRAIN.ENABLE False AVA.DETECTION_SCORE_THRESH 0.7 OUTPUT_DIR ${COUT_DIR}
now=$(date +"%T")
echo "time after checking space : $now"
echo ${TARGET_DIR}
python tools/run_net.py --cfg ${CFG_FILE} AVA.FRAME_DIR ${TARGET_DIR} TRAIN.BATCH_SIZE 48 SOLVER.BASE_LR 0.075 TRAIN.ENABLE False AVA.DETECTION_SCORE_THRESH 0.6 OUTPUT_DIR ${COUT_DIR}
now=$(date +"%T")
echo "time after checking space : $now"
echo ${TARGET_DIR}
python tools/run_net.py --cfg ${CFG_FILE}  AVA.FRAME_DIR ${TARGET_DIR} TRAIN.BATCH_SIZE 48 SOLVER.BASE_LR 0.075 TRAIN.ENABLE False AVA.DETECTION_SCORE_THRESH 0.5 OUTPUT_DIR ${COUT_DIR}
now=$(date +"%T")
echo "time after checking space : $now"
echo ${TARGET_DIR}
python tools/run_net.py --cfg ${CFG_FILE}  AVA.FRAME_DIR ${TARGET_DIR} TRAIN.BATCH_SIZE 48 SOLVER.BASE_LR 0.075 TRAIN.ENABLE False AVA.DETECTION_SCORE_THRESH 0.45 OUTPUT_DIR ${COUT_DIR}
now=$(date +"%T")
echo "time after checking space : $now"
echo ${TARGET_DIR}
python tools/run_net.py --cfg ${CFG_FILE}  AVA.FRAME_DIR ${TARGET_DIR} TRAIN.BATCH_SIZE 48 SOLVER.BASE_LR 0.075 TRAIN.ENABLE False AVA.DETECTION_SCORE_THRESH 0.4 OUTPUT_DIR ${COUT_DIR}
now=$(date +"%T")
echo "time after checking space : $now"
echo ${TARGET_DIR}
python tools/run_net.py --cfg ${CFG_FILE} AVA.FRAME_DIR ${TARGET_DIR} TRAIN.BATCH_SIZE 48 SOLVER.BASE_LR 0.075 TRAIN.ENABLE False AVA.DETECTION_SCORE_THRESH 0.35 OUTPUT_DIR ${COUT_DIR}
now=$(date +"%T")
echo "time after checking space : $now"
echo ${TARGET_DIR}
python tools/run_net.py --cfg ${CFG_FILE} AVA.FRAME_DIR ${TARGET_DIR} TRAIN.BATCH_SIZE 48 SOLVER.BASE_LR 0.075 TRAIN.ENABLE False AVA.DETECTION_SCORE_THRESH 0.3 OUTPUT_DIR ${COUT_DIR}
now=$(date +"%T")
echo "time after checking space : $now"
echo ${TARGET_DIR}
python tools/run_net.py --cfg ${CFG_FILE}  AVA.FRAME_DIR ${TARGET_DIR} TRAIN.BATCH_SIZE 48 SOLVER.BASE_LR 0.075 TRAIN.ENABLE False AVA.DETECTION_SCORE_THRESH 0.2 OUTPUT_DIR ${COUT_DIR}
now=$(date +"%T")
echo "time after checking space : $now"
echo ${TARGET_DIR}
python tools/run_net.py --cfg ${CFG_FILE} AVA.FRAME_DIR ${TARGET_DIR} TRAIN.BATCH_SIZE 48 SOLVER.BASE_LR 0.075 TRAIN.ENABLE False AVA.DETECTION_SCORE_THRESH 0.1 OUTPUT_DIR ${COUT_DIR}
now=$(date +"%T")
echo "time after checking space : $now"
echo ${TARGET_DIR}
