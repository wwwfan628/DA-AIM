
# source ~/slowfast/bin/activate
ANNO_DIR=/raid/susaha/datasets/ava-kinetics/annotations/
# FRAME_LIST_DIR=/raid/susaha/datasets/ava-kinetics/annotations/
FRAMES_DIR=/raid/susaha/datasets/ava-kinetics/images/
COUT_DIR=/raid/susaha/datasets/ava-kinetics/ava/experiments/SLOWFAST_32x2_R50_YOLO_DA/
CFG_FILE=configs/AVA-KIN/AVA/SLOWFAST_32x2_R50_YOLO_DA.yaml
MODEL_PATH=/raid/susaha/datasets/ava-kinetics/pretrained_models/slowfast-mit/SLOWFAST_8x8_R50.pyth

CUDA_VISIBLE_DEVICES=3,4,5,6 python tools/run_net.py --cfg ${CFG_FILE} TRAIN.CHECKPOINT_FILE_PATH ${MODEL_PATH} \
        AVA.FRAME_DIR ${FRAMES_DIR} AVA.ANNOTATION_DIR ${ANNO_DIR}  \
        AVA.FRAME_LIST_DIR ${ANNO_DIR}/ava_frame_lists/ \
        AUX.FRAME_DIR ${FRAMES_DIR} AUX.ANNOTATION_DIR ${ANNO_DIR}  \
        AUX.FRAME_LIST_DIR ${ANNO_DIR}/kinetics_frame_lists/ \
        AUX.CLASSES 6 AUX.AUX_TYPE "clip_order" \
        OUTPUT_DIR ${COUT_DIR} \
        NUM_GPUS 4 \
        TRAIN.BATCH_SIZE 8 \
        AUX.LOSS_FACTOR 0.1 \
        SOLVER.BASE_LR 0.02

now=$(date +"%T")
echo "time after training : $now"

# CFG_FILE=configs/AVA-KIN/SLOWFAST_32x2_R50_YOLO.yaml
# echo ${TARGET_DIR}
# python tools/run_net.py --cfg ${CFG_FILE}  AVA.FRAME_DIR ${TARGET_DIR} TRAIN.ENABLE False TENSORBOARD.ENABLE False AVA.DETECTION_SCORE_THRESH ${EVAL_THRESH} OUTPUT_DIR ${COUT_DIR}
# now=$(date +"%T")
# echo "time after testing AVA on AVA-KIN: $now"
# echo ${TARGET_DIR}

# CFG_FILE=configs/AVA-KIN/AVA/SLOWFAST_32x2_R50_YOLO.yaml
# python tools/run_net.py --cfg ${CFG_FILE}  AVA.FRAME_DIR ${TARGET_DIR} TRAIN.ENABLE False TENSORBOARD.ENABLE False AVA.DETECTION_SCORE_THRESH ${EVAL_THRESH} OUTPUT_DIR ${COUT_DIR}
# now=$(date +"%T")
# echo "time after testing AVA on AVA: $now"
# echo ${TARGET_DIR}

