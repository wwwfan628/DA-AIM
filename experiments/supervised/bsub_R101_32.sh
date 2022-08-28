source ~/slowfast/bin/activate
echo "temp directory for the job" $TMPDIR
TARGET_DIR=${TMPDIR}/data/
mkdir -p $TARGET_DIR
echo  $TARGET_DIR
now=$(date +"%T")
echo "time before unpacking : $now"
SOURCE_DIR=/cluster/work/cvl/gusingh/data/ava/frames-tarballs/
time python unpack_dataset.py ${TARGET_DIR} ${SOURCE_DIR} --num_jobs=16
now=$(date +"%T")
echo "time after unpacking: $now"
ls -lh ${TARGET_DIR}
du -hs ${TARGET_DIR}
now=$(date +"%T")
echo "time after checking space : $now"
echo ${TARGET_DIR}
python tools/run_net.py --cfg configs/AVA/SLOWFAST_32x2_R101.yaml AVA.FRAME_DIR ${TARGET_DIR} TRAIN.BATCH_SIZE 32 SOLVER.BASE_LR 0.05 OUTPUT_DIR /cluster/work/cvl/gusingh/data/ava/experiments/SLOWFAST_32x2_R101_8GPUs_32bs 
