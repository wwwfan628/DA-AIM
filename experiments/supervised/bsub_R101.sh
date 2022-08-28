source ~/slowfast/bin/activate
echo "temp directory for the job" $TMPDIR
TARGET_DIR=${TMPDIR}/data/
mkdir -p $TARGET_DIR
echo  $TARGET_DIR
SOURCE_DIR=/cluster/work/cvl/gusingh/data/ava/frames-tarballs/
time python unpack_dataset.py ${TARGET_DIR} ${SOURCE_DIR} --num_jobs=1
ls -lh ${TARGET_DIR}
#164407244
python tools/run_net.py --cfg configs/AVA/SLOWFAST_32x2_R101.yaml AVA.FRAME_DIR ${TARGET_DIR}
