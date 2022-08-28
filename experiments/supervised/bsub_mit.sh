
source ~/slowfast/bin/activate
echo "temp directory for the job" $TMPDIR
TARGET_DIR=${TMPDIR}/data/
mkdir -p $TARGET_DIR
echo  $TARGET_DIR
SOURCE_DIR=/cluster/work/cvl/gusingh/data/mit/videos-tarballs/
time python data_scripts/unpack_dataset.py ${TARGET_DIR} ${SOURCE_DIR} --num_jobs=16
ls -lh ${TARGET_DIR}
du -hs ${TARGET_DIR}
#164407244
python tools/run_net.py --cfg configs/MIT/SLOWFAST_8x8_R101_SCRATCH.yaml DATA.PATH_PREFIX ${TARGET_DIR}
# python tools/run_net.py --cfg configs/MIT/SLOWFAST_8x8_R50_SCRATCH.yaml DATA.PATH_PREFIX ${TARGET_DIR}
#python tools/run_net.py --cfg configs/MIT/SLOWFAST_8x8_R50_STEP.yaml DATA.PATH_PREFIX ${TARGET_DIR}
#PATH_PREFIX ${TARGET_DIR}
# bsub -n 96 -R "rusage[mem=5000,ngpus_excl_p=8,scratch=5000]" -W 360:00 < experiments/supervised/bsub_mit.sh