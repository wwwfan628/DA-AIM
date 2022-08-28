echo "temp directory for the job" $TMPDIR
TARGET_DIR=${TMPDIR}/data/
mkdir -p $TARGET_DIR
echo  "Data directory for the job is :: " $TARGET_DIR
now=$(date +"%T")
echo "time before unpacking : $now"

## Copy Kinetics+CAD2 data
SOURCE_DIR=/cluster/work/cvl/susaha/dataset/action-dataset/yifan-tars/tar_kinetics_8_5000_all+cad2/
time python data_scripts/unpack_dataset.py ${TARGET_DIR} ${SOURCE_DIR} --num_jobs=16
now=$(date +"%T")
echo "time after Kinetics+CAD2 unpacking: $now"

## list videos and check space
echo "number of videos extracted are:: "
ls ${TARGET_DIR} | wc -l
now=$(date +"%T")
echo "time after moving data : $now"