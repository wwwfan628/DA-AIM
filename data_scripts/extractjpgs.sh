IN_DATA_DIR="videos_15min"
OUT_DATA_DIR="frames"

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi

for video in $(ls -A1 -U ${IN_DATA_DIR}/*)
do
  video_name=${video##*/}

  if [[ $video_name = *".webm" ]]; then
    video_name=${video_name::-5}
  else
    video_name=${video_name::-4}
  fi

  out_video_dir=${OUT_DATA_DIR}/${video_name}/
  echo ${out_video_dir}
  echo ${video_name}
  if [[ ! -d "${out_video_dir}" ]]; then
    echo "${out_video_dir} doesn't exist. Creating it.";
    mkdir -p ${out_video_dir}
  fi


  count=$(ls ${out_video_dir} | wc -l)
  echo "Number of files: " $count
  if [ $count -lt 27028 ]; then
    out_name="${out_video_dir}/${video_name}_%06d.jpg"
    echo "extract" ${out_name}
    ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"
  fi
done
