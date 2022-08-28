IN_DATA_DIR="videos_15min"
OUT_DATA_DIR="videos_15min_x256"

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi

for video in $(ls -A1 -U ${IN_DATA_DIR}/*)
do
  video_name=${video##*/}

  out_video=${OUT_DATA_DIR}/${video_name}

  ffmpeg -i "${video}" -c:v libx264 -preset ultrafast -filter:v scale="trunc(oh*a/2)*2:256" -q:v 1 -c:a copy "${out_video}"
  
done