#!/bin/bash
SCRIPT_DIR=$(cd $(dirname $0); pwd)
BASE_DIR=$SCRIPT_DIR/../..

mkdir -p ${BASE_DIR}/datasets/soccer/videos

if [ ! -f ${BASE_DIR}/datasets/Sequences.zip ]; then
	wget https://www.dropbox.com/s/3zhms9iv4gy3k81/Sequences.zip?dl=1 -O ${BASE_DIR}/datasets/Sequences.zip
fi

unzip ${BASE_DIR}/datasets/Sequences.zip -d ${BASE_DIR}/datasets/soccer/videos
mv ${BASE_DIR}/datasets/soccer/videos/"Film Role-0 ID-1 T-2 m00s00-000-m00s00-185.avi" ${BASE_DIR}/datasets/soccer/videos/ID-1.avi
mv ${BASE_DIR}/datasets/soccer/videos/"Film Role-0 ID-2 T-2 m00s00-000-m00s00-185.avi" ${BASE_DIR}/datasets/soccer/videos/ID-2.avi
mv ${BASE_DIR}/datasets/soccer/videos/"Film Role-0 ID-3 T-2 m00s00-000-m00s00-185.avi" ${BASE_DIR}/datasets/soccer/videos/ID-3.avi
mv ${BASE_DIR}/datasets/soccer/videos/"Film Role-0 ID-4 T-2 m00s00-000-m00s00-185.avi" ${BASE_DIR}/datasets/soccer/videos/ID-4.avi
mv ${BASE_DIR}/datasets/soccer/videos/"Film Role-0 ID-5 T-2 m00s00-000-m00s00-185.avi" ${BASE_DIR}/datasets/soccer/videos/ID-5.avi
mv ${BASE_DIR}/datasets/soccer/videos/"Film Role-0 ID-6 T-2 m00s00-000-m00s00-185.avi" ${BASE_DIR}/datasets/soccer/videos/ID-6.avi

if [ ! -f ${BASE_DIR}/datasets/soccer/soccer_annos.zip ]; then
	wget https://drive.google.com/uc?id=1Oqr33EXOEjERothXvVAqAeKiKuB-per1 -O ${BASE_DIR}/datasets/soccer/soccer_annos.zip
fi

unzip ${BASE_DIR}/datasets/soccer/soccer_annos.zip -d ${BASE_DIR}/datasets/soccer/

python3 main.py --config-name=extract_frame dataset=soccer

