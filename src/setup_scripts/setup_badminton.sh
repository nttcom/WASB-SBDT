#!/bin/bash
SCRIPT_DIR=$(cd $(dirname $0); pwd)
BASE_DIR=$SCRIPT_DIR/../..

mkdir -p ${BASE_DIR}/datasets/badminton

unzip ${BASE_DIR}/datasets/TrackNetV2.zip -d ${BASE_DIR}/datasets

mv ${BASE_DIR}/datasets/TrackNetV2/Professional/* ${BASE_DIR}/datasets/badminton/
mv ${BASE_DIR}/datasets/TrackNetV2/Amateur/match1 ${BASE_DIR}/datasets/badminton/match24
mv ${BASE_DIR}/datasets/TrackNetV2/Amateur/match2 ${BASE_DIR}/datasets/badminton/match25
mv ${BASE_DIR}/datasets/TrackNetV2/Amateur/match3 ${BASE_DIR}/datasets/badminton/match26
mv ${BASE_DIR}/datasets/TrackNetV2/Test/match1 ${BASE_DIR}/datasets/badminton/test_match1
mv ${BASE_DIR}/datasets/TrackNetV2/Test/match2 ${BASE_DIR}/datasets/badminton/test_match2
mv ${BASE_DIR}/datasets/TrackNetV2/Test/match3 ${BASE_DIR}/datasets/badminton/test_match3
rm -r ${BASE_DIR}/datasets/TrackNetV2

python3 main.py --config-name=extract_frame dataset=badminton

