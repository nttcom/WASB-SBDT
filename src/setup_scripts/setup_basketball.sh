#!/bin/bash
SCRIPT_DIR=$(cd $(dirname $0); pwd)
BASE_DIR=$SCRIPT_DIR/../..

mv ${BASE_DIR}/datasets/NBA_data ${BASE_DIR}/datasets/basketball
wget https://drive.google.com/uc?id=1eH3n4uB4d8T-YKLRh46SshD0jYqQbnGF -O ${BASE_DIR}/datasets/basketball/basketball_annos.zip
unzip ${BASE_DIR}/datasets/basketball/basketball_annos.zip -d ${BASE_DIR}/datasets/basketball/

