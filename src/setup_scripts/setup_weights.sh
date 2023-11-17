#!/bin/bash
SCRIPT_DIR=$(cd $(dirname $0); pwd)
BASE_DIR=$SCRIPT_DIR/../..

mkdir -p ${BASE_DIR}/pretrained_weights

# wasb
wget https://drive.google.com/uc?id=1pg0MpMtKZ6ziYEr4oyfKYPOO3hjLw94l -O ${BASE_DIR}/pretrained_weights/wasb_soccer_best.pth.tar
wget https://drive.google.com/uc?id=14AeyIOCQ2UaQmbZLNQJa1H_eSwxUXk7z -O ${BASE_DIR}/pretrained_weights/wasb_tennis_best.pth.tar
wget https://drive.google.com/uc?id=17Ac0pO5oryh1JwgwTFQTjOKHY3umbDQu -O ${BASE_DIR}/pretrained_weights/wasb_badminton_best.pth.tar
wget https://drive.google.com/uc?id=1M9y4wPJqLc0K-z-Bo5DP8Ft5XwJuLqIS -O ${BASE_DIR}/pretrained_weights/wasb_volleyball_best.pth.tar
wget https://drive.google.com/uc?id=1nfECuSyJvPUmz3njZCdFERSQQbERt8FU -O ${BASE_DIR}/pretrained_weights/wasb_basketball_best.pth.tar


