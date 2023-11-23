#!/bin/bash
SCRIPT_DIR=$(cd $(dirname $0); pwd)
BASE_DIR=$SCRIPT_DIR/../..

mkdir -p ${BASE_DIR}/pretrained_weights

# deepball
wget https://drive.google.com/uc?id=1u-Y-RnIgu3H7bZCvv9JVQGkYsBRx0XS9 -O ${BASE_DIR}/pretrained_weights/deepball_soccer_best.pth.tar
wget https://drive.google.com/uc?id=1h5v5_kUw4c_4Uw7tpKSBSh4bJwlWm8ly -O ${BASE_DIR}/pretrained_weights/deepball_tennis_best.pth.tar
wget https://drive.google.com/uc?id=1b-D2lrNuUNhZ9OPaLSrIjvU86y0OawFI -O ${BASE_DIR}/pretrained_weights/deepball_badminton_best.pth.tar
wget https://drive.google.com/uc?id=1mSWsewoBVAG-EdyypkDk1lhIK915wGmo -O ${BASE_DIR}/pretrained_weights/deepball_volleyball_best.pth.tar
wget https://drive.google.com/uc?id=1fUIVCzcKhfY5UiACF5T8mOpkr1x-Gu5C -O ${BASE_DIR}/pretrained_weights/deepball_basketball_best.pth.tar

# deepball-large
wget https://drive.google.com/uc?id=1-ak_vi8BiY9FoFVccxpYaJI_qJ6RjOxW -O ${BASE_DIR}/pretrained_weights/deepball-large_soccer_best.pth.tar
wget https://drive.google.com/uc?id=1hpCH6TM1EvoWmu5JZAkthz141DLgxrgn -O ${BASE_DIR}/pretrained_weights/deepball-large_tennis_best.pth.tar
wget https://drive.google.com/uc?id=1baHb1dtKHkXwQdGH8PB21hbdNVOu1dT1 -O ${BASE_DIR}/pretrained_weights/deepball-large_badminton_best.pth.tar
wget https://drive.google.com/uc?id=1tBLqbiokWpO6Xgh_fsNi-1kn0x6qhOHO -O ${BASE_DIR}/pretrained_weights/deepball-large_volleyball_best.pth.tar
wget https://drive.google.com/uc?id=11ENjf5sAFkw6-jCGSXGfyU0t35axWdGH -O ${BASE_DIR}/pretrained_weights/deepball-large_basketball_best.pth.tar

# ballseg
wget https://drive.google.com/uc?id=1wbuxL-bKpG-OGVYa7BBeWzmzcHKKIAl5 -O ${BASE_DIR}/pretrained_weights/ballseg_soccer_best.pth.tar
wget https://drive.google.com/uc?id=14YEAcxwUUgvaL3km07fPSduJLBBNuk_D -O ${BASE_DIR}/pretrained_weights/ballseg_tennis_best.pth.tar
wget https://drive.google.com/uc?id=1zey7rEX8orvpT0c1qbD5FsQ93jfBmaMl -O ${BASE_DIR}/pretrained_weights/ballseg_badminton_best.pth.tar
wget https://drive.google.com/uc?id=1FLJVwsupt5FXnPxMu41vPDWwIQBtU9d1 -O ${BASE_DIR}/pretrained_weights/ballseg_volleyball_best.pth.tar
wget https://drive.google.com/uc?id=1dTV-udhhO4K0651yRd1tfLhLMjh9uuBM -O ${BASE_DIR}/pretrained_weights/ballseg_basketball_best.pth.tar

# tracknetv2
wget https://drive.google.com/uc?id=1yTCC_8cWnMCLOpErk6UpepzQD9UF8Wpi -O ${BASE_DIR}/pretrained_weights/tracknetv2_soccer_best.pth.tar
wget https://drive.google.com/uc?id=1BCnmvDX-LZpbkk4vlMEXMm-uzCoqJzDx -O ${BASE_DIR}/pretrained_weights/tracknetv2_tennis_best.pth.tar
wget https://drive.google.com/uc?id=1lCVYzua7jJfuKqvWGypkYqr6PHA1EaPq -O ${BASE_DIR}/pretrained_weights/tracknetv2_badminton_best.pth.tar
wget https://drive.google.com/uc?id=103jOdYp4k20avid4uyB9USCuwiphI4Kz -O ${BASE_DIR}/pretrained_weights/tracknetv2_volleyball_best.pth.tar
wget https://drive.google.com/uc?id=1n-R_T0QyENsArYV8Qn8TneEroQkmYbT6 -O ${BASE_DIR}/pretrained_weights/tracknetv2_basketball_best.pth.tar

# restracknetv2
wget https://drive.google.com/uc?id=150_sbSmOXRCMDEJvsIeT7NFURJ_nucTB -O ${BASE_DIR}/pretrained_weights/restracknetv2_soccer_best.pth.tar
wget https://drive.google.com/uc?id=112qHZpPWgqCeZbbFXC0oRulrVsokzWYZ -O ${BASE_DIR}/pretrained_weights/restracknetv2_tennis_best.pth.tar
wget https://drive.google.com/uc?id=1NUeuFp1xKxLzvGR5RHSFhrI3LvzKbaFo -O ${BASE_DIR}/pretrained_weights/restracknetv2_badminton_best.pth.tar
wget https://drive.google.com/uc?id=1gnd-VbUUGmiB_Obn7WHpmTuhd0zARN5c -O ${BASE_DIR}/pretrained_weights/restracknetv2_volleyball_best.pth.tar
wget https://drive.google.com/uc?id=1K0kxE8vdjnLZm7dzGvwVE-nDuzQWn0t9 -O ${BASE_DIR}/pretrained_weights/restracknetv2_basketball_best.pth.tar

# monotrack
wget https://drive.google.com/uc?id=1PBVMfjqLFiUN9M_4NkiAW0i_2alEk_vH -O ${BASE_DIR}/pretrained_weights/monotrack_soccer_best.pth.tar
wget https://drive.google.com/uc?id=1mC3yWf6ySlzF-1d-s_LNH1QWexHz2M3D -O ${BASE_DIR}/pretrained_weights/monotrack_tennis_best.pth.tar
wget https://drive.google.com/uc?id=1b7hDHU6q7HarBOCVSx46_Rurtn1J1Ko9 -O ${BASE_DIR}/pretrained_weights/monotrack_badminton_best.pth.tar
wget https://drive.google.com/uc?id=15dNX0oV_YiP7u2SbcyN5b2nNtMwVt3TT -O ${BASE_DIR}/pretrained_weights/monotrack_volleyball_best.pth.tar
wget https://drive.google.com/uc?id=1uM2FJLG11AtC0fHsurOqBBUuTehRJugs -O ${BASE_DIR}/pretrained_weights/monotrack_basketball_best.pth.tar

# wasb
wget https://drive.google.com/uc?id=1pg0MpMtKZ6ziYEr4oyfKYPOO3hjLw94l -O ${BASE_DIR}/pretrained_weights/wasb_soccer_best.pth.tar
wget https://drive.google.com/uc?id=14AeyIOCQ2UaQmbZLNQJa1H_eSwxUXk7z -O ${BASE_DIR}/pretrained_weights/wasb_tennis_best.pth.tar
wget https://drive.google.com/uc?id=17Ac0pO5oryh1JwgwTFQTjOKHY3umbDQu -O ${BASE_DIR}/pretrained_weights/wasb_badminton_best.pth.tar
wget https://drive.google.com/uc?id=1M9y4wPJqLc0K-z-Bo5DP8Ft5XwJuLqIS -O ${BASE_DIR}/pretrained_weights/wasb_volleyball_best.pth.tar
wget https://drive.google.com/uc?id=1nfECuSyJvPUmz3njZCdFERSQQbERt8FU -O ${BASE_DIR}/pretrained_weights/wasb_basketball_best.pth.tar

