# Get Started

## Data Preparation

Download [Soccer](https://pspagnolo.jimdofree.com/download/), [Tennis](https://nol.cs.nctu.edu.tw:234/open-source/TrackNet), [Badminton](https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2), [Volleyball](https://github.com/mostafa-saad/deep-activity-rec), and [Basketball](https://ruiyan1995.github.io/SAM.html) and put them under ```<WASB-SBDT_HOME>/datasets```.

- For Soccer, we provide a setup script. Run ```cd ./src && sh setup_scripts/setup_soccer.sh``` .
- For Tennis, download ```Dataset.zip``` from [this link](https://nycu1-my.sharepoint.com/:u:/g/personal/tik_m365_nycu_edu_tw/ETCr6-M0e1VDhGCdMbvljcsBu31AJTO5xa_1cW8pHa7niA?e=55tLJ9) and put it under ```<WASB-SBDT_HOME>``` directory. Then unzip the file and rename ```Dataset``` directory as ```tennis``` .
- For Badminton, download ```TrackNetV2.zip``` from [this link](https://nycu1-my.sharepoint.com/:u:/g/personal/tik_m365_nycu_edu_tw/EWisYhAiai9Ju7L-tQp0ykEBZJd9VQkKqsFrjcqqYIDP-g?e=S0AB1Z) and put it under ```<WASB-SBDT_HOME>``` directory. Then use a setup script by running ```cd ./src && sh setup_scripts/setup_badminton.sh``` .
- For Volleyball, download ```volleyball_.zip``` from [this link](https://drive.google.com/drive/folders/1rmsrG1mgkwxOKhsr-QYoi9Ss92wQmCOS?usp=sharing) and ```volleyball_ball_annotations.zip``` from [this link](https://drive.google.com/file/d/1urZpZiiepC85JD1u3VeURgUpztRgI0yl/edit) and unzip them. Then put the resulting directories as follows.
- For Basketball, download all the zip segments shown [here](https://ruiyan1995.github.io/SAM.html) and unzip them to generate ```NBA_data```. Then run a setup script i.e., ```cd ./src && sh setup_scripts/setup_basketball.sh``` .

Data structure should be the following:

```
    datasets
    |-----soccer
    |        └-----videos
    |        └-----frames
    |        └-----annos
    └-----tennis /* renamed from Dataset */
    |        └-----game1
    |        └-----...
    |        └-----game10
    └-----badminton 
    |        └-----match1
    |        └-----...
    |        └-----match26
    |        └-----test_match1
    |        └-----...
    |        └-----test_match3
    └-----volleyball
    |        └-----videos
    |        └-----volleyball_ball_annotation
    └-----basketball /* renamed from NBA_data */
    |        └-----videos
    |        └-----ball-annos
    |
    src
```

## Model Preparation

Download pretrained models listed in [MODEL_ZOO.md](./MODEL_ZOO.md).

We also provide a setup script to download all the listed models at once. Run ```cd ./src && sh setup_scripts/setup_weights.sh```, then models are located in ```<WASB-SBDT_HOME>/pretrained_weights```.

## Evaluation 

Here we show the evaluation commands to reproduce the results of Table 2 and Table 3 in [our paper](https://arxiv.org/abs/2311.05237).

### [WASB (Ours, Step=3)](https://arxiv.org/abs/2311.05237)

```
# Soccer
python3 main.py --config-name=eval dataset=soccer model=wasb detector.model_path=../pretrained_weights/wasb_soccer_best.pth.tar

# Tennis
python3 main.py --config-name=eval dataset=tennis model=wasb detector.model_path=../pretrained_weights/wasb_tennis_best.pth.tar

# Badminton
python3 main.py --config-name=eval dataset=badminton model=wasb detector.model_path=../pretrained_weights/wasb_badminton_best.pth.tar

# Volleyball
python3 main.py --config-name=eval dataset=volleyball model=wasb detector.model_path=../pretrained_weights/wasb_volleyball_best.pth.tar

# Basketball
python3 main.py --config-name=eval dataset=basketball model=wasb detector.model_path=../pretrained_weights/wasb_basketball_best.pth.tar
```

### [WASB (Ours, Step=1)](https://arxiv.org/abs/2311.05237)

```
# Soccer
python3 main.py --config-name=eval dataset=soccer model=wasb detector.model_path=../pretrained_weights/wasb_soccer_best.pth.tar detector.step=1

# Tennis
python3 main.py --config-name=eval dataset=tennis model=wasb detector.model_path=../pretrained_weights/wasb_tennis_best.pth.tar detector.step=1

# Badminton
python3 main.py --config-name=eval dataset=badminton model=wasb detector.model_path=../pretrained_weights/wasb_badminton_best.pth.tar detector.step=1

# Volleyball
python3 main.py --config-name=eval dataset=volleyball model=wasb detector.model_path=../pretrained_weights/wasb_volleyball_best.pth.tar detector.step=1

# Basketball
python3 main.py --config-name=eval dataset=basketball model=wasb detector.model_path=../pretrained_weights/wasb_basketball_best.pth.tar detector.step=1
```

### [MonoTrack [CVPRW2022]](https://ieeexplore.ieee.org/document/9857202)

```
# Soccer
python3 main.py --config-name=eval dataset=soccer model=monotrack detector.postprocessor.use_hm_weight=False detector.model_path=../pretrained_weights/monotrack_soccer_best.pth.tar tracker=intra_frame_peak

# Tennis
python3 main.py --config-name=eval dataset=tennis model=monotrack detector.postprocessor.use_hm_weight=False detector.model_path=../pretrained_weights/monotrack_tennis_best.pth.tar tracker=intra_frame_peak

# Badminton
python3 main.py --config-name=eval dataset=badminton model=monotrack detector.postprocessor.use_hm_weight=False detector.model_path=../pretrained_weights/monotrack_badminton_best.pth.tar tracker=intra_frame_peak

# Volleyball
python3 main.py --config-name=eval dataset=volleyball model=monotrack detector.postprocessor.use_hm_weight=False detector.model_path=../pretrained_weights/monotrack_volleyball_best.pth.tar tracker=intra_frame_peak

# Basketball
python3 main.py --config-name=eval dataset=basketball model=monotrack detector.postprocessor.use_hm_weight=False detector.model_path=../pretrained_weights/monotrack_basketball_best.pth.tar tracker=intra_frame_peak
```

### [ResTrackNetV2](https://arxiv.org/abs/2311.05237)

```
# Soccer
python3 main.py --config-name=eval dataset=soccer model=restracknetv2 detector.postprocessor.use_hm_weight=False detector.model_path=../pretrained_weights/restracknetv2_soccer_best.pth.tar tracker=intra_frame_peak

# Tennis
python3 main.py --config-name=eval dataset=tennis model=restracknetv2 detector.postprocessor.use_hm_weight=False detector.model_path=../pretrained_weights/restracknetv2_tennis_best.pth.tar tracker=intra_frame_peak

# Badminton
python3 main.py --config-name=eval dataset=badminton model=restracknetv2 detector.postprocessor.use_hm_weight=False detector.model_path=../pretrained_weights/restracknetv2_badminton_best.pth.tar tracker=intra_frame_peak

# Volleyball
python3 main.py --config-name=eval dataset=volleyball model=restracknetv2 detector.postprocessor.use_hm_weight=False detector.model_path=../pretrained_weights/restracknetv2_volleyball_best.pth.tar tracker=intra_frame_peak

# Basketball
python3 main.py --config-name=eval dataset=basketball model=restracknetv2 detector.postprocessor.use_hm_weight=False detector.model_path=../pretrained_weights/restracknetv2_basketball_best.pth.tar tracker=intra_frame_peak
```

### [TrackNetV2 [ICPAI2020]](https://ieeexplore.ieee.org/document/9302757)

```
# Soccer
python3 main.py --config-name=eval dataset=soccer model=tracknetv2 detector.postprocessor.use_hm_weight=False detector.model_path=../pretrained_weights/tracknetv2_soccer_best.pth.tar tracker=intra_frame_peak

# Tennis
python3 main.py --config-name=eval dataset=tennis model=tracknetv2 detector.postprocessor.use_hm_weight=False detector.model_path=../pretrained_weights/tracknetv2_tennis_best.pth.tar tracker=intra_frame_peak

# Badminton
python3 main.py --config-name=eval dataset=badminton model=tracknetv2 detector.postprocessor.use_hm_weight=False detector.model_path=../pretrained_weights/tracknetv2_badminton_best.pth.tar tracker=intra_frame_peak

# Volleyball
python3 main.py --config-name=eval dataset=volleyball model=tracknetv2 detector.postprocessor.use_hm_weight=False detector.model_path=../pretrained_weights/tracknetv2_volleyball_best.pth.tar tracker=intra_frame_peak

# Basketball
python3 main.py --config-name=eval dataset=basketball model=tracknetv2 detector.postprocessor.use_hm_weight=False detector.model_path=../pretrained_weights/tracknetv2_basketball_best.pth.tar tracker=intra_frame_peak
```

### [BallSeg [MMSports2019]](https://arxiv.org/abs/2007.11876)

```
# Soccer
python3 main.py --config-name=eval dataset=soccer model=ballseg detector.step=1 detector.postprocessor.use_hm_weight=False detector.model_path=../pretrained_weights/ballseg_soccer_best.pth.tar tracker=intra_frame_peak

# Tennis
python3 main.py --config-name=eval dataset=tennis model=ballseg detector.step=1 detector.postprocessor.use_hm_weight=False detector.model_path=../pretrained_weights/ballseg_tennis_best.pth.tar tracker=intra_frame_peak

# Badminton
python3 main.py --config-name=eval dataset=badminton model=ballseg detector.step=1 detector.postprocessor.use_hm_weight=False detector.model_path=../pretrained_weights/ballseg_badminton_best.pth.tar tracker=intra_frame_peak

# Volleyball
python3 main.py --config-name=eval dataset=volleyball model=ballseg detector.step=1 detector.postprocessor.use_hm_weight=False detector.model_path=../pretrained_weights/ballseg_volleyball_best.pth.tar tracker=intra_frame_peak

# Basketball
python3 main.py --config-name=eval dataset=basketball model=ballseg detector.step=1 detector.postprocessor.use_hm_weight=False detector.model_path=../pretrained_weights/ballseg_basketball_best.pth.tar tracker=intra_frame_peak
```

### [DeepBall [VISAPP2019]](https://arxiv.org/abs/1902.07304)

```
# Soccer
python3 main.py --config-name=eval dataset=soccer model=deepball detector=deepball detector.model_path=../pretrained_weights/deepball_soccer_best.pth.tar detector.step=1 tracker=intra_frame_peak

# Tennis
python3 main.py --config-name=eval dataset=tennis model=deepball detector=deepball detector.model_path=../pretrained_weights/deepball_tennis_best.pth.tar detector.step=1 tracker=intra_frame_peak

# Badminton
python3 main.py --config-name=eval dataset=badminton model=deepball detector=deepball detector.model_path=../pretrained_weights/deepball_badminton_best.pth.tar detector.step=1 tracker=intra_frame_peak

# Volleyball
python3 main.py --config-name=eval dataset=volleyball model=deepball detector=deepball detector.model_path=../pretrained_weights/deepball_volleyball_best.pth.tar detector.step=1 tracker=intra_frame_peak

# Basketball
python3 main.py --config-name=eval dataset=basketball model=deepball detector=deepball detector.model_path=../pretrained_weights/deepball_basketball_best.pth.tar detector.step=1 tracker=intra_frame_peak
```

### [DeepBall-Large](https://arxiv.org/abs/2311.05237)

```
# Soccer
python3 main.py --config-name=eval dataset=soccer model=deepball_large detector=deepball detector.model_path=../pretrained_weights/deepball-large_soccer_best.pth.tar detector.step=1 tracker=intra_frame_peak

# Tennis
python3 main.py --config-name=eval dataset=tennis model=deepball_large detector=deepball detector.model_path=../pretrained_weights/deepball-large_tennis_best.pth.tar detector.step=1 tracker=intra_frame_peak

# Badminton
python3 main.py --config-name=eval dataset=badminton model=deepball_large detector=deepball detector.model_path=../pretrained_weights/deepball-large_badminton_best.pth.tar detector.step=1 tracker=intra_frame_peak

# Volleyball
python3 main.py --config-name=eval dataset=volleyball model=deepball_large detector=deepball detector.model_path=../pretrained_weights/deepball-large_volleyball_best.pth.tar detector.step=1 tracker=intra_frame_peak

# Basketball
python3 main.py --config-name=eval dataset=basketball model=deepball_large detector=deepball detector.model_path=../pretrained_weights/deepball-large_basketball_best.pth.tar detector.step=1 tracker=intra_frame_peak
```

## Training

TBA

