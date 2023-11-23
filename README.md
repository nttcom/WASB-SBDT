# WASB: Widely Applicable Strong Baseline for Sports Ball Detection and Tracking

Code & dataset repository for the paper: **[Widely Applicable Strong Baseline for Sports Ball Detection and Tracking](https://arxiv.org/abs/2311.05237)**

Shuhei Tarashima, Muhammad Abdul Haq, Yushan Wang, Norio Tagawa

[![arXiv](https://img.shields.io/badge/arXiv-2311.05237-00ff00.svg)](https://arxiv.org/abs/2311.05237) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![test](https://img.shields.io/static/v1?label=By&message=Pytorch&color=red)

We present Widely Applicable Strong Baseline (WASB), a Sports Ball Detection and Tracking (SBDT) baseline that can be applied to wide range of sports categories :soccer: :tennis: :badminton: :volleyball: :basketball: .

https://github.com/nttcom/WASB-SBDT/assets/63090948/8889ef53-62c7-4c97-9b33-8bf386489ba1

## News

- [11/23/2023] [Our BMVC2023 proceeding](https://proceedings.bmvc2023.org/310/) is available! Thank you, BMVC2023 organizers!
- [11/23/2023] Evaluation codes of DeepBall, DeepBall-Large and BallSeg are added!
- [11/21/2023] Evaluation codes of TrackNetV2, ResTrackNetV2 and MonoTrack are added!
- [11/17/2023] Repository is released. Now it contains evaluation codes of pretrained WASB models only. Other models will be coming soon!
- [11/09/2023] Our [arXiv preprint](https://arxiv.org/abs/2311.05237) is released.

## Installation and Setup

Tested with Python3.8, CUDA11.3 on Ubuntu 18.04 (4 V100 GPUs inside). We recommend to use the [Dockerfile](./Dockerfile) provided in this repo (with ```-it``` option when running the container). 

- See [GET_STARTED.md](./GET_STARTED.md) for how to get started with SBDT models.
- See [MODEL_ZOO.md](./MODEL_ZOO.md) for available model weights.

## Citation

If you find this work useful, please consider to cite our paper:

```
@inproceedings{tarashima2023wasb,
	title={Widely Applicable Strong Baseline for Sports Ball Detection and Tracking},
	author={Tarashima, Shuhei and Haq, Muhammad Abdul and Wang, Yushan and Tagawa, Norio},
	booktitle={BMVC},
	year={2023}
}
```


