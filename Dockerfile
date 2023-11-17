FROM nvcr.io/nvidia/tensorrt:21.06-py3

MAINTAINER shuhei.tarashima@ntt.com
LABEL OBJECT="Dockerfile for Widely Applicable Strong Baseline for Sports Ball Detection and Tracking [BMVC2023]"
ARG http_proxy
ARG https_proxy
ENV http_proxy ${http_proxy}
ENV https_proxy ${https_proxy}
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends\
	apt-utils\
	apt-transport-https\
	ca-certificates\
	software-properties-common\
&& apt-get clean\
&& rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y \
	git\
	vim\
	cmake\
	libsm6\ 
	libxext6\
	libxrender-dev\
	libgl1-mesa-glx\
	locales\
	build-essential\
	zlib1g-dev\
	libssl-dev\
	libbz2-dev\
	libreadline-dev\
	libsqlite3-dev\
	libhdf5-dev\
	libncursesw5-dev\
	libgdbm-dev\
	liblzma-dev\
	libdb-dev\
	libffi-dev\
	uuid-dev\
	tk-dev\
&& apt-get clean\
&& rm -rf /var/lib/apt/lists/*

ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8
RUN locale-gen en_US.UTF-8

RUN pip3 install numpy==1.22.4 torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install hydra-core==1.2.0 tqdm==4.64.0 opencv-python==4.6.0.66 scikit-learn==1.1.1 scikit-image==0.19.3 pandas==1.3.5 einops==0.4.1 timm==0.6.5 matplotlib==3.5.2

COPY src /root/src
WORKDIR /root/src
#COPY .vimrc /root/

