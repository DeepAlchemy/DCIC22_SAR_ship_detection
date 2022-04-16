#FROM python:3.7

FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

RUN mkdir -p /home/release
RUN mkdir -p /home/release/data

COPY data /home/release/data
COPY images/run.sh /home/release/data/run.sh

# mmdet configuration
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX 8.0 8.6"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

#RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list

RUN sed -i s:/archive.ubuntu.com:/mirrors.tuna.tsinghua.edu.cn/ubuntu:g /etc/apt/sources.list
#RUN apt-get clean
#RUN apt-get -y update --fix-missing

RUN rm /etc/apt/sources.list.d/nvidia-ml.list && apt-get clean && apt-get update --fix-missing

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install --no-cache-dir --upgrade pip wheel setuptools
RUN pip install --no-cache-dir mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

# Install MMDetection
RUN conda clean --all

WORKDIR /home/release/data/code/mmdet
ENV FORCE_CUDA="1"
RUN pip install --no-cache-dir -r requirements/build.txt
RUN pip install --no-cache-dir -e .

# YOLOX configuration

WORKDIR /home/release/data/code/yolox

# install douban pip source, boost installation
#RUN mkdir ~/.pip && echo -e "[global]\nindex-url = https://pypi.doubanio.com/simple\ntrusted-host = pypi.doubanio.com\n" > ~/.pip/pip.conf

RUN pip install -r yolox_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && pip install cython 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install ffmpeg libsm6 libxext6  -y

RUN cd /home/release/data/code/mmdet/apex-master && pip install -v \
    --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

RUN cd /home/release/data/code/yolox && pip install -v -e .

WORKDIR /home/release/data