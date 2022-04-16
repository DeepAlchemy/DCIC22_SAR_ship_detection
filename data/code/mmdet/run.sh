#!/bin/bash

#anaconda 环境以及通过 anaconda 创建的名为 pytorch 的环境
module load anaconda/2020.11

source activate openmmlab

#python 程序运行，需在.py 文件指定调用 GPU，并设置合适的线程数，batch_size 大小等
# ./tools/dist_train.sh ./configs/cascade_rcnn/cascade_mrcnn_x101.py 8

./tools/dist_train.sh ./configs/swin/swin_large.py 8