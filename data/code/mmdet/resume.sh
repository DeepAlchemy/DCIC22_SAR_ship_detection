#!/bin/bash

#anaconda 环境以及通过 anaconda 创建的名为 pytorch 的环境
module load anaconda/2020.11

source activate openmmlab

#python 程序运行，需在.py 文件指定调用 GPU，并设置合适的线程数，batch_size 大小等
./tools/dist_train.sh ./configs/convnext/cascade_convnext_x_2cls_aug_256.py 8 --resume-from /data/home/scv1442/run/DCIC_2022_Swin-Transformer/work_dirs/cascade_convnext_x_exp9_aug/latest.pth
