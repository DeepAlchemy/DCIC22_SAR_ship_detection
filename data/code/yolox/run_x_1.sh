#!/bin/bash

#加载环境，此处加载 anaconda 环境以及通过 anaconda 创建的名为 pytorch 的环境

#python 程序运行，需在.py 文件指定调用 GPU，并设置合适的线程数，batch_size 大小等
python /data/code/yolox/tools/train.py --experiment-name yolox_x_2cls_230epoch -f /data/code/yolox/exps/yolox_x_exp1.py --devices 8 --batch-size 64 -o --ckpt /data/code/yolox/pretrained/yolox_x.pth
# python train.py --resume
# bs = 64/ 8 gpu
