#!/bin/bash

#anaconda 环境以及通过 anaconda 创建的名为 pytorch 的环境

#python 程序运行，需在.py 文件指定调用 GPU，并设置合适的线程数，batch_size 大小等
# ./tools/dist_train.sh ./configs/cascade_rcnn/cascade_mrcnn_x101.py 8

/data/code/mmdet/tools/dist_train.sh /data/code/mmdet/configs/convnext/cascade_convnext_x_2cls_full_exp5.py 8