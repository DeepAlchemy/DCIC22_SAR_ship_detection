#!/bin/bash

#anaconda 环境以及通过 anaconda 创建的名为 pytorch 的环境
module load anaconda/2020.11

source activate openmmlab

#python 程序运行，需在.py 文件指定调用 GPU，并设置合适的线程数，batch_size 大小等
# python ./tools/test.py ./configs/swin/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py /data/home/scv1442/run/DCIC_2022_Swin-Transformer/work_dirs/swin_small_314/epoch_20.pth --format-only --options "jsonfile_prefix=./swin_small_314"

./tools/dist_test.sh ./configs/swin/swin_large.py /data/home/scv1442/run/DCIC_2022_Swin-Transformer/work_dirs/swin_large_18epoch/latest.pth 8 --format-only --options "jsonfile_prefix=./swin_large_18epoch"

# python ./tools/test.py ./configs/convnext/cascade_convnext_x_2cls.py /data/home/scv1442/run/DCIC_2022_Swin-Transformer/work_dirs/cascade_convnext_x_exp5_2cls/latest.pth --format-only --options "jsonfile_prefix=./test_2cls"