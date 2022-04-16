#!/bin/bash

#anaconda 环境以及通过 anaconda 创建的名为 pytorch 的环境

#python 程序运行，需在.py 文件指定调用 GPU，并设置合适的线程数，batch_size 大小等
# python ./tools/test.py ./configs/swin/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py /data/home/scv1442/run/DCIC_2022_Swin-Transformer/work_dirs/swin_small_314/epoch_20.pth --format-only --options "jsonfile_prefix=./swin_small_314"

/data/code/mmdet/tools/dist_test.sh /data/code/mmdet/configs/convnext/cascade_convnext_x_2cls_full_aug_256_exp9.py /data/code/mmdet/work_dirs/cascade_convnext_x_2cls_full_aug_256_exp9/latest.pth 8 --format-only --options "jsonfile_prefix=/data/user_data/intermediate_results/conv_output/cascade_convnext_x_2cls_full_aug_256_exp9"

# python ./tools/test.py ./configs/convnext/cascade_convnext_x_2cls.py /data/home/scv1442/run/DCIC_2022_Swin-Transformer/work_dirs/cascade_convnext_x_exp5_2cls/latest.pth --format-only --options "jsonfile_prefix=./test_2cls"