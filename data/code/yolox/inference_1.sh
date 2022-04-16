#!/bin/bash

#加载环境，此处加载 anaconda 环境以及通过 anaconda 创建的名为 pytorch 的环境
#module load anaconda/2020.11
#
#source activate pytorch

#python 程序运行，需在.py 文件指定调用 GPU，并设置合适的线程数，batch_size 大小等
# python tools/demo.py -n yolox-m -f exps/yolox_m.py -c /data/home/scv1442/run/DCIC2022_YOLOX/YOLOX_outputs/yolox_m/best_ckpt.pth -b 32 -d 1
python /data/code/yolox/tools/demo.py image -expn yolox_x_2cls_230epoch -f /data/code/yolox/exps/yolox_x_exp1.py -c /home/release/data/YOLOX_outputs/yolox_x_2cls_230epoch/latest_ckpt.pth --path /data/user_data/dcic_coco_2cls_dataset/test --conf 0.5 --nms 0.65 --tsize 640 --save_result --device gpu
