#!/bin/bash

# run_1.sh + reference.sh 所有模型执行一次
############################
# 编译yolox代码
# cd /data/code/yolox
# pip install -v -e .
# sh /data/code/yolox/run_x_1.sh
# sh /data/code/yolox/inference_1.sh
############################

# 编译mmdet的apex 与安装mmpycocotools
# cd /data/code/mmdet/apex-master/
# pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# cd /data/code/mmdet/
# pip uninstall pycocotools
# pip install mmpycocotools-12.0.3.tar.gz
############################
# sh /data/code/mmdet/run_7.sh
# sh /data/code/mmdet/test_7.sh

# sh /data/code/mmdet/run_8.sh
# sh /data/code/mmdet/test_8.sh
############################

# 编译yolox代码
# cd /data/code/yolox
# pip install -v -e .
############################
# sh /data/code/yolox/run_x_2.sh
# sh /data/code/yolox/inference_2.sh
############################

# 编译mmdet的apex 与安装mmpycocotools
cd /data/code/mmdet/apex-master/
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd /data/code/mmdet/
pip uninstall pycocotools
pip install mmpycocotools-12.0.3.tar.gz
############################
sh /data/code/mmdet/run_1.sh
sh /data/code/mmdet/test_1.sh

sh /data/code/mmdet/run_5.sh
sh /data/code/mmdet/test_5.sh

sh /data/code/mmdet/run_2.sh
sh /data/code/mmdet/test_2.sh

sh /data/code/mmdet/run_6.sh
sh /data/code/mmdet/test_6.sh

sh /data/code/mmdet/run_3.sh
sh /data/code/mmdet/test_3.sh

sh /data/code/mmdet/run_4.sh
sh /data/code/mmdet/test_4.sh
############################

pip install pandas
pip install ensemble-boxes
# 将所有的 conv-json 转换为txt格式保存至output
python /data/code/utils/coco2yolo.py

# 将所有 modele 使用wbf做 ensemble
python /data/code/utils/wbf_test.py

# 后处理， 将wbf_test的结果输出
python /data/code/utils/post_process.py


