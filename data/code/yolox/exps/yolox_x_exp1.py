#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        # self.data_dir = "/data/home/scv1442/run/DCIC_2022_Swin-Transformer/data/dcic_coco_dataset"
        # self.train_ann = "train.json"
        # self.val_ann = "val.json"
        # self.test_ann = "test.json"
        # 2cls dataset
        # self.data_dir = "/data/home/scv1442/run/DCIC2022_YOLOX/data/dcic_coco_dataset"
        self.data_dir = "/data/user_data/dcic_coco_2cls_dataset"
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        self.test_ann = "test.json"

        self.num_classes = 2

        self.max_epoch = 230
        self.data_num_workers = 4
        self.eval_interval = 1
        self.save_history_ckpt = False
        # prob of applying mosaic aug
        self.mosaic_prob = 1
        # prob of applying mixup aug
        self.mixup_prob = 1
        # last #epoch to close augmention like mosaic
        self.no_aug_epochs = 15


