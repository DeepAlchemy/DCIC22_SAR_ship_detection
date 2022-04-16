_base_ = './cascade_rcnn_r50_fpn_1x_coco.py'
model = dict(
    type='CascadeRCNN',
    pretrained='/data/home/scv1442/run/DCIC_2022_Swin-Transformer/pretrained/cascade_rcnn_x101_64x4d_fpn_1x_coco_20200515_075702-43ce6a30.pth',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'))
