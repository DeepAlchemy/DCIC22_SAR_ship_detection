_base_ = './cascade_mask_rcnn_r50_fpn_1x_coco.py'
model = dict(
    pretrained='/data/home/scv1442/run/DCIC_2022_Swin-Transformer/pretrained/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco_20200203-9a2db89d.pth',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'),
        # dcn=dict(type='DCN', groups=64, deform_groups=1, fallback_on_stride=False),
        # stage_with_dcn=(False, True, True, True)
        )
