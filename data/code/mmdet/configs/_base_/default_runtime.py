checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = '/data/home/scv1442/run/DCIC_2022_Swin-Transformer/work_dirs/swin_small_314/latest.pth'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = '/data/home/scv1442/run/DCIC_2022_Swin-Transformer/work_dirs/swin_small_314_resume'
