
fourier_degree=10
model = dict(
    type='FourierInst',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='FourierInstHead',
        num_classes=15,
        in_channels=256,
        stacked_convs=6,
        feat_channels=256,
        anchor_type='anchor_free',
        fourier_degree=fourier_degree,
        sample_num = 72,
        reg_format='mask',
        show_result = False,
        point_loss=False,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        initial_loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            beta=2.0,
            loss_weight=1.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=0.5),
            loss_mask=dict(type='CrossIOULoss',   loss_weight=2.0, loss_type='polygon', stride=9),
            # loss_mask=dict(type='SmoothL1Loss', loss_weight=0.5),
        ),
    train_cfg=dict(
        initial_epoch=4,#change
        initial_assigner=dict(type='ATSSAssigner', topk=9),
        assigner=dict(type='TaskAlignedAssigner', topk=13),
        alpha=1,
        beta=6,
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        # nms=dict(type='nms', iou_threshold=0.6),
        nms=dict(type='soft_nms', iou_thr=0.6, min_score=0.001),
        max_per_img=1500))


# dataset settings
dataset_type = 'IsaidDataset'
data_root = '/media/zf/E/Dataset/iSAID_1024/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True,poly2mask=False),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='RandomRotatePolyInstances',
        fix_size=True,
        rotate_ratio=0.8,
        max_angle=180,),
    dict(
        type='FCETargets',
        fourier_degree= fourier_degree,
        debug = False,
        num_reconstr_points=256,
       ),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks','gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        # img_scale=[(896, 896), (1024, 1024), (1152, 1152)],
        # flip=True,
        # flip_direction=['horizontal', 'vertical', 'diagonal'],
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=5,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))


# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=3)

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from ='/media/zf/E/checkpoint/tood_r50_fpn_mstrain_2x_coco_20211210_144231-3b23174c.pth'
resume_from = None#'/media/zf/E/FourMask/work_dirs/fourmask6_r50_fpn_1x_isaid_crossiou+giou05/latest.pth'

workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# custom hooks
custom_hooks = [dict(type='SetEpochInfoHook')] 
# evaluation
# evaluation = dict(interval=1, metric='mAP')
evaluation = dict(interval=6, classwise=True,proposal_nums=[1000,1000,1000],metric=['bbox','segm'])
# fp16 = dict(loss_scale='dynamic')
fp16 = dict(loss_scale=512.)