# BEvFormer-tiny consumes at lease 6700M GPU memory
# compared to bevformer_base, bevformer_tiny has
# smaller backbone: R101-DCN -> R50
# smaller BEV: 200*200 -> 50*50
# less encoder layers: 6 -> 3
# smaller input size: 1600*900 -> 800*450
# multi-scale feautres -> single scale features (C5)

#-----------------------配置 Bevformer_tiny 模型在 Nuscenes 数据集上的训练设置_start-----------------------

#_base_ 是一个列表，包含了其他配置文件的相对路径
_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
#用于指定是否启用插件以及插件的目录路径
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

#定义点云的范围和voxel大小
# If point cloud range is changed, the models should also change their point cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

#一个字典，包含了图像的归一化参数，包括均值、标准差和是否转换为 RGB 格式
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

#包含了数据集中物体类别的列表
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

#一个字典，指定了输入的模态是否被使用
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
#-----------------------配置 Bevformer_tiny 模型在 Nuscenes 数据集上的训练设置_end-----------------------


_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1
bev_h_ = 50
bev_w_ = 50
queue_length = 3 # each sequence contains `queue_length` frames.

#-----------------------模型配置定义-------------------------
model = dict(
    type='BEVFormer',
    use_grid_mask=True,
    video_test_mode=True,
    pretrained=dict(img='torchvision://resnet50'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='BEVFormerHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_classes=10,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='PerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=3,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],

                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range))))

#---------------------为数据集加载和处理提供必要的配置信息-------------------
dataset_type = 'CustomNuScenesDataset'
data_root = '/home2/wzc/python_project/Uniad_related/BEVFormer/data/nuscenes/' #------------config从这里读到的data_root = 'data/nuscenes/'  ----------------
file_client_args = dict(backend='disk')


#---------------------train数据处理流程的配置----------------------
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),                                      #从文件加载多视角图像并将其转换为float32类型
    dict(type='PhotoMetricDistortionMultiViewImage'),                                               #对多视角图像进行光度失真处理
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),   #加载带有 3D 边界框和标签的注释信息
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),                            #过滤超出指定点云范围的对象
    dict(type='ObjectNameFilter', classes=class_names),                                             #根据指定的类别名称（class_names）过滤对象
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),                                           #对多视角图像进行归一化处理
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),                                      #对多视角图像进行随机缩放
    dict(type='PadMultiViewImage', size_divisor=32),                                                #对多视角图像进行填充，使其尺寸能够被 32 整除
    dict(type='DefaultFormatBundle3D', class_names=class_names),                                    #将处理后的数据格式化为模型需要的默认格式，包括指定的类别名称
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])                      #自定义收集步骤，指定了需要收集的键为
]


#---------------------test数据处理流程的配置----------------------
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),              #从文件加载多视角图像并将其转换为 float32 类型
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),                   #对多视角图像进行归一化处理
    dict(
        type='MultiScaleFlipAug3D',                                         #多尺度翻转增强，该步骤包含以下子步骤：
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),      #对多视角图像进行随机缩放
            dict(type='PadMultiViewImage', size_divisor=32),                #对多视角图像进行填充，使其尺寸能够被 32 整除
            dict(
                type='DefaultFormatBundle3D',                               #将处理后的数据格式化为模型需要的默认格式，包括指定的类别名称，并且不包括标签信息
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['img'])                      #自定义收集步骤，指定了需要收集的键为 
        ])
]

#---------------------------训练、验证和测试数据的相关配置信息-------------------------------
data = dict(
    samples_per_gpu=1,  #每张卡的batch_size
    workers_per_gpu=1,  #每张卡用来加载数据的进程数，原本是4
    #-----------------------训练数据配置-------------------------
    train=dict(
        type=dataset_type,                                              #数据集类型（上面定义的）
        data_root=data_root,                                            #数据集根目录路径（上面定义的）
        ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',       #label文件路径，这里面是用Lidar数据的标记的groundtruth，即label
        pipeline=train_pipeline,                                        #数据处理流程（上面定义的）
        classes=class_names,                                            #数据集中物体类别的列表（上面定义的）
        modality=input_modality,                                        #指定了输入的模态（上面定义的）
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    #-----------------------验证数据配置-------------------------
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
             pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
             classes=class_names, modality=input_modality, samples_per_gpu=1),
    #-----------------------测试数据配置-------------------------
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              classes=class_names, modality=input_modality),
    #-----------------------采样器配置---------------------------
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

#------------------优化器配置--------------------
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

#-----------------learning policy，学习率策略配置------------------
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
#-----------------总的训练迭代次数epoch-----------------
total_epochs = 24

evaluation = dict(interval=1, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

#-----------------日志配置-------------------
log_config = dict(
    interval=50,  #记录日志的间隔为 50 个 iterations
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

#-----------------模型保存配置------------------
checkpoint_config = dict(interval=1)  #即每个 epoch 结束后保存一次模型参数
