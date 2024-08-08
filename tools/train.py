# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
 
from __future__ import division

import argparse
import copy
import mmcv
import os
import time
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
#from mmdet3d.apis import train_model

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version

from mmcv.utils import TORCH_VERSION, digit_version

#--------解析函数定义，用于解析命令行参数------------
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector') #创建了一个参数解析器

    #----------使用parser.add_argument()方法添加了多个命令行参数-------------
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    #------------------用parser.parse_args()解析命令行参数，并将结果存储在args中-------------
    args = parser.parse_args()

    #------------------设置环境变量：如果os.environ中不存在LOCAL_RANK，则将args.local_rank值赋给LOCAL_RANK环境变量-----
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    #--------解析函数------------
    #被用来解析用户在命令行中输入的参数，在运行dist_train.sh脚本是命令中传入的参数
    #调用 parse_args() 函数，将返回的解析结果存储在 args 变量中
    args = parse_args()

    #--------加载配置文件---------
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    #--------导入自定义模块---------
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    #--------导入插件模块-----------
    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

            from projects.mmdet3d_plugin.bevformer.apis.train import custom_train_model
    
    #--------设置GPU相关参数--------
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # set tf32
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    #--------设置工作目录------------
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    
    #--------如果指定了resume_from参数并且文件存在，则从指定的检查点文件中恢复训练状态--------
    # if args.resume_from is not None:
    if args.resume_from is not None and osp.isfile(args.resume_from):
        cfg.resume_from = args.resume_from
    
    #--------GPU IDs设置：用于指定程序在哪些GPU上运行-----------------------------------
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    
    #--------PyTorch版本和优化器类型检查--
    #--------检查当前使用的PyTorch版本是否为1.8.1，并且检查配置文件中的优化器类型是否为'AdamW'-
    if digit_version(TORCH_VERSION) == digit_version('1.8.1') and cfg.optimizer['type'] == 'AdamW':
        cfg.optimizer['type'] = 'AdamW2' # fix bug in Adamw
    
    #--------自动调整学习率--------------
    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        #设置为当前学习率乘以GPU数量除以8的结果。这种操作可以用于在分布式训练环境中自动调整学习率，以保持训练的稳定性和效率
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    #--------初始化分布式环境------------
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    #--------create work_dir：用来保存日志和模型的---------
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    #--------dump config：将配置文件保存在工作目录中，文件名与命令行参数中指定的配置文件名相同--
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    #--------init the logger before other steps：初始化日志记录器：---------------------
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level, name=logger_name)

    #-------------收集环境信息，并将其记录在日志中：系统信息、CUDA信息等_start--------------
    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()

    #----log env info----------
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    #----log some basic info----
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    #----set random seeds-------
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed

    #--将实验名称保存在meta字典的exp_name键中--
    #实验名称是根据命令行参数中配置文件路径的基本名称（去除路径部分）得到的
    meta['exp_name'] = osp.basename(args.config)
    #-------------收集环境信息，并将其记录在日志中：系统信息、CUDA信息等_end--------------

    #-------------模型构建：传入配置文件中指定的模型参数以及训练和测试配置-----------------
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    #-------------在日志中记录模型的详细信息，包括模型的结构和参数信息---------------------
    logger.info(f'Model:\n{model}')

    #-------------构建训练数据集-----------------------------------------------------
    datasets = [build_dataset(cfg.data.train)]
    
    #-------------若配置文件中的 workflow 长度是否为2，为模型准备验证数据集----------------
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    
    #-------------若配置文件中存在 checkpoint_config 的情况下，为模型的检查点配置添加一些元数据信息----
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,      #存储了配置文件的内容，通常是格式化后的文本
            CLASSES=datasets[0].CLASSES, #存储了数据集的类别信息
            PALETTE=datasets[0].PALETTE  #用于分割器（segmentors）的调色板信息
            if hasattr(datasets[0], 'PALETTE') else None)
    
    #-------------add an attribute for visualization convenience-------------------
    model.CLASSES = datasets[0].CLASSES

    #-------------执行模型的训练过程，使用指定的数据集、配置参数和其他信息来训练模型----------
    #/BEVFormer/projects/mmdet3d_plugin/bevformer/apis/train.py中定义的函数
    custom_train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
