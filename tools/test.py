# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import argparse
import mmcv
import os
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp

#--------解析函数定义，用于解析命令行参数------------
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')  #参数前面有标记--代表可填可不填
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
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
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    #------------------用parser.parse_args()解析命令行参数，并将结果存储在args中-------------
    args = parser.parse_args()

    #------------------设置环境变量：如果os.environ中不存在LOCAL_RANK，则将args.local_rank值赋给LOCAL_RANK环境变量-----
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    #--------解析函数定义，用于解析命令行参数------------
    args = parse_args()

    #--------assert语句，至少要指定一个操作（保存/评估/仅格式化/显示结果/保存结果----
    # 通过参数 --out, --eval, --format-only, --show 或 --show-dir（dist_test.sh文件中写了）
    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')
    #--------异常情况处理----------------------------
    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')
    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

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

    #--------设置GPU相关参数--------
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # set tf32
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    #--------模型在测试阶段不会使用预训练的权重-----
    cfg.model.pretrained = None

    #--------处理测试数据集在连接(concatenated)的情况下的配置--
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    #--------初始化分布式环境------------------
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    #--------设置随机种子---------------------
    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)


    #--------build the dataloader-----------
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed, 
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

 
    #----------------------------build the model and load checkpoint------------------------
    cfg.model.train_cfg = None #表明在测试阶段不会使用任何训练配置
    
    #--------构建模型----------
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg')) #使用配置文件中的 cfg.model，以及使用测试对应的配置信息
    
    #--------若配置文件中存在 fp16 配置，将其提取为 fp16_cfg--------
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    
    #--------load checkpoint，根据配置信息加载模型参数-------------
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    #--------若命令行参数 args.fuse_conv_bn 为真，对模型进行卷积和批归一化的融合--
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    #--------处理模型类别信息和调色板信息----------
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    #--------根据是否在分布式环境下运行，选择了不同的模型并行化策略；这里才是运行model------
    if not distributed:
        # assert False
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

    #--------基于当前进程的 rank 值，对主进程（rank == 0）进行特定操作---------
    rank, _ = get_dist_info() #获取分布式信息
    if rank == 0:             #rank 为 0，表示当前进程是主进程
        #----若参数配置为--out----
        if args.out:
            print(f'\nwriting results to {args.out}')
            assert False
            #mmcv.dump(outputs['bbox_results'], args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
            '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
        
        #----若参数配置为--format_only----
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        
        #----若参数配置为--eval----
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy() #复制配置中的评估参数到 eval_kwargs 中
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None) #从 eval_kwargs 中移除这些键对应的值
            #------更新评估参数-------
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            #------执行评估操作,打印结果----
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
