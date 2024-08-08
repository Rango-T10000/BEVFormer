#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-28509}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic


#文件名dist_train.sh: dist是分布式

#终端运行时输入的命令：./tools/dist_train.sh ./projects/configs/bevformer/bevformer_base.py 8
#CONFIG=$1：将命令中传递给脚本的第一个参数（即配置文件的路径）赋值给变量 CONFIG。
#GPUS=$2：将命令中传递给脚本的第二个参数（GPU 数量）赋值给变量 GPUS。
#PORT=${PORT:-28509}：如果变量 PORT 未定义，则设置默认端口号为 28509。这个端口号将用于分布式训练中的通信。

#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#--设置 Python 模块搜索路径（PYTHONPATH），将当前脚本所在目录的父目录添加到搜索路径中，以使 Python 能够找到相关的模块和库

#python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#--运行 Python 解释器并使用 torch.distributed.launch 模块来启动分布式训练
#----nproc_per_node=$GPUS 指定每个节点（每台机器）使用的 GPU 数量
#----master_port=$PORT 指定主节点的通信端口。

#$(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic
#--$(dirname "$0")/train.py：运行一个名为 train.py 的 Python 脚本，该脚本用于训练模型。
#把一下参数传给train.py:
#-----------------$CONFIG：指定配置文件的路径，用于定义模型、数据集等训练参数。
#-----------------launcher pytorch：指定使用 PyTorch 启动器
#-----------------${@:3}：将传递给脚本的第三个参数及之后的所有参数传递给 train.py 脚本。
#-----------------deterministic：启用确定性训练，以确保训练过程的可重现性
