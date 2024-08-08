#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29503}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox


#--$(dirname "$0")/test.py：运行一个名为 test.py 的 Python 脚本
#把一下参数传给test.py:
#--$CONFIG
#--$CHECKPOINT
#--${@:4}：从第四个位置开始到最后一个位置的所有额外参数，这些参数将被传递给test.py脚本
#--launcher pytorch
#--eval bbox：指定进行边界框（bounding box）评估

#注意：
#你在命令行输入命令运行bash脚本的时候：
#如命令为：./your_script.sh config_file checkpoint_file 2 arg1 arg2 arg3
#config_file checkpoint_file 2 arg1 arg2 arg3就是参数
#分别对应：
#--$CONFIG：config_file
#--$CHECKPOINT：checkpoint_file
#--${@:4}：从第四个位置开始到最后一个位置的所有额外参数：arg1 arg2 arg3（如果没有输入多余的参数那就没有）