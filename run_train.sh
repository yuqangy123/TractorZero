#!/bin/bash

# 设置文件描述符限制
ulimit -n 10240

# 可选：打印当前限制，便于确认
echo "当前文件描述符限制: $(ulimit -n)"

export CUDA_VISIBLE_DEVICES=1,0

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 运行训练脚本
python3 train.py