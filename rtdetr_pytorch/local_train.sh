#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

CONFIG_FILE=${1}
python tools/train.py -c $CONFIG_FILE -t checkpoints/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth --seed 42