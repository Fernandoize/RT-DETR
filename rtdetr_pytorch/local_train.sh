#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

#rm -rf /root/RT-DETR/rtdetr_pytorch/configs/dataset/dfui
#ln -s /root/autodl-tmp/dfui /root/RT-DETR/rtdetr_pytorch/configs/dataset

CONFIG_FILE=${1}
python tools/train.py -c configs/rtdetr/$CONFIG_FILE -t checkpoints/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth --seed 42