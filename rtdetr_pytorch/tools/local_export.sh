#!/bin/bash
CONFIG_FILE=${1}

python export_onnx.py -c ../configs/rtdetr/$CONFIG_FILE.yml -r ../output/rtdetr_r18vd_6x_coco/$CONFIG_FILE.pth -f onnx/$CONFIG_FILE.onnx --check --simplify
trtexec --onnx=onnx/$CONFIG_FILE.onnx --saveEngine=$CONFIG_FILE.engine
