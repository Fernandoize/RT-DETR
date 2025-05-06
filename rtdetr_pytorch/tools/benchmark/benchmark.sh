#!/bin/bash
CONFIG_FILE=${1}
python getinfo.py -c ../../configs/rtdetr/$CONFIG_FILE.yml
python trt_benchmark.py --engine_dir ../tensorrt/$CONFIG_FILE.engine --infer_dir configs/dataset/dfui/images