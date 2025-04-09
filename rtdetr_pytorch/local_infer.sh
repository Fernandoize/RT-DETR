python tools/infer.py -c configs/rtdetr/local_test.yml -r log/rtdetr_r18vd_deformable/checkpoint.pth --device cpu --im-file configs/dataset/dfui/images/u002102.jpg
#- `torchrun --master_port=8844 --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml -t https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth`
