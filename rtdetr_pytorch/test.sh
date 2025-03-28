export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml -t checkpoints/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth --seed 4 --test-only &> test.log 2>&1 &

#- `torchrun --master_port=8844 --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml -t https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth`
