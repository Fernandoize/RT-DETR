### PResNet 和 DLANet的区别

PResNet采用了并行残差模块设计，DLANet在不同层级之间进行融合，强调浅层和深层特征的融合

### 如何训练自己的数据集
* 修改类别数
```yaml
num_classes: 5
remap_mscoco_category: False
```
* 修改dataloader中coco格式json和image的配置 注意：此处需要同时修改transform的配置，否则会导致默认的transform丢失
```yaml
train_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: ./configs/dataset/dfui/images/
    ann_file: ./configs/dataset/dfui/annotations/instances_train2017.json
    return_masks: False
    transforms:
      type: Compose
      ops:
        - { type: RandomPhotometricDistort, p: 0.5 }
        - { type: RandomZoomOut, fill: 0 }
        - { type: RandomIoUCrop, p: 0.8 }
        - { type: SanitizeBoundingBox, min_size: 1 }
        - { type: RandomHorizontalFlip }
        - { type: Resize, size: [ 640, 640 ], }
        # - {type: Resize, size: 639, max_size: 640}
        # - {type: PadToSize, spatial_size: 640}
        - { type: ToImageTensor }
        - { type: ConvertDtype }
        - { type: SanitizeBoundingBox, min_size: 1 }
        - { type: ConvertBox, out_fmt: 'cxcywh', normalize: True }
  shuffle: True
  batch_size: 8
  num_workers: 4
  drop_last: True

  collate_fn: default_collate_fn


val_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: configs/dataset/dfui/images/
    ann_file: configs/dataset/dfui/annotations/instances_val2017.json
    transforms:
      type: Compose
      ops:
        # - {type: Resize, size: 639, max_size: 640}
        # - {type: PadToSize, spatial_size: 640}
        - {type: Resize, size: [640, 640]}
        - {type: ToImageTensor}
        - {type: ConvertDtype}

  shuffle: False
  batch_size: 8
  num_workers: 4
  drop_last: False
  collate_fn: default_collate_fn
```