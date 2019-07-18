# GIOU-mmdetection
An unofficial Implementation of Generalized Intersection over Union loss for mmdetection.  
The project has referenced a numpy version [GIOU](https://github.com/diggerdu/Generalized-Intersection-over-Union) implement and the [IOU](https://github.com/open-mmlab/mmdetection/blob/466926eb499f4b5c93ce03bd7670ce766bb28e18/mmdet/models/losses/iou_loss.py) implement of mmdetection.
# Usage
1. test
```python
bbox1 = torch.Tensor([[0, 0, 100, 100], [0, 0, 100, 100]])
bbox2 = torch.Tensor([[0, 50, 150, 100], [0, 50, 150, 100]])
gious_loss = GIoULoss()
gloss = gious_loss(bbox1, bbox2)
print(gloss)
```
2. use in mmdetection
edit a config file from mmdetection example, and modify the loss function to GIOULoss.
```python
from giou_loss import GIoULoss
model = dict(
    type='FasterRCNN',
    pretrained='modelzoo://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=81,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0)))
```
