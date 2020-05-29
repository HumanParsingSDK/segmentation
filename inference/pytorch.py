import os
from abc import ABCMeta, abstractmethod
from random import randint
from typing import List

import cv2
import torch
from pietoolbelt.models.utils import ModelContainer
from pietoolbelt.steps.segmentation.inference import SegmentationInference
from pietoolbelt.tta import HFlipTTA, CLAHETTA, VFlipTTA, RotateTTA, GaussNoiseTTA
from torch.nn import Module
import numpy as np
from albumentations import CenterCrop, SmallestMaxSize, LongestMaxSize, Compose, Rotate, HorizontalFlip, BasicTransform, CLAHE, \
    VerticalFlip, GaussNoise

from train_config.train_config import ResNet34SegmentationTrainConfig


def vertical2quad(force_apply=False, **kwargs):
    image = kwargs['image']
    max_size, min_size = np.max(image.shape), np.min([image.shape[0], image.shape[1]])
    image_tmp = np.ones((max_size, max_size, image.shape[2]), dtype=np.uint8)
    pos = (max_size - min_size) // 2
    image_tmp[:, pos: pos + min_size, :] = image

    if 'mask' in kwargs:
        mask_tmp = np.zeros((max_size, max_size), dtype=np.uint8)
        mask_tmp[:, pos: pos + min_size] = kwargs['mask']
        return {'image': image_tmp, 'mask': mask_tmp}

    return {'image': image_tmp}


if __name__ == '__main__':
    models = []
    for i, fold in enumerate(['fold_0', 'fold_1', 'fold_2']):
        models.append(ResNet34SegmentationTrainConfig.create_model(pretrained=False).cuda())
        path = os.path.join('train', 'resnet34', fold, 'checkpoints', 'best', 'best_checkpoint', 'weights.pth')
        models[-1].load_state_dict(torch.load(path))
        torch.save(models[-1], 'model{}.pth'.format(i))

    model = ModelContainer(models, reduction=lambda x: torch.mean(x, dim=0))

    data_transform = Compose([SmallestMaxSize(max_size=512, always_apply=True),
                              CenterCrop(height=512, width=512, always_apply=True)], p=1)

    target_transform = Compose([Rotate(limit=(-90, -90), p=1), HorizontalFlip(p=1)])

    # r1 = RotateTTA(angle_range=(107, 107))
    # r2 = RotateTTA(angle_range=(34, 34))
    # r3 = RotateTTA(angle_range=(63, 63))
    r1 = RotateTTA(angle_range=(-84, -84))
    r2 = RotateTTA(angle_range=(69, 69))
    r3 = RotateTTA(angle_range=(64, 64))
    inference = SegmentationInference(model) \
        .set_data_transform(data_transform) \
        .set_target_transform(target_transform) \
        .set_tta([r1, r2, r3, HFlipTTA()]) \
        .set_threshold(0.5).set_device('cuda')

    # inference.run_webcam("Human segmentation")

    img_path = r"/home/toodef/tmp/scale_1200.webp"
    cv2.namedWindow("Human segmentation", cv2.WINDOW_GUI_NORMAL)
    for _ in range(100):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img, mask = inference.run_image(img)
        print([r1._last_angle, r2._last_angle, r3._last_angle])
        res_img = inference.vis_result(img, mask)
        cv2.imwrite("result5.jpg", res_img)
        cv2.imshow("Human segmentation", res_img)
        cv2.waitKey()
