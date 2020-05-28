from abc import ABCMeta, abstractmethod
from random import randint
from typing import List

import cv2
import torch
from cv_utils.viz import ColormapVisualizer
from pietoolbelt.steps.segmentation.inference import SegmentationInference
from pietoolbelt.tta import HFlipTTA, CLAHETTA, VFlipTTA, RotateTTA
from torch.nn import Module
import numpy as np
from albumentations import CenterCrop, SmallestMaxSize, LongestMaxSize, Compose, Rotate, HorizontalFlip, BasicTransform, CLAHE, \
    VerticalFlip, GaussNoise


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
    model = torch.load('model.pth')

    data_transform = Compose([SmallestMaxSize(max_size=512, always_apply=True),
                              CenterCrop(height=512, width=512, always_apply=True)], p=1)

    target_transform = Compose([Rotate(limit=(-90, -90), p=1), HorizontalFlip(p=1)])

    r1 = RotateTTA(angle_range=(107, 107))
    r2 = RotateTTA(angle_range=(34, 34))
    r3 = RotateTTA(angle_range=(63, 63))
    # r1 = RotateTTA(angle_range=(90, 90))
    # r2 = RotateTTA(angle_range=(-50, -50))
    # r3 = RotateTTA(angle_range=(144, 144))
    inference = SegmentationInference(model) \
        .set_data_transform(data_transform) \
        .set_target_transform(target_transform) \
        .set_tta([r1, r2, r3])\
        .set_threshold(0.5).set_device('cuda')

    inference.run_webcam("Human segmentation")

    # img_path = r"C:\workspace\datasets\human_photos_and_measurements\3\front.jpeg"
    # for _ in range(100):
    #     img = cv2.imread(img_path)
    #     img, mask = inference.run_image(img)
    #     print(r1._last_angle, r2._last_angle, r3._last_angle)
    #     cv2.imshow("Human segmentation", inference.vis_result(img, mask))
    #     cv2.waitKey()
