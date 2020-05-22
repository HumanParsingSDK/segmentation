import os

import cv2
from albumentations.core.composition import BaseCompose
from human_datasets import PicsartDataset, SuperviselyPersonDataset, AISegmentDataset, CHIP, MHPV2, ClothingCoParsingDataset

import numpy as np
import torch
from albumentations import Compose, SmallestMaxSize, HorizontalFlip, Resize, Rotate, OneOf, RandomContrast, RandomGamma, \
    RandomBrightness, \
    ElasticTransform, GridDistortion, OpticalDistortion, RandomSizedCrop, RandomCrop, RandomBrightnessContrast, GaussNoise, \
    BasicTransform

from pietoolbelt.datasets.utils import AugmentedDataset, DatasetsContainer, InstanceSegmentationDataset
from pietoolbelt.datasets.common import BasicDataset

__all__ = ['create_dataset', 'create_augmented_dataset']

DATA_HEIGHT, DATA_WIDTH = 512, 512


class SegmentationAugmentations:
    def __init__(self, is_train: bool, to_pytorch: bool, preprocess):
        if is_train:
            self._aug = Compose([
                preprocess,
                HorizontalFlip(p=0.5),
                GaussNoise(p=0.3),
                OneOf([
                    RandomBrightnessContrast(),
                    RandomGamma(),
                ], p=0.3),
                Rotate(limit=20),
            ], p=1)
        else:
            self._aug = preprocess

        self._need_to_pytorch = to_pytorch

    def augmentate(self, data: {}):
        augmented = self._aug(image=data['data'], mask=data['target'] / data['target'].max())

        img, mask = augmented['image'], augmented['mask']
        if self._need_to_pytorch:
            img, mask = self.img_to_pytorch(img), self.mask_to_pytorch(mask)

        return {'data': img, 'target': mask}

    @staticmethod
    def img_to_pytorch(image):
        return np.moveaxis(image, -1, 0).astype(np.float32) / 128 - 1

    @staticmethod
    def mask_to_pytorch(mask):
        return torch.from_numpy(np.expand_dims(mask.astype(np.float32), axis=0))


def _create_dataset(is_train: bool, augmented: bool, to_pytorch: bool = True, indices_path: str = None) -> BasicDataset:
    def vertical2quad(force_apply=False, **kwargs):
        image, mask = kwargs['image'], kwargs['mask']
        max_size, min_size = np.max(image.shape), np.min([image.shape[0], image.shape[1]])
        image_tmp, mask_tmp = np.ones((max_size, max_size, image.shape[2]), dtype=np.uint8), np.zeros((max_size, max_size), dtype=np.uint8)
        pos = (max_size - min_size) // 2
        image_tmp[:, pos: pos + min_size, :] = image
        mask_tmp[:, pos: pos + min_size] = mask
        return {'image': image_tmp, 'mask': mask_tmp}

    regular_preprocess = OneOf([Compose([SmallestMaxSize(max_size=DATA_HEIGHT), RandomCrop(height=DATA_HEIGHT, width=DATA_WIDTH)]),
                                Compose([SmallestMaxSize(max_size=int(DATA_HEIGHT * 1.2)),
                                         RandomCrop(height=DATA_HEIGHT, width=DATA_WIDTH)])], p=1)
    vertical_img_preprocess = Compose([vertical2quad, regular_preprocess])

    if augmented:
        datasets = [
            AugmentedDataset(ClothingCoParsingDataset())
                .add_aug(SegmentationAugmentations(is_train, to_pytorch, vertical_img_preprocess).augmentate),
            AugmentedDataset(AISegmentDataset()).add_aug(SegmentationAugmentations(is_train, to_pytorch, regular_preprocess).augmentate),
            AugmentedDataset(InstanceSegmentationDataset(SuperviselyPersonDataset())).add_aug(SegmentationAugmentations(is_train, to_pytorch, regular_preprocess).augmentate),
            AugmentedDataset(PicsartDataset()).add_aug(SegmentationAugmentations(is_train, to_pytorch, regular_preprocess).augmentate),
            AugmentedDataset(CHIP()).add_aug(SegmentationAugmentations(is_train, to_pytorch, regular_preprocess).augmentate),
            AugmentedDataset(MHPV2()).add_aug(SegmentationAugmentations(is_train, to_pytorch, regular_preprocess).augmentate),
        ]
    else:
        datasets = [
            ClothingCoParsingDataset(),
            AISegmentDataset(),
            InstanceSegmentationDataset(SuperviselyPersonDataset()),
            PicsartDataset(),
            CHIP(),
            MHPV2(),
        ]

    dataset = DatasetsContainer(datasets)

    if indices_path is not None:
        dataset.load_indices(indices_path).remove_unused_data()
    return dataset


def create_dataset(indices_path: str = None) -> 'BasicDataset':
    return _create_dataset(is_train=False, augmented=False, indices_path=indices_path)


def create_augmented_dataset(is_train: bool, to_pytorch: bool = True, indices_path: str = None) -> 'BasicDataset':
    return _create_dataset(is_train=is_train, augmented=True, to_pytorch=to_pytorch, indices_path=indices_path)
