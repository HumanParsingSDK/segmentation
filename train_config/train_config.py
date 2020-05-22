import os
from abc import ABCMeta, abstractmethod

import torch
from pietoolbelt.datasets.utils import DatasetsContainer
from pietoolbelt.losses.common import Reduction
from pietoolbelt.losses.segmentation import BCEDiceLoss
from pietoolbelt.metrics.torch.classification import ClassificationMetricsProcessor
from pietoolbelt.metrics.torch.segmentation import SegmentationMetricsProcessor
from pietoolbelt.models import ResNet18, ModelsWeightsStorage, ModelWithActivation, ResNet34, ClassificationModel, InceptionV3Encoder
from pietoolbelt.models.decoders.unet import UNetDecoder
from piepline import TrainConfig, DataProducer, TrainStage, ValidationStage
from torch import nn
from torch.optim import Adam
from torch.nn import Module, BCEWithLogitsLoss, BCELoss
import numpy as np

from train_config.dataset import create_augmented_dataset
from train_config.focal_loss import FocalLoss, FocalDiceLoss

__all__ = ['MyTrainConfig', 'ResNet18SegmentationTrainConfig', 'ResNet34SegmentationTrainConfig']


class MyTrainConfig(TrainConfig, metaclass=ABCMeta):
    experiment_dir = os.path.join('train')
    batch_size = 2
    folds_num = 3

    def __init__(self, fold_indices: {}):
        model = self.create_model().cuda()

        dir = os.path.join('data', 'indices')

        train_dts = []
        for indices in fold_indices['train']:
            train_dts.append(create_augmented_dataset(is_train=True, indices_path=os.path.join(dir, indices + '.npy')))

        val_dts = create_augmented_dataset(is_train=False, indices_path=os.path.join(dir, fold_indices['val'] + '.npy'))

        self._train_data_producer = DataProducer(DatasetsContainer(train_dts), batch_size=self.batch_size, num_workers=8). \
            global_shuffle(True).pin_memory(True)
        self._val_data_producer = DataProducer(val_dts, batch_size=self.batch_size, num_workers=8). \
            global_shuffle(True).pin_memory(True)

        self.train_stage = TrainStage(self._train_data_producer, SegmentationMetricsProcessor('train'))
        self.val_stage = ValidationStage(self._val_data_producer, SegmentationMetricsProcessor('validation'))

        loss = BCEDiceLoss(0.5, 0.5, reduction=Reduction('mean')).cuda()
        optimizer = Adam(params=model.parameters(), lr=1e-4)

        super().__init__(model, [self.train_stage, self.val_stage], loss, optimizer)

    @staticmethod
    @abstractmethod
    def create_model() -> Module:
        pass


class ResNet18SegmentationTrainConfig(MyTrainConfig):
    experiment_dir = os.path.join(MyTrainConfig.experiment_dir, 'resnet18')

    @staticmethod
    def create_model() -> Module:
        """
        It is better to init model by separated method
        :return:
        """
        enc = ResNet18(in_channels=3)
        ModelsWeightsStorage().load(enc, 'imagenet')
        model = UNetDecoder(enc, classes_num=1)
        return ModelWithActivation(model, activation='sigmoid')


class ResNet34SegmentationTrainConfig(MyTrainConfig):
    experiment_dir = os.path.join(MyTrainConfig.experiment_dir, 'resnet34')

    @staticmethod
    def create_model() -> Module:
        """
        It is better to init model by separated method
        :return:
        """
        enc = ResNet34(in_channels=3)
        ModelsWeightsStorage().load(enc, 'imagenet')
        model = UNetDecoder(enc, classes_num=1)
        return ModelWithActivation(model, activation='sigmoid')
