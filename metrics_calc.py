import argparse
import os
import sys

import torch
from cv_utils.metrics.torch.segmentation import SegmentationMetricsProcessor
from cv_utils.utils import generate_folds_names
from neural_pipeline import Predictor, FileStructManager, AbstractDataset, MetricsProcessor
from tqdm import tqdm
import numpy as np

from train_config.dataset import create_dataset, SegmentationAugmentations, create_augmented_dataset
from train_config.train_config import TrainConfig, ResNet18SegmentationTrainConfig, \
    ResNet34SegmentationTrainConfig


class MetricsEval:
    def __init__(self, dataset: AbstractDataset, predictor: Predictor, metrics_processor: MetricsProcessor):
        self._dataset = dataset
        self._predictor = predictor
        self._metric = metrics_processor

        self._data_preproc = None
        self._target_preproc = None

    def set_data_preprocess(self, preproc: callable) -> 'MetricsEval':
        self._data_preproc = preproc
        return self

    def set_target_preprocess(self, preproc: callable) -> 'MetricsEval':
        self._target_preproc = preproc
        return self

    def run(self):
        metrics = []
        for d in tqdm(self._dataset):
            predict = self._predictor.predict({'data': d['data'] if self._data_preproc is None else self._data_preproc(d['data'])})  # TODO: change in in Predictor class
            metrics.append(self._metric.calc_metrics(predict, d['target'] if self._target_preproc is None else self._target_preproc(d['target'])))
        return self._metric


def run(config_type: object, out: str):
    dataset = create_augmented_dataset(is_train=False, to_pytorch=True, indices_path='data/indices/test.npy')

    folds = generate_folds_names(TrainConfig.folds_num)

    for fold in folds:
        fsm = FileStructManager(base_dir=os.path.join(config_type.experiment_dir, os.path.splitext(fold['val'])[0]), is_continue=True)
        predictor = Predictor(config_type.create_model(False).cuda(), fsm=fsm)
        metrics = MetricsEval(dataset, predictor, SegmentationMetricsProcessor('eval'))\
            .set_data_preprocess(lambda x: torch.from_numpy(np.expand_dims(x, 0)).cuda())\
            .set_target_preprocess(lambda x: torch.reshape(x, (1, x.shape[0], x.shape[1], x.shape[2]))).run().get_metrics()
        print(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('-m', '--model', type=str, help='Model to predict', required=True, choices=['resnet18', 'resnet34'])
    parser.add_argument('-o', '--out', type=str, help='Output file path', required=True)

    if len(sys.argv) < 3:
        print('Bad arguments passed', file=sys.stderr)
        parser.print_help(file=sys.stderr)
        exit(2)
    args = parser.parse_args()

    if args.model == 'resnet18':
        run(ResNet18SegmentationTrainConfig, args.out)
    elif args.model == 'resnet34':
        run(ResNet34SegmentationTrainConfig, args.out)
    else:
        raise Exception("Train pipeline doesn't implemented for model '{}'".format(args.model))
