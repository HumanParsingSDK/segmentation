import argparse
import os
import sys

import torch
import numpy as np

from piepline import Trainer, FileStructManager
from piepline.builtin.monitors.tensorboard import TensorboardMonitor
from pietoolbelt.steps.regression.train import FoldedTrainer

from train_config.train_config import ResNet18SegmentationTrainConfig, ResNet34SegmentationTrainConfig, TrainConfig, MyTrainConfig


def init_trainer(config_type: type(TrainConfig), folds: {}, fsm):
    config = config_type(folds)

    trainer = Trainer(config, fsm, device=torch.device('cuda'))
    tensorboard = TensorboardMonitor(fsm, is_continue=False)
    trainer.monitor_hub.add_monitor(tensorboard)

    trainer.set_epoch_num(100)
    trainer.enable_lr_decaying(coeff=0.5, patience=5, target_val_clbk=lambda: np.mean(config.val_stage.get_losses()))
    trainer.add_on_epoch_end_callback(lambda: tensorboard.update_scalar('params/lr', trainer.data_processor().get_lr()))
    trainer.enable_best_states_saving(lambda: np.mean(config.val_stage.get_losses()))
    trainer.add_stop_rule(lambda: trainer.data_processor().get_lr() < 1e-6)

    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('-m', '--model', type=str, help='Train one model', required=True,
                        choices=['resnet18', 'resnet34'])

    if len(sys.argv) < 2:
        print('Bad arguments passed', file=sys.stderr)
        parser.print_help(file=sys.stderr)
        exit(2)
    args = parser.parse_args()

    trainer = FoldedTrainer(['fold_{}'.format(i) for i in range(MyTrainConfig.folds_num)])

    if args.model == 'resnet18':
        trainer.run(lambda fsm, folds: init_trainer(ResNet18SegmentationTrainConfig, folds, fsm), args.model, 'train')
    elif args.model == 'resnet34':
        trainer.run(lambda fsm, folds: init_trainer(ResNet34SegmentationTrainConfig, folds, fsm), args.model, 'train')
    else:
        raise Exception("Train pipeline doesn't implemented for model '{}'".format(args.model))
