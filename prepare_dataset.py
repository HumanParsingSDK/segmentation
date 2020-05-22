import numpy as np
import os
from pietoolbelt.steps.stratification import DatasetStratification

from train_config.dataset import create_dataset
from train_config.train_config import MyTrainConfig


def calc_label(x):
    return int(10 * np.count_nonzero(x) / x.size)


if __name__ == '__main__':
    out_dir = os.path.join('data', 'indices')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    test_part = 0.1
    folds_dict = {'fold_{}.npy'.format(i): (1 - test_part) / MyTrainConfig.folds_num for i in range(MyTrainConfig.folds_num)}

    strat = DatasetStratification(create_dataset(), calc_label, workers_num=12)
    strat.run(dict(folds_dict, **{'test.npy': test_part}), out_dir)
