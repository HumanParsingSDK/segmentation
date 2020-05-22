import json
import os
from itertools import combinations

import numpy as np
import torch
from cv_utils.metrics.torch.segmentation import dice

from train_config.dataset import create_dataset, Dataset


def calc_metric(predicts: np.ndarray, targets: np.ndarray) -> float:
    metric = dice(torch.from_numpy(np.expand_dims(predicts, axis=1)), torch.from_numpy(np.expand_dims(targets, axis=1)))
    return float(np.mean(metric))


def convert_data_to_array(predicts_dict: {}, dataset: Dataset) -> [np.ndarray, np.ndarray]:
    predicts, targets = [], []
    images_paths = dataset.get_items()
    for i, data in enumerate(dataset):
        targets.append(data['target'])
        predicts.append(predicts_dict[os.path.splitext(os.path.basename(images_paths[i]))[0]])

    return np.array(predicts, dtype=np.float32), np.array(targets, dtype=np.uint8)


def get_best_thresh(predicts_dict: {}, dataset: Dataset):
    predicts, targets = convert_data_to_array(predicts_dict['preds'], dataset)
    thresolds = np.linspace(0.5, 1, num=10)
    best_metric, best_thresh = 0, None
    for thresh in thresolds:
        cur_predicts = predicts.copy()
        cur_predicts[cur_predicts < thresh] = 0
        cur_predicts[cur_predicts > 0] = 1
        cur_predicts = cur_predicts.astype(np.uint8)
        cur_metric = calc_metric(cur_predicts, targets)
        if cur_metric > best_metric:
            best_metric = cur_metric
            best_thresh = thresh
    return best_metric, best_thresh


def unite_predicts(predicts: [{}]) -> {}:
    res = {k: [v] for k, v in predicts[0].items()}
    for pred in predicts[1:]:
        for k, v in pred.items():
            res[k].append(v)

    for k, v in res.items():
        res[k] = np.mean(res[k])

    return res


if __name__ == '__main__':
    predicts_path = r'out/class'
    all_predicts = []
    for f in os.listdir(predicts_path):
        predicts_from_file = np.loadtxt(os.path.join(predicts_path, f), delimiter=',')[1:]
        predicts_from_file = {v[0]: v[1] for v in predicts_from_file}

        all_predicts.append({'preds': predicts_from_file, 'net': f})

    for num in range(2, len(all_predicts) + 1):
        for cmb in list(combinations(all_predicts, num)):
            all_predicts.append({'preds': unite_predicts([all_predicts[i]['preds'] for i in cmb]),
                                 'net': ','.join([all_predicts[i]['net'] for i in cmb])})

    dataset = create_dataset(is_test=False, for_segmentation=True, indices_path='data/out/test_class.npy')

    best_thresh, best_pred, best_metric = None, None, 0
    for pred in all_predicts:
        cur_metric, thresh = get_best_thresh(pred, dataset)
        if cur_metric > best_metric:
            best_metric = cur_metric
            best_thresh = thresh
            best_pred = pred

    with open(os.path.join(predicts_path, 'class_best_predict_config.json'), 'w') as out:
        json.dump(out, {'net': best_pred['net'], 'thresh': best_thresh})
