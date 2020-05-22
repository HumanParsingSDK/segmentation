import argparse
import json
import os
import sys
from random import shuffle
import numpy as np

from neural_pipeline import dict_recursive_bypass, AbstractMonitor, MetricsGroup

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class MPLMonitor(AbstractMonitor):
    """
    This monitor show all data in Matplotlib plots
    """

    class _Plot:
        __cmap = plt.cm.get_cmap('hsv', 10)
        __cmap_indices = [i for i in range(10)]
        shuffle(__cmap_indices)

        def __init__(self, names: [str]):
            self._handle = names[0]

            self._prev_values = {}
            self._colors = {}
            self._axis = None

        def add_values(self, values: {}, epoch_idx: int) -> None:
            for n, v in values.items():
                self.add_value(n, v, epoch_idx)

        def add_value(self, name: str, val: float, epoch_idx: int) -> None:
            if name not in self._prev_values:
                self._prev_values[name] = None
                self._colors[name] = self.__cmap(self.__cmap_indices[len(self._colors)])
            prev_value = self._prev_values[name]
            if prev_value is not None and self._axis is not None:
                self._axis.plot([prev_value[1], epoch_idx], [prev_value[0], val], label=name, c=self._colors[name])
            self._prev_values[name] = [val, epoch_idx]

        def place_plot(self, axis) -> None:
            self._axis = axis

            for n, v in self._prev_values.items():
                self._axis.scatter(v[1], v[0], label=n, c=self._colors[n])

            self._axis.set_ylabel(self._handle)
            self._axis.set_xlabel('epoch')
            self._axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            self._axis.legend()
            plt.grid()

    def __init__(self):
        super().__init__()

        self._realtime = True
        self._plots = {}
        self._plots_placed = False

    def update_losses(self, losses: {}):
        def on_loss(name: str, values: np.ndarray):
            plot = self._cur_plot(['loss', name])
            plot.add_value(name, np.mean(values), self.epoch_num)

        self._iterate_by_losses(losses, on_loss)

        if not self._plots_placed:
            self._place_plots()
            self._plots_placed = True

        if self._realtime:
            plt.pause(0.01)

    def update_metrics(self, metrics: {}) -> None:
        for metric in metrics['metrics']:
            self._process_metric(metric)

        for metrics_group in metrics['groups']:
            for group_name_lv2, metric in metrics_group['groups']:
                for group_name_lv3, metric_lv3 in metric['groups']:
                    self._process_metric(metric, plots_group=group_name_lv2, plot_name=group_name_lv3)
                for metric in metrics_group['metrics']:
                    self._process_metric(metric)
            for metric in metrics_group['metrics']:
                self._process_metric(metric)

    def realtime(self, is_realtime: bool) -> 'MPLMonitor':
        """
        Is need to show data updates in realtime
        :param is_realtime: is need realtime
        :return: self object
        """
        self._realtime = is_realtime

    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.show()

    def _process_metric(self, cur_metric, plots_group: str = None, plot_name: str = None):
        metric_name = cur_metric.keys()[0]
        names = self._compile_names(plots_group, plot_name, metric_name)
        plot = self._cur_plot(names)
        if cur_metric[metric_name].size > 0:
            plot.add_value(cur_metric.keys()[0], np.mean(cur_metric[metric_name]), self.epoch_num)

    @staticmethod
    def _compile_names(plots_group: str, plot_name: str, metric_name: str) -> [str]:
        return [n for n in [plots_group, plot_name, metric_name] if n is not None]

    def _cur_plot(self, names: [str]) -> '_Plot':
        if names[0] not in self._plots:
            self._plots[names[0]] = self._Plot(names)
        return self._plots[names[0]]

    def _place_plots(self):
        number_of_subplots = len(self._plots)
        idx = 1
        for n, v in self._plots.items():
            v.place_plot(plt.subplot(number_of_subplots, 1, idx))
            idx += 1


def collect_curves(experiments: [str]) -> {}:
    res = {}
    for exp in experiments:
        log_file = os.path.join('experiments', exp, 'monitors', 'metrics_log', 'metrics_log.json')

        if not os.path.exists(log_file):
            print("Doesn't find metrics log file for experiment '{}'".format(exp))
            continue

        with open(log_file, 'r') as log:
            res[exp] = json.load(log)

    return res


def group_curves_for_compare(curves: {}):
    def rename_endlists(dictionary: dict, exp_name: str) -> dict:
        """
        Recursive bypass dictionary
        :param dictionary:
        """
        res = {}
        for k, v in dictionary.items():
            if isinstance(v, dict):
                res[k] = rename_endlists(v, exp_name)
            else:
                res['{}_{}'.format(k, exp_name)] = v

        return res

    def merge(a: dict, b: dict, path=[]):
        for key in b:
            if key in a:
                if isinstance(a[key], dict) and isinstance(b[key], dict):
                    merge(a[key], b[key], path + [str(key)])
                elif a[key] == b[key]:
                    pass
                else:
                    raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
            else:
                a[key] = b[key]
        return a

    res = {}
    for exp_name, crvs in curves.items():
        res = merge(res, rename_endlists(crvs, exp_name))
    return res


def convert_to_mpl_monitor_format(metrics: dict) -> {}:
    res = {'groups': [], 'metrics': []}
    for k, v in metrics.items():
        if isinstance(v, dict):
            res['groups'].append({k: metrics[k]})
        else:
            res['metrics'].append({k: metrics[k]})
    return res


def substract_by_epoch(metrics: {}, epoch_idx):
    return dict_recursive_bypass(metrics, lambda nod: nod[epoch_idx])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare experiments')
    parser.add_argument('-e', '--exp_names', type=str, help='Experiment name', required=True, nargs='+')
    parser.add_argument('-v', '--visualize', type=str, help='Visualize via Matplotlib')
    parser.add_argument('-o', '--out', type=str, help='Save as image')

    if len(sys.argv) < 2:
        print('Bad arguments passed', file=sys.stderr)
        parser.print_help(file=sys.stderr)
        exit(2)
    args = parser.parse_args()

    curves = collect_curves(args.exp_names)
    to_viz = group_curves_for_compare(curves)

    with MPLMonitor() as monitor:
        cur_epoch = 0
        by_epoch = substract_by_epoch(to_viz, cur_epoch)
        while by_epoch is not None:
            monitor.set_epoch_num(cur_epoch)
            monitor.update_metrics(convert_to_mpl_monitor_format(by_epoch))

            cur_epoch += 1
            by_epoch = substract_by_epoch(to_viz, cur_epoch)
