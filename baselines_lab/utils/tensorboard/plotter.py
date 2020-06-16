import logging
import os
from pathlib import Path
from typing import List, Optional, Dict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# logging.getLogger('matplotlib.font_manager').disabled = True
from baselines_lab.utils.tensorboard.log_reader import TensorboardLogReader


class Plotter(TensorboardLogReader):
    """
    Class for automated plot creation from tensorboard log files.
    :param file_format: (str) File format for the created plots.
    :param log_dir: (str) Root directory for the tensorboard logs.
    """
    def __init__(self, file_format: str, log_dirs: List[str], output_path: Optional[str] = None) -> None:
        super(Plotter, self).__init__(log_dirs)
        self.path = Path(output_path).joinpath("figures") if output_path is not None else Path(log_dirs[0]).joinpath("figures")
        self.file_format = file_format
        if len(log_dirs) < 11:
            self.cmap = plt.get_cmap("tab10")
        else:
            self.cmap = plt.cm.get_cmap("hsv", len(log_dirs) + 5)

    def make_plot(self,
                  tags: List[str],
                  names: List[str],
                  y_labels: Optional[List[str]] = None,
                  alias: Optional[Dict[str, str]] = None,
                  plot_avg_only: bool = False,
                  smoothing: float = 0.6) -> None:
        """
        Creates and saves the plots defined by the given tags.
        :param y_labels: (Optional[List[str]]) Labels for the y axis.
        :param smoothing: (float) Factor for the exponential weighted average smoothing.
        :param plot_avg_only: (bool) Weather or not to only plot the average for runs with multiple trials, or additional std around.
        :param tags: (List[str]) Tags which correspond to summary tag names in the tensorboard logs
        :param names: (List[str]) Names for the tags. Will be used as ylabel in the plot and as file name.
        """
        y_labels = y_labels if y_labels is not None else names
        assert len(tags) == len(names) == len(y_labels), "There must be a name for each tag and vise versa!"
        if alias:
            assert len(alias) == len(self.tb_logs), "There must be an alias for every log directory!"
        self.path.mkdir(exist_ok=True)

        logging.info("Creating plots.")
        tag_values = self._read_tensorboard_data(tags)

        logging.info("Saving plots to {}.".format(self.path))

        for tag, name, label in zip(tags, names, y_labels):
            self._prepare_plot("steps", label, name)

            for i, log_dir in enumerate(self.tb_logs):
                step_data, value_data = tag_values[log_dir][tag]
                step_data, value_data = np.asarray(step_data), np.asarray(value_data)
                label = None
                if alias:
                    label = alias[os.path.basename(log_dir)]
                self._add_plot(step_data, value_data, self.cmap(i), label, plot_avg_only, smoothing)

            plt.legend()
            self._save_fig(str(self.path.joinpath("{}.{}".format(name.replace(" ", "_"), self.file_format))))
            plt.close()

    def _add_plot(self, step_data: np.ndarray, value_data: np.ndarray, color, label=None, plot_avg_only: bool = False, smoothing: float = 0.6):
        if len(step_data[0]) == 0:
            return

        if len(step_data) > 1:
            # Check if all rows in step data are equal. If not interpolate.
            if step_data.dtype == np.object or not (step_data == step_data[0]).all():
                step_data, value_data = self._interpolate(step_data, value_data)

            arr = np.array(value_data)
            if smoothing > 0:
                # arr = self._moving_average(arr, smoothing)
                for idx, run in enumerate(arr):
                    arr[idx] = self._smooth(arr[idx], weight=smoothing)
            mu = np.mean(arr, axis=0)
            std = np.std(arr, axis=0)

            self._make_multi_plot(step_data[0], mu, std, color, avg_only=plot_avg_only, label=label)
        else:
            self._make_plot(step_data[0], value_data[0], color, smoothing=smoothing, label=label)

    def _make_multi_plot(self, x, mu, std, color, label=None, avg_only=False):
        if not avg_only:
            plt.fill_between(x, mu + std, mu - std, facecolor=color, alpha=0.2)
        plt.plot(x, mu, color=color, linewidth=1.0, label=label)

    def _make_plot(self, x, y, color, label=None, smoothing=0.0):
        if smoothing > 0.0:
            y = self._smooth(y, weight=smoothing)
        plt.plot(x, y, color=color, linewidth=1.0, label=label)

    def _moving_average(self, arr, n=3):
        if len(arr.shape) > 1:
            return np.apply_along_axis(lambda m: np.convolve(m, np.ones(n), 'valid') / n, axis=1, arr=arr)
        else:
            return np.convolve(arr, np.ones(n), 'valid') / n

    def _smooth(self, scalars, weight=0.6):
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point  # exponential moving average
            smoothed.append(smoothed_val)
            last = smoothed_val

        return smoothed

    def _prepare_plot(self, xlabel=None, ylabel=None, name=None):
        plt.figure(figsize=(8, 4))
        if name:
            plt.title(name)
        plt.grid()

        if xlabel:
            plt.xlabel(xlabel)

        if ylabel:
            plt.ylabel(ylabel)

    def _save_fig(self, path, tight_layout=True, fig_extension="pdf", resolution=300):
        if tight_layout:
            plt.tight_layout()
        ax = plt.gca()
        ax.xaxis.set_major_formatter(ticker.EngFormatter(sep="\N{THIN SPACE}"))
        #plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.savefig(path, format=fig_extension, dpi=resolution)
