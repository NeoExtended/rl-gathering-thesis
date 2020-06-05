import logging
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np

# logging.getLogger('matplotlib.font_manager').disabled = True
from baselines_lab.utils.tensorboard.log_reader import TensorboardLogReader


class Plotter(TensorboardLogReader):
    """
    Class for automated plot creation from tensorboard log files.
    :param file_format: (str) File format for the created plots.
    :param log_dir: (str) Root directory for the tensorboard logs.
    """
    def __init__(self, file_format: str, log_dir: Union[str, List[str]]) -> None:
        super(Plotter, self).__init__(log_dir)
        self.path = self.log_dir.joinpath("figures")
        self.file_format = file_format

    def make_plot(self, tags: List[str], names: List[str], plot_avg_only: bool = False, smoothing: float = 0.6) -> None:
        """
        Creates and saves the plots defined by the given tags.
        :param tags: (List[str]) Tags which correspond to summary tag names in the tensorboard logs
        :param names: (List[str]) Names for the tags. Will be used as ylabel in the plot and as file name.
        """
        assert len(tags) == len(names), "There must be a name for each tag and vise versa!"
        self.path.mkdir(exist_ok=True)

        logging.info("Creating plots.")
        tag_values = self._read_tensorboard_data(tags)

        logging.info("Saving plots to {}.".format(self.path))
        for tag, name in zip(tags, names):
            step_data, value_data = tag_values[tag]
            step_data, value_data = np.asarray(step_data), np.asarray(value_data)

            if len(step_data[0]) == 0:
                continue

            if len(self.files) > 1:
                # Check if all rows in step data are equal. If not interpolate.
                if step_data.dtype == np.object or not (step_data == step_data[0]).all():
                    step_data, value_data = self._interpolate(step_data, value_data)

                arr = np.array(value_data)
                if smoothing > 0:
                    #arr = self._moving_average(arr, smoothing)
                    for idx, run in enumerate(arr):
                        arr[idx] = self._smooth(arr[idx], weight=smoothing)
                mu = np.mean(arr, axis=0)
                std = np.std(arr, axis=0)

                self._make_multi_plot(step_data[0], mu, std, xlabel="steps", ylabel=name, avg_only=plot_avg_only)
            else:
                self._make_plot(step_data[0], value_data[0], xlabel="steps", ylabel=name, smoothing=smoothing)
            self._save_fig(str(self.path.joinpath("{}.{}".format(name.replace(" ", "_"), self.file_format))))

    def _make_multi_plot(self, x, mu, std, xlabel=None, ylabel=None, name=None, avg_only=False, smoothing=False):
        self._prepare_plot(xlabel, ylabel, name)
        if not avg_only:
            plt.fill_between(x, mu + std, mu - std, facecolor='blue', alpha=0.25)
        plt.plot(x, mu, linewidth=1.0)

    def _make_plot(self, x, y, xlabel=None, ylabel=None, name=None, smoothing=0.0):
        self._prepare_plot(xlabel, ylabel, name)
        if smoothing > 0.0:
            y = self._smooth(y, weight=smoothing)
        plt.plot(x, y, linewidth=1.0)

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
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.savefig(path, format=fig_extension, dpi=resolution)