import logging
from pathlib import Path
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.signal import savgol_filter

# logging.getLogger('matplotlib.font_manager').disabled = True

IMAGES_PATH = "."


def read_summary_values(file, tags):
    steps = [list() for tag in tags]
    values = [list() for tag in tags]
    for summary in tf.train.summary_iterator(file):
        for value in summary.summary.value:
            for i, tag in enumerate(tags):
                if tag in value.tag:
                    steps[i].append(summary.step)
                    values[i].append(value.simple_value)
    return {tag: (step, val) for tag, step, val in zip(tags, steps, values)}


class Plotter:
    """
    Class for automated plot creation from tensorboard log files.
    :param file_format: (str) File format for the created plots.
    :param log_dir: (str) Root directory for the tensorboard logs.
    """
    def __init__(self, file_format: str, log_dir: str) -> None:
        log_dir_path = Path(log_dir)
        self.files = list(log_dir_path.glob("**/events.out.tfevents.*"))
        self.path = log_dir_path.joinpath("figures")
        self.path.mkdir(exist_ok=True)
        self.file_format = file_format

    def make_plot(self, tags: List[str], names: List[str]) -> None:
        """
        Creates and saves the plots defined by the given tags.
        :param tags: (List[str]) Tags which correspond to summary tag names in the tensorboard logs
        :param names: (List[str]) Names for the tags. Will be used as ylabel in the plot and as file name.
        """
        assert len(tags) == len(names), "There must be a name for each tag and vise versa!"
        tag_values = {}

        logging.info("Creating plots.")
        for file in self.files:
            logging.info("Reading tensorboard logs from {}. This may take a while...".format(file))
            data = read_summary_values(str(file), tags)
            for tag in data:
                if tag not in tag_values:
                    tag_values[tag] = (list(), list())
                tag_values[tag][0].append(data[tag][0])
                tag_values[tag][1].append(data[tag][1])

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
                mu = np.mean(arr, axis=0)
                std = np.std(arr, axis=0)

                self._make_multi_plot(step_data[0], mu, std, xlabel="steps", ylabel=name)
            else:
                self._make_plot(step_data[0], value_data[0], xlabel="steps", ylabel=name)
            self._save_fig(str(self.path.joinpath("{}.{}".format(name.replace(" ", "_"), self.file_format))))

    def _make_multi_plot(self, x, mu, std, xlabel=None, ylabel=None, name=None):
        self._prepare_plot(xlabel, ylabel, name)
        plt.fill_between(x, mu + std, mu - std, facecolor='blue', alpha=0.25)
        plt.plot(x, mu, linewidth=1.0)

    def _make_plot(self, x, y, xlabel=None, ylabel=None, name=None, smoothing=False):
        self._prepare_plot(xlabel, ylabel, name)
        if smoothing:
            y = savgol_filter(y, 51, 3)
        plt.plot(x, y, linewidth=1.0)

    def _interpolate(self, step_data: np.ndarray, value_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mins = [np.min(steps) for steps in step_data]
        maxs = [np.max(steps) for steps in step_data]
        min_step = np.min(mins)
        max_step = np.max(maxs)
        steps = len(step_data[0])
        space = np.linspace(min_step, max_step, steps)
        interpolated = list()
        for steps, values in zip(step_data, value_data):
            interpolated.append(np.interp(space, steps, values))

        return np.array([space]*len(step_data)), np.array(interpolated)

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