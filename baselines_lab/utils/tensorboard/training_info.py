import logging
from typing import List, Union

import numpy as np
# logging.getLogger('matplotlib.font_manager').disabled = True
from baselines_lab.utils.tensorboard.log_reader import TensorboardLogReader, interpolate


class TrainingInformation(TensorboardLogReader):
    """
    Class for automated plot creation from tensorboard log files.
    :param file_format: (str) File format for the created plots.
    :param log_dir: (str) Root directory for the tensorboard logs.
    """
    def __init__(self, log_dir: str) -> None:
        super(TrainingInformation, self).__init__([log_dir])
        self.log_dir = log_dir
        self.avg = None
        self.min = None
        self.time_delta = None
        self.std = None
        self.var = None
        self.drop_train = None
        self.drop_test = None
        self.cv = None

    def log_key_points(self, drop_level=0.05, max_step=None):
        tags = ["episode_length/ep_length_mean", "episode_length/eval_ep_length_mean"]
        self.read_data(tags, max_step=max_step)
        tag_values = self.values[self.log_dir]

        self.drop_train = self._get_drop(tag_values.get("episode_length/ep_length_mean"), drop_level=drop_level)
        self.drop_test = self._get_drop(tag_values.get("episode_length/eval_ep_length_mean"), drop_level=drop_level)
        step_data, value_data = tag_values.get("episode_length/eval_ep_length_mean")
        step_data, value_data = np.asarray(step_data), np.asarray(value_data)
        # Check if all rows in step data are equal. If not interpolate.
        if step_data.dtype == np.object or not (step_data == step_data[0]).all():
            step_data, value_data = interpolate(step_data, value_data)
        self.avg = np.mean(value_data, axis=0)[-1]
        self.min = np.min(value_data, axis=0)[-1]
        self.time_delta = np.average(self.deltas[self.log_dir])
        self.std = np.mean(np.std(value_data, axis=0))
        self.var = np.mean(np.var(value_data, axis=0))
        self.cv = np.mean(np.std(value_data, axis=0) / np.mean(value_data, axis=0))

        logging.info(str(self.log_dir))
        logging.info("Drop Train: {}".format(self.drop_train))
        logging.info("Drop Test: {}".format(self.drop_test))
        logging.info("Avg: {}".format(self.avg))
        logging.info("Min: {}".format(self.min))
        logging.info("Time: {}".format(self.time_delta))
        logging.info("Deviation: {}".format(self.std))
        logging.info("Variance: {}".format(self.var))
        logging.info("Coefficient of Variation: {:.2%}".format(self.cv))

    def _get_drop(self, drop_data, drop_level=0.05, drop_min=100000) -> int:
        step_data, value_data = drop_data
        step_data, value_data = np.asarray(step_data), np.asarray(value_data)
        # Check if all rows in step data are equal. If not interpolate.
        if step_data.dtype == np.object or not (step_data == step_data[0]).all():
            step_data, value_data = interpolate(step_data, value_data)
        mu = np.mean(value_data, axis=0)
        mu0 = np.max(mu)

        for step, m in zip(step_data[0], mu):
            if m < (1 - drop_level) * mu0 and step > drop_min:
                return step
        return step_data[0][-1]
