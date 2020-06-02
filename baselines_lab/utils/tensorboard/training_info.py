import logging
from typing import List, Union

import numpy as np
# logging.getLogger('matplotlib.font_manager').disabled = True
from baselines_lab.utils.tensorboard.log_reader import TensorboardLogReader


class TrainingInformation(TensorboardLogReader):
    """
    Class for automated plot creation from tensorboard log files.
    :param file_format: (str) File format for the created plots.
    :param log_dir: (str) Root directory for the tensorboard logs.
    """
    def __init__(self, log_dir: Union[str, List[str]]) -> None:
        super(TrainingInformation, self).__init__(log_dir)

    def log_key_points(self, drop_level=0.05):
        tags = ["episode_length/ep_length_mean", "episode_length/eval_ep_length_mean"]
        tag_values = self._read_tensorboard_data(tags)

        drop1 = self._get_drop(tag_values.get("episode_length/ep_length_mean"), drop_level=drop_level)
        drop2 = self._get_drop(tag_values.get("episode_length/eval_ep_length_mean"), drop_level=drop_level)
        step_data, value_data = tag_values.get("episode_length/eval_ep_length_mean")
        step_data, value_data = np.asarray(step_data), np.asarray(value_data)
        # Check if all rows in step data are equal. If not interpolate.
        if step_data.dtype == np.object or not (step_data == step_data[0]).all():
            step_data, value_data = self._interpolate(step_data, value_data)
        avg = np.mean(value_data, axis=0)[-1]
        min = np.min(value_data, axis=0)[-1]

        logging.info(str(self.log_dir))
        logging.info("Drop Train: {}".format(drop1))
        logging.info("Drop Test: {}".format(drop2))
        logging.info("Avg: {}".format(avg))
        logging.info("Min: {}".format(min))
        return drop1, drop2, avg, min

    def _get_drop(self, drop_data, drop_level=0.05, drop_min=100000) -> int:
        step_data, value_data = drop_data
        step_data, value_data = np.asarray(step_data), np.asarray(value_data)
        # Check if all rows in step data are equal. If not interpolate.
        if step_data.dtype == np.object or not (step_data == step_data[0]).all():
            step_data, value_data = self._interpolate(step_data, value_data)
        mu = np.mean(value_data, axis=0)
        mu0 = np.max(mu)

        for step, m in zip(step_data[0], mu):
            if m < (1 - drop_level) * mu0 and step > drop_min:
                return step
        return step_data[0][-1]
