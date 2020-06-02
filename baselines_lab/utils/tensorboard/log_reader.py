import logging
from pathlib import Path
from typing import Tuple, List, Union, Dict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.signal import savgol_filter

# logging.getLogger('matplotlib.font_manager').disabled = True

IMAGES_PATH = ".."


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


class TensorboardLogReader:
    """
    Class for automated plot creation from tensorboard log files.
    :param file_format: (str) File format for the created plots.
    :param log_dir: (str) Root directory for the tensorboard logs.
    """
    def __init__(self, log_dir: Union[str, List[str]]) -> None:
        self.log_dir = Path(log_dir)
        self.files = list(self.log_dir.glob("**/events.out.tfevents.*"))

    def _read_tensorboard_data(self, tags: List[str]) -> Dict[str, Tuple[List, List]]:
        tag_values = {}
        for file in self.files:
            logging.info("Reading tensorboard logs from {}. This may take a while...".format(file))
            data = read_summary_values(str(file), tags)
            for tag in data:
                if tag not in tag_values:
                    tag_values[tag] = (list(), list())
                tag_values[tag][0].append(data[tag][0])
                tag_values[tag][1].append(data[tag][1])
        return tag_values

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