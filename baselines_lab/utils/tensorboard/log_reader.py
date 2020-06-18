import logging
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import tensorflow as tf

# logging.getLogger('matplotlib.font_manager').disabled = True

IMAGES_PATH = ".."


def read_summary_values(file, tags, max_step=None):
    steps = [list() for tag in tags]
    values = [list() for tag in tags]
    begin = None
    end = 0
    for summary in tf.train.summary_iterator(file):
        if not begin:
            begin = summary.wall_time
        if max_step is not None and summary.step > max_step:
            continue
        if summary.wall_time > end:
            end = summary.wall_time
        for value in summary.summary.value:
            for i, tag in enumerate(tags):
                if tag in value.tag:
                    steps[i].append(summary.step)
                    values[i].append(value.simple_value)
    delta = end - begin
    return {tag: (step, val) for tag, step, val in zip(tags, steps, values)}, delta


class TensorboardLogReader:
    """
    Class for automated plot creation from tensorboard log files.
    :param file_format: (str) File format for the created plots.
    :param log_dir: (str) Root directory for the tensorboard logs.
    :param max_step: (int) Last step that is read from the log.
    """
    def __init__(self, log_dirs: List[str]) -> None:
        self.tb_logs = {}

        for file in log_dirs:
            self.tb_logs[file] = list(Path(file).glob("**/events.out.tfevents.*"))

        self.values = None
        self.deltas = None

    def _read_tensorboard_data(self, tags: List[str], max_step: Optional[int] = None) -> None:
        self.values = {}
        self.deltas = {}
        for dir in self.tb_logs:
            tag_values = {}
            deltas = []
            for log_file in self.tb_logs[dir]:
                logging.info("Reading tensorboard logs from {}. This may take a while...".format(log_file))
                data, delta = read_summary_values(str(log_file), tags, max_step)
                deltas.append(delta)
                for tag in data:
                    if tag not in tag_values:
                        tag_values[tag] = (list(), list())
                    tag_values[tag][0].append(data[tag][0])
                    tag_values[tag][1].append(data[tag][1])

            self.values[dir] = tag_values
            self.deltas[dir] = deltas

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