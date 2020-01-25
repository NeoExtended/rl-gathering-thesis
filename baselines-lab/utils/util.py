"""
General helper functions.
"""

from datetime import datetime
import time
import os
import numpy as np

from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import VecEnvWrapper


log_dir = None
TIMESTAMP_FORMAT="%Y_%m_%d_%H%M%S"

def set_random_seed(config):
    """
    Sets the random seed to python, numpy and tensorflow. The selected seed will be saved in the config['meta'] section.
    :param config: The lab config file
    """
    random_seed = config['meta'].get('seed', time.time())
    config['meta']['seed'] = random_seed

    set_global_seeds(random_seed)


def get_timestamp(pattern=TIMESTAMP_FORMAT):
    """
    Generates a string timestamp to use for logging.
    :param pattern: (str) Pattern for the timestamp following the python datetime format.
    :return: (str) The current timestamp formated according to pattern
    """
    time = datetime.now()
    return time.strftime(pattern)


def create_log_directory(root):
    """
    Creates a global log directory at a given place. The directory will be named with a current timestamp.
    :param root: (str) Parent directory for the log directory. Will be created if it does not exist.
    :return: (str) Location of the created log directory.
    """
    if not root:
        return None

    global log_dir
    if not log_dir:
        timestamp = get_timestamp()
        path = os.path.join(root, timestamp)
        log_dir = os.path.abspath(path)
        os.makedirs(log_dir)
    return log_dir


def get_log_directory():
    """
    Returns the current log directory. May be None if create_log_directory() has not been called before.
    """
    global log_dir
    return log_dir


def safe_mean(arr):
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return nan. It is used for logging only.

    :param arr: (np.ndarray)
    :return: (float)
    """
    return np.nan if len(arr) == 0 else np.mean(arr)

def unwrap_vec_env(env, target_wrapper):
    """
    Unwraps the given environment until the target wrapper is found.
    Returns the first wrapper if target wrapper was not found.
    """
    while not isinstance(env, target_wrapper) and isinstance(env.venv, VecEnvWrapper):
        env = env.venv
    return env