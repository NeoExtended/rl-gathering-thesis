"""
General helper functions.
"""

from datetime import datetime
import time
import os
from stable_baselines.common import set_global_seeds


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

def get_lastest_checkpoint(dir, prefix="", suffix=""):
    # TODO: Move to model.Saver
    files = os.listdir(dir)

    latest = datetime.fromisoformat('1970-01-01')
    counter = None
    for savepoint in files:
        datestring = os.path.splitext(savepoint)[0]
        if not (datestring.startswith(prefix) and datestring.endswith(suffix)):
            continue

        if len(prefix) > 0:
            datestring = datestring[len(prefix) + 1:]
        if len(suffix) > 0:
            datestring = datestring[:-(len(suffix)+1)]

        step, datestring = datestring.split("_", maxsplit=1)
        # If no suffix is given the datestring may contain invalid data.
        if len(datestring) > 17:
            continue

        date = datetime.strptime(datestring, TIMESTAMP_FORMAT)
        if date > latest:
            latest = date
            counter = step

    return (counter, latest.strftime(TIMESTAMP_FORMAT))

def get_savepoint_file(log_dir, prefix, suffix="", extension="zip"):
    counter, latest = get_lastest_checkpoint(log_dir, prefix, suffix)
    if len(suffix) > 0:
        savepoint = os.path.join(log_dir, "{}_{}_{}_{}.{}".format(prefix, counter, latest, suffix, extension))
    else:
        savepoint = os.path.join(log_dir, "{}_{}_{}.{}".format(prefix, counter, latest, extension))

    assert os.path.exists(savepoint), "Could not find savepoint {} in {}".format(savepoint, log_dir)
    return savepoint

def get_model_location(log_dir, type="best"):
    if type == "best":
        return get_savepoint_file(log_dir, prefix="model", suffix="best")
    else:
        return get_savepoint_file(log_dir, prefix="model")

def get_normalization_params(log_dir, type="best"):
    if type == "best":
        return get_savepoint_file(log_dir, prefix="normalization", suffix="best", extension="pkl")
    else:
        return get_savepoint_file(log_dir, prefix="normalization", extension="pkl")

def parse_enjoy_mode(log_dir, lab_mode):
    """
    Parses and checks the parameters for the lab enjoy mode.
    :param log_dir: (str) The log dir as defined in the lab config.
    :param lab_mode: (str) Unparsed lab mode sting.
    :return: (str, str) Tuple containing the directory containing the saved models and the type of the checkpoint which
        should be loaded (best or last).
    """
    mode, location = lab_mode.split("@")
    assert mode == "enjoy", "Lab must be in enjoy mode to parse enjoy mode parameters!"

    if ':' in location:
        dir, ckpt_type = location.split(":")
    else:
        ckpt_type = location
        runs = os.listdir(log_dir)
        runs.sort()
        dir = os.path.join(log_dir, runs[-1]) # Get latest run

    dir = os.path.join(dir, "savepoints")
    assert os.path.exists(dir), "Could not find any savepoints in logdir!"
    assert ckpt_type in ['best', 'last'], "Checkpoint type can only be best or last!"
    return dir, ckpt_type