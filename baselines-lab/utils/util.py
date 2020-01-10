from datetime import datetime
import time
import os
from stable_baselines.common import set_global_seeds


log_dir = None
TIMESTAMP_FORMAT="%Y_%m_%d_%H%M%S"

def set_random_seed(config):
    random_seed = config['meta'].get('seed', time.time())
    config['meta']['seed'] = random_seed

    set_global_seeds(random_seed)


def get_timestamp(pattern=TIMESTAMP_FORMAT):
    time = datetime.now()
    return time.strftime(pattern)

def create_log_directory(root):
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
    global log_dir
    return log_dir

def parse_enjoy_mode(log_dir, lab_mode):
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

    savepoints = os.listdir(dir)
    best = None
    last = None
    latest = datetime.fromisoformat('1970-01-01')

    for savepoint in savepoints:
        split = os.path.splitext(savepoint)[0].split("_")
        if split[-1] == 'best':
            best = os.path.join(dir, savepoint)
        else:
            datestring = "_".join(split[2:6])
            date = datetime.strptime(datestring, TIMESTAMP_FORMAT)
            if date > latest:
                last = os.path.join(dir, savepoint)
                latest = date

    if ckpt_type == 'best':
        assert best, "Could not find best model in {}".format(dir)
        return best
    else:
        assert last, "Could not find last model in {}".format(dir)
        return last