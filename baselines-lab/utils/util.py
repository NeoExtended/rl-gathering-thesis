import datetime
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
    time = datetime.datetime.now()
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
    if not log_dir:
        raise FileNotFoundError("You have to create a log directory first!")
    return log_dir