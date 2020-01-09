import datetime
import time
from stable_baselines.common import set_global_seeds




TIMESTAMP_FORMAT="%Y_%m_%d_%H%M%S"

def set_random_seed(config):
    random_seed = config['meta'].get('seed', time.time())
    config['meta']['seed'] = random_seed

    set_global_seeds(random_seed)


def get_timestamp(pattern=TIMESTAMP_FORMAT):
    time = datetime.datetime.now()
    return time.strftime(pattern)