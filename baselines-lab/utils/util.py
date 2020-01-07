import datetime
import time
from stable_baselines.common import set_global_seeds

TIMESTAMP_FORMAT="%Y_%m_%d_%H%M%S"

def set_random_seed(config):
    if 'seed' in config['meta']:
        random_seed = config['meta']['seed']
    else:
        random_seed = time.time()
    config['meta']['random_seed'] = random_seed

    set_global_seeds(random_seed)



def get_timestamp(pattern=TIMESTAMP_FORMAT):
    time = datetime.now()
    return time.strftime(pattern)