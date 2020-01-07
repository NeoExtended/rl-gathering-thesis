import os
import json
import yaml

from utils import util

def read_config(config_file):
    file = open(config_file, "r")
    ext = os.path.splitext(config_file)[-1]

    if ext == '.json':
        config = json.load(file)
    elif ext == '.yml':
        config = yaml.load(file)
    else:
        raise NotImplementedError("File format unknown")

    file.close()
    config = extend_meta_data(config)
    return config

def extend_meta_data(config):
    extended_info = {
        "trial": -1,
        "session": -1,
        "timestamp": util.get_timestamp(),
        "ckpt": None,
        "random_seed": None,
    }
    config['meta'].update(extended_info)
    return config
