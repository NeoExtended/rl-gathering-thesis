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

def save_config(config, path):
    ext = os.path.splitext(path)[-1]
    file = open(path, "w")
    if ext == '.json':
        json.dump(config, file, indent=2, sort_keys=True)
    elif ext == '.yml':
        yaml.dump(config, file, indent=2)
    file.close()


def extend_meta_data(config):
    extended_info = {
        "timestamp": util.get_timestamp(),
        "ckpt": None,
        "random_seed": None,
    }
    config['meta'].update(extended_info)
    return config
