import os
import json
import yaml

from utils import util


def get_config(config_file, lab_mode):
    config = read_config(config_file)
    config = extend_meta_data(config)
    config = clean_config(config, lab_mode)
    return config


def clean_config(config, lab_mode):
    mode = lab_mode.split("@")[0]
    if mode == 'enjoy':
        dir, ckpt_type = util.parse_enjoy_mode(config['meta']['log_dir'], lab_mode)
        config['algorithm']['trained_agent'] = util.get_model_location(dir, ckpt_type)
        config['meta'].pop('log_dir') # No logs in enjoy mode!

        # Do not change running averages in enjoy mode
        if 'normalize' in config['env']:
            if isinstance(config['env']['normalize'], bool):
                config['env'].pop('normalize')
                config['env']['normalize'] = {'training' : False}
            else:
                config['env']['normalize']['training'] = False

            config['env']['normalize']['trained_agent'] = util.get_normalization_params(dir, ckpt_type)

    return config


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
        "random_seed": None,
    }
    config['meta'].update(extended_info)
    return config

