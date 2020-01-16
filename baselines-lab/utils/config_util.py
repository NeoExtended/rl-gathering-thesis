"""
Defines helper functions for reading and writing the lab config file
"""

import os
import json
import yaml

from utils import util


def get_config(config_file, lab_mode):
    """
    Reads the lab config from a given file and configures it for use with to the current lab mode.
    :param config_file: (str) Path to the config file.
    :param lab_mode: (str) Unparsed lab mode string as given by the user.
    :return: (dict) The parsed config file as dictionary.
    """
    config = read_config(config_file)
    config = extend_meta_data(config)
    config = clean_config(config, lab_mode)
    return config


def clean_config(config, lab_mode):
    """
    Deletes or modifies keys from the config which are not compatible with the current lab mode.
    :param config: (dict) The config dictionary
    :param lab_mode: (str) The unparsed lab mode as given by the user
    :return: (dict) The cleaned config dictionary
    """
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
    """
    Reads a config file from disc. The file must follow JSON or YAML standard.
    :param config_file: (str) Path to the config file.
    :return: (dict) A dict with the contents of the file.
    """
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
    """
    Saves a given lab configuration to a file.
    :param config: (dict) The lab configuration.
    :param path: (str) Desired file location.
    """
    ext = os.path.splitext(path)[-1]
    file = open(path, "w")
    if ext == '.json':
        json.dump(config, file, indent=2, sort_keys=True)
    elif ext == '.yml':
        yaml.dump(config, file, indent=2)
    file.close()


def extend_meta_data(config):
    """
    Extends the meta-data dictionary of the file to save additional information at training time.
    :param config: (dict) The config dictionary.
    :return: (dict) The updated config dictionary.
    """
    extended_info = {
        "timestamp": util.get_timestamp(),
        "random_seed": None,
    }
    config['meta'].update(extended_info)
    return config

