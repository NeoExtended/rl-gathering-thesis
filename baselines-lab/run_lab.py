import sys
import argparse
import logging
import re

from utils import config_util, util
from experiment.session import Session
from env.gym_maze.envs.MazeBase import MazeBase

import warnings

def check_lab_mode(arg_value, pat=re.compile(r"^train$|^enjoy@.+$|^search$")):
    if not pat.match(arg_value):
        raise argparse.ArgumentTypeError
    return arg_value

def parse_args(args):
    parser = argparse.ArgumentParser("Run script for baselines lab.")
    parser.add_argument("lab_mode", type=check_lab_mode,
                        help="Mode for the lab - use 'train' for training and enjoj@[best, last] enjoy@{log_location}:[best, last] for replay")
    parser.add_argument("config_file", type=str, help="Location of the lab config file")
    parser.add_argument("--verbose", type=int, default=10, help="Verbosity level - corresponds to python logging levels")
    return parser.parse_args(args=args)



def main(args=None):
    # TODO: Replay/Enjoy Lab Mode
    # TODO: Lr schedules
    # TODO: Hyperparameter optimization / Search Lab Mode

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    logging.getLogger().setLevel(args.verbose)

    config = config_util.read_config(args.config_file)
    lab_mode = args.lab_mode.split("@")[0]

    if lab_mode == 'enjoy':
        config['algorithm']['trained_agent'] = util.parse_enjoy_mode(config['meta']['log_dir'], args.lab_mode)
        config['meta'].pop('log_dir') # No logs in enjoy mode!

    s = Session(config, lab_mode)
    s.run()


if __name__ == "__main__":
    main()

