import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # Ignore future warnings from numpy/tensorflow version problems

import sys
import argparse
import logging
import re

from utils import config_util, util
from experiment.session import Session
from env.gym_maze.envs.MazeBase import MazeBase

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
    # TODO: Env kwargs configuration
    # TODO: Additional logging, e.g. mean episode length
    # TODO: Normalization with precomputed values on random actions
    # TODO: Fix evaluation - Saver Normalization must be the same as on main env
    # TODO: Video/Image Export
    # TODO: Replay mode with evaluation (get mean reward and episode length), also include/exclude failed runs
    # TODO: Hyperparameter optimization / Search Lab Mode
    # TODO: HER/GAIL - experience replay / expert training

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    logging.getLogger().setLevel(args.verbose)

    config = config_util.get_config(args.config_file, args.lab_mode)

    lab_mode = args.lab_mode.split("@")[0]
    s = Session(config, lab_mode)
    s.run()


if __name__ == "__main__":
    main()

