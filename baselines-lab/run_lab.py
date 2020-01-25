import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # Ignore future warnings from numpy/tensorflow version problems

import sys
import argparse
import logging
import re

from utils import config_util
from experiment.session import Session
from env.gym_maze.envs.MazeBase import MazeBase

def check_lab_mode(arg_value, pat=re.compile(r"^train$|^enjoy@.+$|^search$")):
    if not pat.match(arg_value):
        raise argparse.ArgumentTypeError
    return arg_value

def parse_args(args):
    parser = argparse.ArgumentParser("Run script for baselines lab.")
    #parser.add_argument("lab_mode", type=check_lab_mode,
    #                   help="Mode for the lab - use 'train' for training and enjoj@[best, last] enjoy@{log_location}:[best, last] for replay")

    subparsers = parser.add_subparsers(help="Argument defining the lab mode.", dest="lab_mode", required=True)

    enjoy_parser = subparsers.add_parser("enjoy")
    enjoy_parser.add_argument("--type", help="Checkpoint type to load", choices=["best", "last"], default="best")
    enjoy_parser.add_argument("--checkpoint-path", help="Path to a directory containing a model checkpoint (defaults to config log dir)", default="")
    enjoy_parser.add_argument("--video", help="Create a video file in enjoy mode", action="store_true")

    train_parser = subparsers.add_parser("train")

    search_parser = subparsers.add_parser("search")

    parser.add_argument("config_file", type=str, help="Location of the lab config file")
    parser.add_argument("--verbose", type=int, default=10, help="Verbosity level - corresponds to python logging levels")
    return parser.parse_args(args=args)


def main(args=None):
    # TODO: Video/Image Export
    # TODO: Replay mode with evaluation (get mean reward and episode length), also include/exclude failed runs
    # TODO: Hyperparameter optimization / Search Lab Mode
    # TODO: HER/GAIL - experience replay / expert training
    # TODO: Allow user to run multiple experiments
    # TODO: Make GoalRewardGenerator configurable
    # TODO: New MazeEnv with random number of robots
    # TODO: New MazeEnv with random goal position
    # TODO: New MazeEnv with random maze

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    logging.getLogger().setLevel(args.verbose)

    config = config_util.get_config(args.config_file, args)

    s = Session(config, args.lab_mode)
    s.run()


if __name__ == "__main__":
    main()

