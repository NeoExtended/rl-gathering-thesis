import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # Ignore future warnings from numpy/tensorflow version problems

import sys
import argparse
import logging
import re

from utils import config_util
from experiment.session import Session
from env.gym_maze.envs.MazeBase import MazeBase

def parse_args(args):
    parser = argparse.ArgumentParser("Run script for baselines lab.")
    #parser.add_argument("lab_mode", type=check_lab_mode,
    #                   help="Mode for the lab - use 'train' for training and enjoj@[best, last] enjoy@{log_location}:[best, last] for replay")

    subparsers = parser.add_subparsers(help="Argument defining the lab mode.", dest="lab_mode", required=True)

    enjoy_parser = subparsers.add_parser("enjoy")
    enjoy_parser.add_argument("--type", help="Checkpoint type to load", choices=["best", "last"], default="best")
    enjoy_parser.add_argument("--checkpoint-path",
                              help="Path to a directory containing a model checkpoint (defaults to config log dir)",
                              default="")
    enjoy_parser.add_argument("--video", help="Create a video file in enjoy mode", action="store_true")
    enjoy_parser.add_argument("--obs-video",
                              help="Create a video file capturing the observations (only works if the env outputs image-like obs)",
                              action="store_true")
    enjoy_parser.add_argument("--stochastic",
                              help="Execute the neural network in stochastic instead of deterministic mode.",
                              action="store_true")
    enjoy_parser.add_argument("--evaluate",
                              help="Activates the model evaluation over at least x given episodes and saves the result to the model dir. "
                                   "(Use with --strict option for more accurate evaluation!)",
                              type=int,
                              default=None)
    enjoy_parser.add_argument("--strict",
                              help="Sets the number of environments to 1. Results in more accurate but far slower evaluation.",
                              action="store_true")

    train_parser = subparsers.add_parser("train")

    search_parser = subparsers.add_parser("search")

    parser.add_argument("config_file", type=str, help="Location of the lab config file")
    parser.add_argument("--verbose", type=int, default=10, help="Verbosity level - corresponds to python logging levels")
    return parser.parse_args(args=args)


def main(args=None):
    # TODO: Hyperparameter optimization / Search Lab Mode
    # TODO: HER/GAIL - experience replay / expert training
    # TODO: Allow user to run multiple experiments
    # TODO: Make GoalRewardGenerator configurable
    # TODO: New MazeEnv with random maze
    # TODO: Config dependencies: Link configs together for clearer params between configs.

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    logging.getLogger().setLevel(args.verbose)

    config = config_util.get_config(args.config_file, args)

    s = Session.create_session(config, args)
    s.run()


if __name__ == "__main__":
    main()

