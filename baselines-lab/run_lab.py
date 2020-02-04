import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # Ignore future warnings from numpy/tensorflow version problems

import sys
import argparse
import logging

from utils import config_util
from experiment import Session

# Import env package to init gym registry
import env.gym_maze


import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)

def parse_args(args):
    parser = argparse.ArgumentParser("Run script for baselines lab.")
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
    search_parser.add_argument("--plot",
                               help="Weather or not to plot the distribution of choosen hyperparameters",
                               action="store_true")

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
    # TODO: Multi-Level obs videos: Provide obs videos after each? wrapper.
    # TODO: Fix --checkpoint-path option
    # TODO: Investigate performance (float vs int obs, etc)

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

