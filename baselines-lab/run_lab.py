import sys
import argparse
import logging

from utils import config_util, util
from experiment.session import Session

def parse_args(args):
    parser = argparse.ArgumentParser("Run script for baselines lab.")
    parser.add_argument("lab_mode", type=str, choices=["train", "enjoy", "search"], help="Mode for the lab - use 'train' for training and enjoy@{savepoint_location} for replay")
    parser.add_argument("config_file", type=str, help="Location of the lab config file")
    parser.add_argument("--verbose", type=int, default=10, help="Verbosity level - corresponds to python logging levels")
    return parser.parse_args(args=args)



def main(args=None):
    # TODO: Model saving
    # TODO: Replay/Enjoy Lab Mode
    # TODO: Lr schedules
    # TODO: Hyperparameter optimization / Search Lab Mode

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    logging.getLogger().setLevel(args.verbose)

    config = config_util.read_config(args.config_file)
    s = Session(config)
    s.run()


if __name__ == "__main__":
    main()

