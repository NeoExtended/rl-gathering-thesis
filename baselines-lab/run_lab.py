import sys
import argparse
import logging

from utils import config_util

def parse_args(args):
    parser = argparse.ArgumentParser("Run script for baselines lab.")
    parser.add_argument("lab_mode", type=str, choices=["dev", "train", "enjoy", "search"], help="Mode for the lab - use 'train' for training and enjoy@{savepoint_location} for replay")
    parser.add_argument("config_file", type=str, help="Location of the lab config file")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level - corresponds to python logging levels")
    return parser.parse_args(args=args)



def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    logging.getLogger().setLevel(args.verbose)






if __name__ == "__main__":
    main()

