import argparse
import logging
import sys

import gym
from stable_baselines.gail import generate_expert_traj

from algorithms.gathering.move_together_algorithm import DynamicShortestPathFollowingAlgorithm, StaticShortestPathFollowingAlgorithm
from algorithms.preprocessing.to_the_corners import MoveToRandomCornerAlgorithm
from algorithms.replay_wrapper import ReplayWrapper
from algorithms.target.target_point_mover import TargetPointMoverAlgorithm
from baselines_lab.algorithms.gathering.move_to_extreme_algorithm import OriginalMoveToExtremeAlgorithm, OriginalMoveToMinSumExtremumAlgorithm
from baselines_lab.algorithms.gym_maze_wrapper import GymMazeWrapper

ALGORITHMS = {
    'ssp': StaticShortestPathFollowingAlgorithm,
    'dsp': DynamicShortestPathFollowingAlgorithm,
    'mte': OriginalMoveToExtremeAlgorithm,
    'mtse': OriginalMoveToMinSumExtremumAlgorithm
}


def parse_args(args):
    parser = argparse.ArgumentParser("Run script for gathering algorithms.")
    parser.add_argument("algorithm", type=str, choices=['ssp', 'dsp', 'mte', 'mtse'], help="Gathering algorithm to use.")
    parser.add_argument("--env", type=str, default="Maze0318Continuous-v0", help="Name of the particle environment to solve.")
    parser.add_argument("--seed", default=82, help="Random seed.")
    parser.add_argument("--n-particles", default=256, help="Number of particles.")
    parser.add_argument("--preprocessing", action="store_true", help="Weather or not to use preprocessing to merge particles in corners.")
    parser.add_argument("--render", action="store_true", help="Weather or not to render the environment.")
    parser.add_argument("--generate-pretrain-data", type=str, default=None, help="Indicates that an archive for pretraining should be generated with a given name.")
    return parser.parse_args(args=args)


def run_alg(args):
    logging.info("Creating environment {}.".format(args.env))
    env = gym.make(args.env, n_particles=args.n_particles)
    env.seed(args.seed)
    env = GymMazeWrapper(env, render=args.render)

    total_moves = 0
    if args.preprocessing:
        logging.info("Running preprocessing algorithm")
        pre_alg = MoveToRandomCornerAlgorithm(env)
        pre_alg.run()
        total_moves += pre_alg.get_number_of_movements()

    logging.info("Running algorithm {}.".format(args.algorithm))
    alg = ALGORITHMS[args.algorithm](env)
    # alg.set_optimization(True)
    alg.run()
    total_moves += alg.get_number_of_movements()

    logging.info("Moving particles to goal position.")
    target_alg = TargetPointMoverAlgorithm(env, tuple(env.get_goal()))
    target_alg.run()
    total_moves += target_alg.get_number_of_movements()

    logging.info("Finished execution. Total number of moves: {}".format(total_moves))


def generate_pretrain_data(args):
    env = gym.make(args.env, n_particles=args.n_particles)
    env_copy = gym.make(args.env, n_particles=args.n_particles)
    env.seed(args.seed)
    env.reset()
    env_copy.seed(args.seed)
    env_copy = GymMazeWrapper(env_copy, render=args.render)
    pre_alg = MoveToRandomCornerAlgorithm(env_copy)
    alg = ALGORITHMS[args.algorithm](env_copy)
    target_alg = TargetPointMoverAlgorithm(env_copy, tuple(env_copy.get_goal()))

    env = ReplayWrapper(env, env_copy, [pre_alg], alg, target_alg, downscale=True, frame_stack=True)
    generate_expert_traj(env.next, args.generate_pretrain_data, env=env, n_episodes=2)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    logging.getLogger().setLevel(logging.INFO)

    if args.generate_pretrain_data:
        generate_pretrain_data(args)
    else:
        run_alg(args)


if __name__ == "__main__":
    main()
