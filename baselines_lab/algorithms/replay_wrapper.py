import logging
from typing import List

import gym

from algorithms.gym_maze_wrapper import GymMazeWrapper
from algorithms.particle_moving_algorithm import ParticleMovingAlgorithm


class ReplayWrapper(gym.Wrapper):
    """
    This wrapper replays the moves made by a particle moving algorithm step by step to another env.
    This simulates the behaviour of a RL model and can be used to generate an expert dataset for pretraining.
    """
    def __init__(self, replay_env: gym.Env, calculation_env: GymMazeWrapper, prep_algs: List[ParticleMovingAlgorithm], alg: ParticleMovingAlgorithm,
                 target_alg: ParticleMovingAlgorithm):
        super().__init__(replay_env)
        self.prep_algs = prep_algs
        self.alg = alg
        self.target_alg = target_alg
        self.env_copy = calculation_env

        self.action_counter = 0
        self.movement_buffer = []

    def seed(self, seed=None):
        super(ReplayWrapper, self).seed(seed)
        self.env_copy.seed(seed)

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.movement_buffer = []
        self.action_counter = 0
        self.env_copy.reset()
        logging.info("Precalculating moves...")
        for alg in self.prep_algs:
            logging.info("Running preprocessing algorithm {}".format(alg))
            alg.reset_and_run()
            self.movement_buffer.extend(alg.get_movements())

        logging.info("Running algorithm {}".format(self.alg))
        self.alg.reset_and_run()
        self.movement_buffer.extend(self.alg.get_movements())
        logging.info("Running target algorithm.")
        self.target_alg.reset_and_run()
        self.movement_buffer.extend(self.target_alg.get_movements())
        return obs

    def next(self, obs):
        #self.env.render(mode="human")
        self.action_counter += 1
        return self.env_copy.action_to_number(self.movement_buffer[self.action_counter-1])