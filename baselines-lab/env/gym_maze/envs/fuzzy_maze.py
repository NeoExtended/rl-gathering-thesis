from typing import Tuple, Union, List

import numpy as np

from env.gym_maze.envs.maze_base import MazeBase
from env.gym_maze.rewards import RewardGenerator, ContinuousRewardGenerator


class FuzzyMaze(MazeBase):
    """
    Class which emulates random particle movements and/or fuzzy actions. Fuzzy mean that for each particle the chosen action may not be applied.
    Note that merged particles will not be divided and still move equally.

    :param random_move_chance: (float) Chance for every particle to move randomly.
    :param random_move_distance: (int) Distance (per dimension) every particle may randomly move.
    :param fuzzy_action_probability: (float) Probability for each particle to be not affected by an action.
    """
    def __init__(self, map_file: str, goal: Union[Tuple[int, int], List[Tuple[int, int]]], goal_range: int, reward_generator: RewardGenerator = ContinuousRewardGenerator,
                 reward_kwargs=None, n_particles:int = 256, random_move_chance: float = 0.1, random_move_distance: int = 1, fuzzy_action_probability: float = 0.1):

        super().__init__(map_file, goal, goal_range, reward_generator, reward_kwargs, n_particles)
        self.random_move_chance = random_move_chance
        self.random_move_distance = random_move_distance
        self.fuzzy_action_probability = fuzzy_action_probability

    def step(self, action):
        info = {}

        new_loc = self._make_move(action)
        new_loc = self._make_random_moves(new_loc)
        self._update_locations(new_loc)

        done, reward = self.reward_generator.step(action, self.particle_locations)
        return (self._generate_observation(), reward, done, info)

    def _make_random_moves(self, new_loc):
        if self.random_move_chance > 0.0:
            random_moves = self.np_random.randint(-self.random_move_distance, self.random_move_distance + 1, self.maze.shape + (2,))
            #random_moves_y = self.np_random.randint(0, self.random_move_distance + 1, self.maze.shape)
            mask = self.np_random.choice([0, 1], self.maze.shape + (1,), p=[1 - self.random_move_chance, self.random_move_chance])
            random_moves = random_moves * mask

            random_particle_moves = np.squeeze((np.array([random_moves[tuple(self.particle_locations.T)]])))
            return new_loc + random_particle_moves

    def _make_move(self, action):
        if self.fuzzy_action_probability == 0.0:
            dy, dx = self.action_map[action]
            return self.particle_locations + [dx, dy]
        else:
            dy, dx = self.action_map[action]
            new_loc = self.particle_locations + [dx, dy]
            global_mask = self.np_random.choice([0, 1], self.maze.shape + (1,), p=[1 - self.fuzzy_action_probability, self.fuzzy_action_probability])
            particle_mask = np.squeeze(np.array([global_mask[tuple(self.particle_locations.T)]]))
            return np.where(particle_mask, new_loc, self.particle_locations)
