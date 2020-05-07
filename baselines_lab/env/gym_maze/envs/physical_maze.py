from typing import Union, Type, Tuple

import numpy as np

from baselines_lab.env.gym_maze.envs import MazeBase
from baselines_lab.env.gym_maze.maze_generators import InstanceGenerator


class PhysicalMaze(MazeBase):
    def __init__(self, instance: Union[str, Type[InstanceGenerator]], goal: Tuple[int, int], goal_range: int, reward_generator: str,
                 reward_kwargs: dict = None, n_particles:int = 256, allow_diagonal: bool = True, instance_kwargs:dict = None) -> None:

        super(PhysicalMaze, self).__init__(instance, goal, goal_range, reward_generator, reward_kwargs, n_particles, allow_diagonal, instance_kwargs)

        random_particle_weights = False
        self.particle_speed = np.zeros((n_particles, 2))
        self.exact_locations = np.zeros((n_particles, 2))
        if random_particle_weights:
            self.particle_weight = self.np_random.randint(1, 2, size=(n_particles,))
        else:
            self.particle_weight = np.full((n_particles,), 1)

        #self.acceleration = 0.25
        self.force = 0.25
        self.acceleration = self.force / self.particle_weight

    def reset(self):
        obs = super(PhysicalMaze, self).reset()
        self.exact_locations = np.copy(self.particle_locations)
        return obs

    def step(self, action):
        info = {}
        dy, dx = self.action_map[action]

        self.particle_speed = self.particle_speed + (np.stack([dy * self.acceleration, dx * self.acceleration], axis=1))
        new_loc = self.exact_locations + self.particle_speed

        rounded_locations = np.rint(new_loc).astype(int)
        valid_locations = self._update_locations(rounded_locations)
        self.exact_locations = np.where(valid_locations, new_loc, self.exact_locations)
        self.particle_speed = self.particle_speed / 1.1  # drag
        self.particle_speed = np.where(valid_locations, self.particle_speed, self.particle_speed / 2)  # collision
        self.particle_speed = np.where(np.abs(self.particle_speed) > 0.01, self.particle_speed, 0)  # stop really slow particles.

        done, reward = self.reward_generator.step(action, self.particle_locations)
        return (self._generate_observation(), reward, done, info)
