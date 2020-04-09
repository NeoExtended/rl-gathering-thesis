from typing import Tuple, Union, List

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.utils import seeding

from env.gym_maze.rewards import GENERATORS

PARTICLE_MARKER = 150
GOAL_MARKER = 200
BACKGROUND_COLOR = (120, 220, 240)
PARTICLE_COLOR = (150, 150, 180)


class MazeBase(gym.Env):
    """
    Base class for a maze-like environment for particle navigation tasks.
    :param map_file: (str or list) *.csv file containing the map data. May be a list for random randomized maps.
    :param goal: (Union[Tuple[int, int], List[Tuple[int, int]]]) A point coordinate in form [x, y] ([column, row]).
        In case of multiple maps, multiple maps, multiple goal positions can be given.
        Can be set to None for a random goal position.
    :param goal_range: (int) Circle radius around the goal position that should be counted as goal reached.
    :param reward_generator: (str) The type of RewardGenerator to use for reward generation. (e.g. "goal" or "continuous")
        based on the current state.
    :param n_particles: (int) Number of particles to spawn in the maze.
        Can be set to -1 for a random number of particles.
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, map_file: str, goal: Union[Tuple[int, int], List[Tuple[int, int]]], goal_range: int,
                 reward_generator: str, reward_kwargs=None, n_particles:int = 256) -> None:

        self.np_random = None
        self.seed()
        self.reward_kwargs = {} if reward_kwargs is None else reward_kwargs
        self.goal_range = goal_range

        if n_particles < 0:
            self.randomize_n_particles = True
            self.n_particles = 0
        else:
            self.randomize_n_particles = False
            self.n_particles = n_particles

        self.reward_generator_class = GENERATORS[reward_generator]
        self.reward_generator = None

        self.map_file = map_file
        self.map_index = -1
        self.goal_proposition = goal
        self._load_map(map_file, goal)

        self.actions = [0, 1, 2, 3, 4, 5, 6, 7]  # {S, SE, E, NE, N, NW, W, SW}
        self.action_map = {0: (1, 0), 1: (1, 1), 2: (0, 1), 3: (-1, 1),
                           4: (-1, 0), 5: (-1, -1), 6: (0, -1), 7: (1, -1)}
        self.rev_action_map = {v: k for k, v in self.action_map.items()}
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(*self.maze.shape, 1), dtype=np.uint8)

        self.particle_locations = np.array([])
        self.reset()

    def _load_map(self, map_file, goal):
        if isinstance(map_file, list):
            self.randomize_map = True
            self.last_map_index = self.map_index
            self.map_index = self.np_random.randint(len(map_file))
            map = map_file[self.map_index]
        else:
            self.randomize_map = False
            self.map_index = None
            map = map_file

        # Load map if necessary
        if isinstance(map_file, str) or self.map_index != self.last_map_index:
            self.freespace = np.loadtxt(map).astype(int)  # 1: Passable terrain, 0: Wall
            self.maze = np.ones(self.freespace.shape,
                                dtype=int) - self.freespace  # 1-freespace: 0: Passable terrain, 1: Wall
            self.height, self.width = self.maze.shape
            self.cost = None

        if goal:
            self.randomize_goal = False
            if isinstance(map_file, list):
                self.goal = goal[self.map_index]
            else:
                self.goal = goal
            self.reward_generator = self.reward_generator_class(self.maze, self.goal, self.goal_range, self.n_particles, **self.reward_kwargs)
        else:
            self.randomize_goal = True
            self.goal = [0, 0]
            # Random goals require a dynamic cost map which will be calculated on each reset.

    def reset(self):
        if self.randomize_map:
            self._load_map(self.map_file, self.goal_proposition)

        locations = np.transpose(np.nonzero(self.freespace))

        # Randomize number of particles if necessary
        self._randomize_n_particles(locations)

        # Randomize goal position if necessary
        self._randomize_goal_position(locations)

        # Reset particle positions
        self._randomize_particle_locations(locations)
        return self._generate_observation()

    def _randomize_particle_locations(self, locations):
        """
        Computes new locations for all particles.
        :param locations: (list) Number of possible locations for the particles.
        """
        choice = self.np_random.choice(len(locations), self.n_particles, replace=False)
        self.particle_locations = locations[choice, :]  # Particle Locations are in y, x (row, column) order
        self.reward_generator.reset(self.particle_locations)

    def _randomize_goal_position(self, locations):
        """
        Computes a new random goal position.
        :param locations: (list) List of possible goal locations to choose from.
        """
        if self.randomize_goal:
            new_goal = locations[self.np_random.randint(0, len(locations))]
            self.goal = [new_goal[1], new_goal[0]]
            self.reward_generator = self.reward_generator_class(self.maze, self.goal, self.goal_range, self.n_particles, **self.reward_kwargs)

    def _randomize_n_particles(self, locations, fan_out=5):
        """
        Computes a random number of particles for the current map.
        :param locations: (list) Number of free locations for particles
        :param fan_out: (int) Parameter to control the maximum number of particles as a fraction of possible particle locations.
        """
        if self.randomize_n_particles:
            self.n_particles = self.np_random.randint(1, len(locations) // fan_out)
            self.reward_generator.set_particle_count(self.n_particles)

    def step(self, action):
        info = {}
        dy, dx = self.action_map[action]

        new_loc = self.particle_locations + [dx, dy]
        self._update_locations(new_loc)

        done, reward = self.reward_generator.step(action, self.particle_locations)
        return (self._generate_observation(), reward, done, info)

    def render(self, mode='human'):
        rgb_image = np.full((*self.maze.shape, 3), BACKGROUND_COLOR, dtype=int)
        maze_rgb = np.stack((self.freespace * 255,) * 3, axis=-1)  # White maze
        rgb_image = rgb_image + maze_rgb
        for y, x in self.particle_locations:
            rgb_image[y - 1:y + 1, x - 1:x + 1] = PARTICLE_COLOR

        cv2.circle(rgb_image, tuple(self.goal), self.goal_range, (255, 0, 0), thickness=1)
        rgb_image = np.clip(rgb_image, 0, 255)

        if mode == 'human':  # Display image
            plt.gcf().clear()  # Clear current figure
            plt.imshow(rgb_image.astype(np.uint8), vmin=0, vmax=255)
            plt.show(False)
            plt.pause(0.0001)
        # rgb_image = cv2.resize(rgb_image.astype(np.uint8), (100, 100), interpolation=cv2.INTER_AREA)
        return rgb_image.astype(np.uint8)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _generate_observation(self):
        self.state_img = np.zeros(self.maze.shape)

        # self.state_img[self.robot_locations[:, 0], self.robot_locations[:, 1]] = ROBOT_MARKER
        self.state_img[tuple(self.particle_locations.T)] = PARTICLE_MARKER
        observation = self.state_img + self.maze * 255
        if self.randomize_goal:
            cv2.circle(observation, tuple(self.goal), self.goal_range, (GOAL_MARKER))
            observation[self.goal[1] - 1:self.goal[1] + 1, self.goal[0] - 1:self.goal[0] + 1] = GOAL_MARKER

        return np.expand_dims(observation, axis=2).astype(np.uint8)  # Convert to single channel image and uint8 to save memory

    def _update_locations(self, new_locations):
        """
        Updates the position for each particle based on the new locations, if the new location is valid (not blocked).
        :param new_locations (int)
        """
        # validate_locations = (np.array([self.freespace[tuple(new_loc.T)]]) & (0 <= new_loc[:, 0]) & (new_loc[:, 0] < self.height) & (0 <= new_loc[:, 1]) & (new_loc[:, 1] < self.width)).transpose()
        valid_locations = (np.array([self.freespace[tuple(new_locations.T)]])).transpose()  # Border does not need to be checked as long as all maps have borders.
        self.particle_locations = np.where(valid_locations, new_locations, self.particle_locations)
