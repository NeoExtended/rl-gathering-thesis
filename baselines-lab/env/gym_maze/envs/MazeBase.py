import gym
import numpy as np
import collections
import matplotlib.pyplot as plt
import cv2
from gym.utils import seeding

from env.gym_maze.rewards import GoalRewardGenerator, ContinuousRewardGenerator

ROBOT_MARKER = 150
GOAL_MARKER = 200
BACKGROUND_COLOR = (120, 220, 240)
ROBOT_COLOR = (150, 150, 180)


class MazeBase(gym.Env):
    """
    Base class for a maze-like environment for particle navigation tasks.
    :param map_file: (str) *.csv file containing the map data.
    :param goal: (list) A point coordinate in form [x, y] ([column, row]).
        Can be set to None for a random goal position.
    :param goal_range: (list) Circle radius around the goal position that should be counted as goal reached.
    :param reward_generator: (RewardGenerator) A class of type RewardGenerator generating reward
        based on the current state.
    :param robot_count: (int) Number of robots/particles to spawn in the maze.
        Can be set to -1 for a random number of particles.
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, map_file, goal, goal_range, reward_generator=ContinuousRewardGenerator, robot_count=256):
        self.np_random = None
        self.seed()
        self.freespace = np.loadtxt(map_file).astype(int) # 1: Passable terrain, 0: Wall
        self.maze = np.ones(self.freespace.shape, dtype=int)-self.freespace # 1-freespace: 0: Passable terrain, 0: Wall
        self.cost = None

        if robot_count < 0:
            self.randomize_n_robots = True
            self.robot_count = 0
        else:
            self.randomize_n_robots = False
            self.robot_count = robot_count

        self.reward_generator = reward_generator(self.cost, goal_range, self.robot_count)
        self.actions = [0, 1, 2, 3, 4, 5, 6, 7]  # {S, SE, E, NE, N, NW, W, SW}
        self.action_map = {0: (1, 0), 1: (1, 1), 2: (0, 1), 3: (-1, 1),
                           4: (-1, 0), 5: (-1, -1), 6: (0, -1), 7: (1, -1)}
        self.rev_action_map = {v : k for k, v in self.action_map.items()}
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(*self.maze.shape, 1), dtype=np.uint8)
        self.height, self.width = self.maze.shape

        self.goal_range = goal_range
        if goal:
            self.randomize_goal = False
            self.goal = goal
            self._calculate_cost_map()
        else:
            self.randomize_goal = True
            self.goal = [0, 0]

        self.robot_locations = []
        self.reset()

    def reset(self):
        locations = np.transpose(np.nonzero(self.freespace))

        # Randomize number of robots if necessary
        if self.randomize_n_robots:
            self.robot_count = self.np_random.randint(1, len(locations) // 5)
            self.reward_generator.set_robot_count(self.robot_count)
            print(self.robot_count)

        # Randomize goal position if necessary
        if self.randomize_goal:
            new_goal = locations[self.np_random.randint(0, len(locations))]
            self.goal = [new_goal[1], new_goal[0]]
            self._calculate_cost_map()

        # Reset robot positions
        choice = self.np_random.choice(len(locations), self.robot_count, replace=False)
        self.robot_locations = locations[choice, :] # Robot Locations are in y, x (row, column) order
        self.reward_generator.reset(self.robot_locations)
        return self._generate_observation()

    def step(self, action):
        info = {}
        dy, dx = self.action_map[action]

        for i, (ry, rx) in enumerate(self.robot_locations):
            x2, y2 = rx + dx, ry + dy
            if 0 <= x2 < self.width and 0 <= y2 < self.height and self.freespace[y2, x2] == 1:
                self.robot_locations[i] = [y2, x2]

        done, reward = self.reward_generator.step(action, self.robot_locations)
        return (self._generate_observation(), reward, done, info)

    def render(self, mode='human'):
        rgb_image = np.full((*self.maze.shape, 3), BACKGROUND_COLOR, dtype=int)
        maze_rgb = np.stack((self.freespace*255,)*3, axis=-1) # White maze
        rgb_image = rgb_image + maze_rgb
        for y, x in self.robot_locations:
            rgb_image[y-1:y+1, x-1:x+1] = ROBOT_COLOR

        cv2.circle(rgb_image, tuple(self.goal), self.goal_range, (255, 0, 0), thickness=1)
        rgb_image = np.clip(rgb_image, 0, 255)

        if mode == 'human': # Display image
            plt.gcf().clear()  # Clear current figure
            plt.imshow(rgb_image.astype(np.uint8), vmin=0, vmax=255)
            plt.show(False)
            plt.pause(0.0001)
        return rgb_image.astype(np.uint8)

    def _generate_observation(self):
        self.state_img = np.zeros(self.maze.shape)

        self.state_img[self.robot_locations[:, 0], self.robot_locations[:, 1]] = ROBOT_MARKER
        observation = self.state_img + self.maze * 255
        if self.randomize_goal:
            cv2.circle(observation, tuple(self.goal), self.goal_range, (GOAL_MARKER))
            observation[self.goal[1] - 1:self.goal[1] + 1, self.goal[0] - 1:self.goal[0] + 1] = GOAL_MARKER

        return np.expand_dims(observation, axis=2).astype(np.uint8) # Convert to single channel image and uint8 to save memory

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _calculate_cost_map(self):
        """
        Calculates the cost map based on a given goal position via bfs
        """

        queue = collections.deque([self.goal]) # [x, y] pairs in point notation order!
        seen = np.zeros(self.maze.shape, dtype=int)
        seen[self.goal[1], self.goal[0]] = 1
        self.cost = np.zeros(self.maze.shape, dtype=int)

        while queue:
            x, y = queue.popleft()
            for x2, y2 in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                if 0 <= x2 < self.width and 0 <= y2 < self.height and self.maze[y2, x2] != 1 and seen[y2, x2] != 1:
                    queue.append([x2, y2])
                    seen[y2, x2] = 1
                    self.cost[y2, x2] = self.cost[y, x] + 1

        if self.reward_generator:
            self.reward_generator.set_costmap(self.cost)
        return self.cost
