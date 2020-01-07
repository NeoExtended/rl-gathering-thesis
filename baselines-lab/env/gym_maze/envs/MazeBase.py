import gym
import numpy as np
import collections
import matplotlib.pyplot as plt

from env.gym_maze.rewards import GoalRewardGenerator

ROBOT_MARKER = 150
BACKGROUND_COLOR = (120, 220, 240)
ROBOT_COLOR = (180, 180, 180)

class MazeBase(gym.Env):
    def __init__(self, map_file, goal, goal_range, reward_generator=GoalRewardGenerator, robot_count=256):
        self.freespace = np.loadtxt(map_file).astype(int)
        self.maze = np.ones(self.freespace.shape, dtype=int)-self.freespace
        self.goal = goal
        self.goal_range = goal_range
        self.robot_count = robot_count

        self.actions = [0, 1, 2, 3, 4, 5, 6, 7]  # {N, NE, E, SE, S, SW, W, NW}
        self.action_map = {0: (1, 0), 1: (1, 1), 2: (0, 1), 3: (-1, 1),
                           4: (-1, 0), 5: (-1, -1), 6: (0, -1), 7: (1, -1)}
        self.rev_action_map = {v : k for k, v in self.action_map.items()}
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(*self.maze.shape, 1), dtype=np.uint8)
        self.height, self.width = self.maze.shape

        self._calculate_cost_map(self.goal)
        self.reward_generator = reward_generator(self.cost, self.goal_range, self.robot_count)

    def reset(self):
        locations = np.nonzero(self.freespace)
        choice = np.random.choice(len(locations), self.robot_count, replace=False)
        self.robot_locations = locations[choice, :]
        self.reward_generator.reset() # TODO: Implement reward generator

    def step(self, action):
        info = {}
        dy, dx = self.action_map[action]

        for i, (rx, ry) in enumerate(self.robot_locations):
            if self.maze[ry + dy, rx + dx] == 1:
                self.robot_locations[i] = self.robot_locations[i] + [dy, dx]

        self.state_img = np.zeros(self.maze.shape)

        np.put(self.state_img, self.robot_locations, [ROBOT_MARKER]*self.robot_count)
        observation = self.state_img + self.maze*255

        done, reward = self.reward_generator.step(action) #TODO: Implement reward generator
        return (np.expand_dims(observation, axis=2), reward, done, info)


    def render(self, mode='human'):
        rgb_image = np.full((*self.maze.shape, 3), BACKGROUND_COLOR, dtype=np.uint8)
        for y, x in self.robot_locations:
            rgb_image[y, x] = ROBOT_COLOR

        rgb_image[self.goal[0], self.goal[1]] = [255, 0, 0]

        if mode == 'human': # Display image
            plt.gcf().clear()  # Clear current figure
            plt.imshow(rgb_image, vmin=0, vmax=255)
            plt.show(False)
            plt.pause(0.0001)
        return rgb_image

    def _calculate_cost_map(self, goal_position):
        """
        Calculates the cost map based on a given goal position via bfs
        :param goal_position: Goal position for all particles - start for the costmap
        """

        queue = collections.deque([goal_position])
        seen = np.zeros(self.maze.shape, dtype=int)
        seen[goal_position[0], goal_position[1]] = 1
        self.cost = np.zeros(self.maze.shape, dtype=int)

        while queue:
            x, y = queue.popleft()
            for x2, y2 in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                if 0 <= x2 < self.width and 0 <= y2 < self.height and self.maze[x2, y2] != 1 and seen[x2, y2] != 1:
                    queue.append([x2, y2])
                    seen[x2, y2] = 1
                    self.cost[x2, y2] = self.cost[x, y] + 1

        return self.cost



