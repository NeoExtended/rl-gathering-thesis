from abc import ABC, abstractmethod

import numpy as np


class RewardGenerator(ABC):
    """
    Base Class for reward generators for the maze environments
    :param costmap (np.ndarray) two dimensional array defining a cost-to-go value (distance) for each pixel to the
        the goal position.
    :param goal_range: (int) Range around the goal position which should be treated as 'goal reached'.
    :param robot_count: (int) Total number of robots.
    """
    def __init__(self, costmap, goal_range, robot_count):
        self.costmap = costmap
        self.goal_range = goal_range
        self.robot_count = robot_count

    def reset(self, robot_locations):
        self.initial_robot_locations = np.copy(robot_locations)

    def set_robot_count(self, num_robots):
        self.robot_count = num_robots

    def set_costmap(self, costmap):
        self.costmap = costmap

    @abstractmethod
    def step(self, action, locations):
        pass


class GoalRewardGenerator(RewardGenerator):
    """
    Gives rewards based on achievement of certain goals. Rewards are only granted a single time.
    Current rewards include measurements about the average and maximum distance to the goal position.
    Also induces a secondary goal of minimizing episode length by adding a constant negative reward.
    """
    def __init__(self, costmap, goal_range, robot_count):
        super().__init__(costmap, goal_range, robot_count)
        self.reward_grad = np.zeros(40).astype(np.uint8)

    def step(self, action, locations):
        cost_to_go = np.sum(self.costmap[locations[:, 0], locations[:, 1]])
        max_cost_agent = np.max(self.costmap[locations[:, 0], locations[:, 1]])

        done = False
        reward = -0.1

        if max_cost_agent <= self.goal_range:
            done = True
            reward += 100
            return done, reward
        elif max_cost_agent <= 2 * self.goal_range and not self.reward_grad[0]:
            self.reward_grad[0] = 1
            reward += 4
        elif max_cost_agent <= 3 * self.goal_range and not self.reward_grad[1]:
            self.reward_grad[1] = 1
            reward += 4
        elif max_cost_agent <= 4 * self.goal_range and not self.reward_grad[2]:
            self.reward_grad[2] = 1
            reward += 4
        elif max_cost_agent <= 5 * self.goal_range and not self.reward_grad[3]:
            self.reward_grad[3] = 1
            reward += 4
        elif max_cost_agent <= 7 * self.goal_range and not self.reward_grad[4]:
            self.reward_grad[4] = 1
            reward += 2
        elif max_cost_agent <= 9 * self.goal_range and not self.reward_grad[5]:
            self.reward_grad[5] = 1
            reward += 2

        if cost_to_go <= self.goal_range * self.robot_count and not self.reward_grad[20]:
            self.reward_grad[20] = 1
            reward += 4
        elif cost_to_go <= 2 * self.goal_range * self.robot_count and not self.reward_grad[21]:
            self.reward_grad[21] = 1
            reward += 4
        elif cost_to_go <= 3 * self.goal_range * self.robot_count and not self.reward_grad[22]:
            self.reward_grad[22] = 1
            reward += 2
        elif cost_to_go <= 4 * self.goal_range * self.robot_count and not self.reward_grad[23]:
            self.reward_grad[23] = 1
            reward += 2
        elif cost_to_go <= 6 * self.goal_range * self.robot_count and not self.reward_grad[24]:
            self.reward_grad[24] = 1
            reward += 2
        elif cost_to_go <= 8 * self.goal_range * self.robot_count and not self.reward_grad[25]:
            self.reward_grad[25] = 1
            reward += 2

        return done, reward


class ContinuousRewardGenerator(RewardGenerator):
    """
    Gives a continuous reward signal after every step based on the total cost-to-go. The cost is normalized by the
    initial cost. Also induces a secondary goal of minimizing episode length by adding a constant negative reward.
    """
    def __init__(self, costmap, goal_range, robot_count):
        super().__init__(costmap, goal_range, robot_count)
        self.initialCost = 0
        self.lastCost = 0

    def reset(self, robot_locations):
        self.initial_robot_locations = np.copy(robot_locations)
        self.initialCost = np.sum(self.costmap[robot_locations[:, 0], robot_locations[:, 1]])
        self.lastCost = self.initialCost

    def step(self, action, locations):
        done = False
        cost_to_go = np.sum(self.costmap[locations[:, 0], locations[:, 1]])
        max_cost_agent = np.max(self.costmap[locations[:, 0], locations[:, 1]])

        reward = ((self.lastCost - cost_to_go) / self.initialCost) - 0.001
        self.lastCost = cost_to_go

        if max_cost_agent <= self.goal_range:
            done = True

        return done, reward
