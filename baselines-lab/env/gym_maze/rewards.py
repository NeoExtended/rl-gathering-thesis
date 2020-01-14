from abc import ABC

import numpy as np

class RewardGenerator(ABC):
    def __init__(self, costmap, goal_range, robot_count):
        self.costmap = costmap
        self.goal_range = goal_range
        self.robot_count = robot_count

    def reset(self, robot_locations):
        self.initial_robot_locations = np.copy(robot_locations)

    def step(self, action, locations):
        pass


class GoalRewardGenerator(RewardGenerator):
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
        elif max_cost_agent <= 4 * self.goal_range and not self.reward_grad[1]:
            self.reward_grad[1] = 1
            reward += 4
        elif max_cost_agent <= 8 * self.goal_range and not self.reward_grad[2]:
            self.reward_grad[2] = 1
            reward += 4
        elif max_cost_agent <= 12 * self.goal_range and not self.reward_grad[3]:
            self.reward_grad[3] = 1
            reward += 4

        if cost_to_go <= self.goal_range * self.robot_count and not self.reward_grad[20]:
            self.reward_grad[20] = 1
            reward += 2
        elif cost_to_go <= 2 * self.goal_range * self.robot_count and not self.reward_grad[21]:
            self.reward_grad[21] = 1
            reward += 2
        elif cost_to_go <= 4 * self.goal_range * self.robot_count and not self.reward_grad[22]:
            self.reward_grad[22] = 1
            reward += 2
        elif cost_to_go <= 8 * self.goal_range * self.robot_count and not self.reward_grad[23]:
            self.reward_grad[23] = 1
            reward += 2
        elif cost_to_go <= 12 * self.goal_range * self.robot_count and not self.reward_grad[24]:
            self.reward_grad[24] = 1
            reward += 2

        return done, reward

class ContinuousRewardGenerator(RewardGenerator):
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
