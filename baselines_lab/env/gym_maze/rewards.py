import collections
from abc import ABC, abstractmethod

import numpy as np


class RewardGenerator(ABC):
    """
    Base Class for reward generators for the maze environments
    :param maze: (np.ndarray) two dimensional array defining the maze where 0 indicates passable terrain and 1 indicates an obstacle
    :param goal: (list) A point coordinate in form [x, y] ([column, row]) defining the goal.
    :param goal_range: (int) Range around the goal position which should be treated as 'goal reached'.
    :param n_particles: (int) Total number of robots.
    :param action_map: (dict) Map containing allowed actions.
    """
    def __init__(self, maze, goal, goal_range, n_particles, action_map):
        self.goal_range = goal_range
        self.n_particles = n_particles
        self.initial_robot_locations = None
        self.action_map = action_map

        self._calculate_cost_map(maze, goal)

    def reset(self, locations):
        self.initial_robot_locations = np.copy(locations)

    def set_particle_count(self, n_particles):
        self.n_particles = n_particles

    @abstractmethod
    def step(self, action, locations):
        pass

    def _calculate_cost_map(self, maze, goal):
        """
        Calculates the cost map based on a given goal position via bfs
        """
        queue = collections.deque([goal])  # [x, y] pairs in point notation order!
        seen = np.zeros(maze.shape, dtype=int)
        seen[goal[1], goal[0]] = 1
        self.cost = np.zeros(maze.shape, dtype=int)
        height, width = maze.shape

        while queue:
            x, y = queue.popleft()
            for action in self.action_map.values():
                x2, y2 = x + action[1], y + action[0]
                if 0 <= x2 < width and 0 <= y2 < height and maze[y2, x2] != 1 and seen[y2, x2] != 1:
                    queue.append([x2, y2])
                    seen[y2, x2] = 1
                    self.cost[y2, x2] = self.cost[y, x] + 1


class GoalRewardGenerator(RewardGenerator):
    """
    Gives rewards based on achievement of certain goals. Rewards are only granted a single time.
    Current rewards include measurements about the average and maximum distance to the goal position.
    Also induces a secondary goal of minimizing episode length by adding a constant negative reward.
    """
    def __init__(self, maze, goal, goal_range, n_particles, action_map, n_subgoals=30, final_reward=100, min_performance=0.95, min_reward=2, max_reward=4):
        super().__init__(maze, goal, goal_range, n_particles, action_map)
        self.final_reward = final_reward
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.n_subgoals = n_subgoals
        self.min_performance = min_performance

        self.reward_scale = np.rint(np.linspace(min_reward, max_reward, n_subgoals))
        self.max_cost_reward_goals = None
        self.avg_cost_reward_goals = None

        self.next_max_cost_goal = 0
        self.next_avg_cost_goal = 0

    def reset(self, locations):
        super().reset(locations)
        cost = self.cost.ravel()[(locations[:, 1] + locations[:, 0] * self.cost.shape[1])]
        total_cost = np.sum(cost)
        max_cost = np.max(cost)

        self.max_cost_reward_goals = np.flip(np.rint(np.linspace(2 * self.goal_range, self.min_performance * max_cost, self.n_subgoals)))
        self.avg_cost_reward_goals = np.flip(np.rint(np.linspace(2 * self.goal_range * self.n_particles, self.min_performance * total_cost, self.n_subgoals)))
        self.next_max_cost_goal = 0
        self.next_avg_cost_goal = 0

    def step(self, action, locations):
        step_cost = self.cost.ravel()[(locations[:, 1] + locations[:, 0] * self.cost.shape[1])]
        cost_to_go = np.sum(step_cost)
        max_cost_agent = np.max(step_cost)

        done = False
        reward = -0.1

        if max_cost_agent <= self.goal_range:
            done = True
            reward += self.final_reward
            return done, reward

        if self.next_max_cost_goal < self.n_subgoals and max_cost_agent <= self.max_cost_reward_goals[self.next_max_cost_goal]:
            reward += self.reward_scale[self.next_max_cost_goal]
            self.next_max_cost_goal += 1

        if self.next_avg_cost_goal < self.n_subgoals and cost_to_go <= self.avg_cost_reward_goals[self.next_avg_cost_goal]:
            reward += self.reward_scale[self.next_avg_cost_goal]
            self.next_avg_cost_goal += 1

        return done, reward


class ContinuousRewardGenerator(RewardGenerator):
    """
    Gives a continuous reward signal after every step based on the total cost-to-go. The cost is normalized by the
    initial cost. Also induces a secondary goal of minimizing episode length by adding a constant negative reward.
    :param gathering_reward: (float) Scaling factor for the gathering reward
    :param positive_only: (bool) Weather or not to suppress negative rewards from moving particles further away from the goal position.
        Note that positive rewards will not be granted more than once in this setting.
    """
    def __init__(self, maze, goal, goal_range, n_particles, action_map, gathering_reward=0.0, positive_only=False):
        super().__init__(maze, goal, goal_range, n_particles, action_map)
        self.initialCost = 0
        self.lastCost = 0
        self.uniqueParticles = n_particles
        self.gathering_reward_scale = gathering_reward
        self.time_penalty = 0.0
        self.positive_only = positive_only

    def reset(self, locations):
        super().reset(locations)
        self.initialCost = np.sum(self.cost.ravel()[(locations[:, 1] + locations[:, 0] * self.cost.shape[1])])
        self.lastCost = self.initialCost

        max_cost = np.max(self.cost)
        self.time_penalty = 1 / (max_cost * np.log(self.initialCost))

    def step(self, action, locations):
        done = False
        step_cost = self.cost.ravel()[(locations[:, 1] + locations[:, 0] * self.cost.shape[1])]

        gathering_reward = self._calculate_gathering_reward(locations)
        goal_reward = self._calculate_goal_reward(step_cost)
        reward = gathering_reward + goal_reward - self.time_penalty     # -0.001

        if np.max(step_cost) <= self.goal_range:
            done = True

        return done, reward

    def _calculate_goal_reward(self, step_cost):
        cost_to_go = np.sum(step_cost)
        goal_reward = 0.0

        if self.positive_only:
            if self.lastCost - cost_to_go > 0:
                goal_reward = ((self.lastCost - cost_to_go) / self.initialCost)
                self.lastCost = cost_to_go
        else:
            goal_reward = ((self.lastCost - cost_to_go) / self.initialCost)
            self.lastCost = cost_to_go

        return goal_reward

    def _calculate_gathering_reward(self, locations):
        gathering_reward = 0.0
        if self.gathering_reward_scale > 0.0:
            particles = len(np.unique(locations, axis=0))
            gathering_reward = self.gathering_reward_scale * ((self.uniqueParticles - particles) / self.n_particles)
            self.uniqueParticles = particles

        return gathering_reward


GENERATORS = {
    'goal': GoalRewardGenerator,
    'continuous': ContinuousRewardGenerator
}