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
    """
    def __init__(self, maze, goal, goal_range, n_particles):
        self.goal_range = goal_range
        self.n_particles = n_particles
        self.initial_robot_locations = None

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
            for x2, y2 in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
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
    def __init__(self, maze, goal, goal_range, n_particles, n_subgoals=30, final_reward=100, min_performance=0.95, min_reward=2, max_reward=4):
        super().__init__(maze, goal, goal_range, n_particles)
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
        cost = self.cost[tuple(locations.T)]
        total_cost = np.sum(cost)
        max_cost = np.max(cost)

        self.max_cost_reward_goals = np.flip(np.rint(np.linspace(2 * self.goal_range, self.min_performance * max_cost, self.n_subgoals)))
        self.avg_cost_reward_goals = np.flip(np.rint(np.linspace(2 * self.goal_range * self.n_particles, self.min_performance * total_cost, self.n_subgoals)))
        self.next_max_cost_goal = 0
        self.next_avg_cost_goal = 0

    def step(self, action, locations):
        cost_to_go = np.sum(self.cost[tuple(locations.T)])
        max_cost_agent = np.max(self.cost[tuple(locations.T)])

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
    """
    def __init__(self, maze, goal, goal_range, n_particles, gathering_reward=0.0):
        super().__init__(maze, goal, goal_range, n_particles)
        self.initialCost = 0
        self.lastCost = 0
        self.uniqueParticles = n_particles
        self.gathering_reward_scale = gathering_reward
        self.time_penalty = 0.0

    def reset(self, locations):
        super().reset(locations)
        self.initialCost = np.sum(self.cost[tuple(locations.T)])
        self.lastCost = self.initialCost

        max_cost = np.max(self.cost)
        self.time_penalty = 1 / (max_cost * np.log(self.initialCost))

    def step(self, action, locations):
        done = False
        cost_to_go = np.sum(self.cost[tuple(locations.T)])
        max_cost_agent = np.max(self.cost[tuple(locations.T)])

        if self.gathering_reward_scale > 0.0:
            particles = len(np.unique(locations, axis=0))
            gathering_reward = self.gathering_reward_scale * ((self.uniqueParticles - particles) / self.n_particles)
            self.uniqueParticles = particles
        else:
            gathering_reward = 0

        goal_reward = ((self.lastCost - cost_to_go) / self.initialCost)
        self.lastCost = cost_to_go

        reward = gathering_reward + goal_reward - self.time_penalty     # -0.001

        if max_cost_agent <= self.goal_range:
            done = True

        return done, reward
