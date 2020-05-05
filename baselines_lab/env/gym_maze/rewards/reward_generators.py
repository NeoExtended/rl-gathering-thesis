from abc import abstractmethod
from typing import Tuple

import numpy as np

from baselines_lab.env.gym_maze.rewards.base_reward_generator import RewardGenerator, StepInformationProvider


class TimePenaltyReward(RewardGenerator):
    def __init__(self, information_provider: StepInformationProvider = None, scale: float = 1.0):
        super().__init__(information_provider, scale)
        self.time_penalty = 0.0

    def _reset(self, locations):
        if self.time_penalty == 0.0 or self.calculator.is_relative:
            self._calculate_time_penalty()

    def _step(self, action, locations) -> Tuple[bool, float]:
        done, reward = super()._step(action, locations)
        reward -= self.time_penalty
        return done, reward * self.scale

    def _calculate_time_penalty(self):
        self.time_penalty = 1 / (self.calculator.max_start_cost * np.log(self.calculator.total_start_cost))


class GatheringReward(RewardGenerator):
    def __init__(self, information_provider: StepInformationProvider = None, scale: float = 1.0):
        super().__init__(information_provider, scale)

        self.unique_particles = None

    def _reset(self, locations):
        self.unique_particles = self.calculator.n_particles

    def _step(self, action, locations) -> Tuple[bool, float]:
        done, reward = super()._step(action, locations)
        particle_count = len(self.calculator.unique_particles)
        reward += ((self.unique_particles - particle_count) / self.calculator.n_particles)
        return done, reward * self.scale


class ContinuousTotalCostReward(RewardGenerator):
    def __init__(self, information_provider: StepInformationProvider = None, scale: float = 1.0,
                 normalize: bool = True, positive_only: bool = False):
        super().__init__(information_provider, scale)

        self.normalize = normalize
        self.normalization = 1.0
        self.positive_only = positive_only
        self.last_cost = 0

    def _reset(self, locations):
        if self.normalize:
            self.normalization = self.calculator.total_start_cost
        else:
            self.normalization = self.calculator.n_particles

        self.last_cost = self.calculator.total_start_cost

    def _step(self, action, locations) -> Tuple[bool, float]:
        done, reward = super()._step(action, locations)
        current_cost = self.calculator.total_cost
        if self.positive_only:
            reward += max(0, (self.last_cost - current_cost) / self.normalization)
        else:
            reward += (self.last_cost - current_cost) / self.normalization
        self.last_cost = current_cost

        return done, reward * self.scale


class DiscreteRewardGenerator(RewardGenerator):
    def __init__(self, information_provider: StepInformationProvider = None, scale: float = 1.0,
                 normalize: bool = False, n_subgoals: int = None, min_reward: int = 2, max_reward: int = 4):
        self.n_subgoals = n_subgoals
        self.reward_scale = None
        self.min_reward = min_reward
        self.max_reward = max_reward
        super().__init__(information_provider, scale)

        self.normalize = normalize
        self.normalization = 1.0

        self.reward_goals = None
        self.next_goal = 0

    def set_information_provider(self, calculator: StepInformationProvider):
        super(DiscreteRewardGenerator, self).set_information_provider(calculator)

        if self.n_subgoals is None:
            self.n_subgoals = int(np.max(self.calculator.costmap) / 2)

        self.reward_scale = np.rint(np.linspace(self.min_reward, self.max_reward, self.n_subgoals))

    def _reset(self, locations):
        self.next_goal = 0

    def _step(self, action, locations) -> Tuple[bool, float]:
        done, reward = super(DiscreteRewardGenerator, self)._step(action, locations)

        if self.next_goal < self.n_subgoals and self._goal_reached(action, locations):
            reward += self.reward_scale[self.next_goal]
            self.next_goal += 1

        return done, reward * self.scale

    @abstractmethod
    def _goal_reached(self, action, locations) -> bool:
        pass


class DiscreteMaxCostReward(DiscreteRewardGenerator):
    def _reset(self, locations):
        super(DiscreteMaxCostReward, self)._reset(locations)
        self.reward_goals = np.flip(np.rint(np.linspace(2 * self.calculator.goal_range, 0.95 * self.calculator.max_start_cost, self.n_subgoals)))

    def _goal_reached(self, action, locations) -> bool:
        return self.calculator.max_cost <= self.reward_goals[self.next_goal]


class DiscreteTotalCostReward(DiscreteRewardGenerator):
    def _reset(self, locations):
        super(DiscreteTotalCostReward, self)._reset(locations)
        self.reward_goals = np.flip(np.rint(np.linspace(2 * self.calculator.goal_range * self.calculator.n_particles,
                                                        0.95 * self.calculator.total_start_cost,
                                                        self.n_subgoals)))

    def _goal_reached(self, action, locations) -> bool:
        return self.calculator.total_cost <= self.reward_goals[self.next_goal]