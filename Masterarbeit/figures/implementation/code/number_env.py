import numpy as np
import gym
from gym.utils import seeding

class NumberEnv(gym.Env):
    def __init__(self, max_num: int = 100):
        # Define actions and observations
        self.action_map = {0: 1, 1: 5, 2: 10, 3: -1, 4: -5, 5: -10}
        self.action_space = gym.spaces.Discrete(len(self.action_map))
        self.observation_space = gym.spaces.Box(low=0, high=max_num,
                                                shape=(2,), dtype=int)
        # Initialize variables
        self.np_random = np.random.default_rng()
        self.current = 0
        self.target = 0
        self.max_num = max_num

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.current = np.random.randint(0, self.max_num)
        self.target = np.random.randint(0, self.max_num)
        return self._make_observation()

    def step(self, action):
        new = self.current + self.action_map[action]
        reward = -0.5 + abs(self.current - self.target) - abs(self.target - new)
        self.current = new
        done = True if self.current == self.target else False
        return self._make_observation(), reward, done, {}

    def _make_observation(self) -> np.ndarray:
        return np.array([self.current, self.target])

    def render(self, mode='human'):
        pass
