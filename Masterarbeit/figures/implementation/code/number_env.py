import numpy as np
import gym
from gym.utils import seeding

class NumberEnv(gym.Env):
    def __init__(self, max_num: int = 100):
        # Define actions and observations
        self.action_map = {0: 1, 1: 5, 2: 10, 3: -1, 4: -5, 5: -10}
        self.action_space = gym.spaces.Discrete(len(self.action_map))
        self.observation_space = gym.spaces.Box(low=0, high=max_num, shape=(2,), dtype=int)

        # Initialize variables
        self.np_random = np.random.default_rng()
        self.current_number = 0
        self.target_number = 0
        self.max_num = max_num

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.current_number = np.random.randint(0, self.max_num)
        self.target_number = np.random.randint(0, self.max_num)
        return self._make_observation()

    def step(self, action):
        done = False
        new_number = self.current_number + self.action_map[action]
        reward = abs(self.current_number - self.target_number) - abs(self.target_number - new_number)
        self.current_number = new_number
        if self.current_number == self.target_number:
            done = True
        return self._make_observation(), reward, done, {}

    def _make_observation(self) -> np.ndarray:
        return np.array([self.current_number, self.target_number])

    def render(self, mode='human'):
        pass
