from stable_baselines import PPO2
from stable_baselines.common import set_global_seeds
from stable_baselines.common.atari_wrappers import MaxAndSkipEnv, FrameStack
from stable_baselines.common.policies import CnnPolicy, MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines.common.env_checker import check_env
from stable_baselines.common import make_vec_env

from env.gym_maze.rewards import GoalRewardGenerator, ContinuousRewardGenerator
from env.gym_maze.envs.MazeBase import MazeBase
import gym
import logging


if __name__ == '__main__':
    env = make_vec_env("CartPole-v1", n_envs=4)
    obs = env.reset()


    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=10000)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()