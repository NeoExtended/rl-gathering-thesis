from stable_baselines import PPO2
from stable_baselines.common import set_global_seeds
from stable_baselines.common.atari_wrappers import MaxAndSkipEnv, FrameStack
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines.common.env_checker import check_env

from env.gym_maze.rewards import GoalRewardGenerator, ContinuousRewardGenerator
from env.gym_maze.envs.MazeBase import MazeBase
import gym
import logging


if __name__ == '__main__':
    logging.getLogger().setLevel(10)
    env = gym.make("Maze0318Continuous-v0")
    check_env(env, skip_render_check=False)
    print("Done")