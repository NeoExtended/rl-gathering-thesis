from stable_baselines import PPO2
from stable_baselines.common import set_global_seeds
from stable_baselines.common.atari_wrappers import MaxAndSkipEnv, FrameStack, ScaledFloatFrame
from stable_baselines.common.policies import CnnPolicy, MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, VecFrameStack, VecNormalize
from stable_baselines.common.env_checker import check_env
from stable_baselines.common import make_vec_env
from stable_baselines import DQN

from env.wrappers import VecScaledFloatFrame
from env.gym_maze.rewards import GoalRewardGenerator, ContinuousRewardGenerator
from env.gym_maze.envs.MazeBase import MazeBase
import gym
import logging



if __name__ == '__main__':
    env = make_vec_env("CartPole-v0", n_envs=1)
    env2 = make_vec_env("CartPole-v0", n_envs=8, vec_env_cls=SubprocVecEnv)
    # env = VecNormalize(env)
    # env = VecScaledFloatFrame(env)
    #env = VecFrameStack(env, n_stack=4)
    obs = env.reset()


    #model = PPO2(MlpPolicy, env, verbose=1)
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    obs = env2.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env2.step(action)
        env2.render()