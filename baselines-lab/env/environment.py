import importlib
import os
import logging

import gym
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines.common.atari_wrappers import FrameStack
from stable_baselines.common.vec_env import VecFrameStack, SubprocVecEnv, VecNormalize, DummyVecEnv

def make_env(env_id, rank=0, seed=0, log_dir=None, wrappers=None):
    """
    Helper function to multiprocess training and log the progress.
    :param env_id: (str)
    :param rank: (int)
    :param seed: (int)
    :param log_dir: (str)
    :param wrappers: (list) a list with subclasses of gym.Wrapper to wrap the original env with
    """
    def _init():
        set_global_seeds(seed + rank)
        env = gym.make(env_id)

        if wrappers:
            for wrapper in wrappers:
                env = wrapper(env)

        env.seed(seed + rank)
        if log_dir:
            env = Monitor(env, os.path.join(log_dir, str(rank)), allow_early_resets=True)
        return env

    return _init


def get_wrapper_class(wrapper):
    """
    Get a Gym environment wrapper class from a string describing the module and class name
    e.g. env_wrapper: gym_minigrid.wrappers.FlatObsWrapper

    :param wrapper: (string)
    :return: a subclass of gym.Wrapper (class object) you can use to create another Gym env giving an original env.
    """
    if wrapper:
        wrapper_module = importlib.import_module(wrapper.rsplit(".", 1)[0])
        return getattr(wrapper_module, wrapper.split(".")[-1])
    else:
        return None


def create_environment(config, algo_name, seed, log_dir=None):
    logging.info("Creating environment.")
    env_id = config['name']
    n_envs = config.get('n_envs', 1)
    normalize = config.pop('normalize', None)
    frame_stack = config.get('frame_stack', None)
    multiprocessing = config.get('multiprocessing', True)

    wrappers = []
    if 'wrappers' in config:
        for wrapper in config['wrappers']:
            wrappers.append(get_wrapper_class(wrapper))

    if algo_name in ['dqn', 'ddpg']:
        return _create_standard_env(env_id, seed, log_dir, wrappers, normalize, frame_stack)
    else:
        return _create_vectorized_env(env_id, n_envs, multiprocessing, seed, log_dir, wrappers, normalize, frame_stack)


def _create_vectorized_env(env_id, n_envs, multiprocessing, seed, log_dir, wrappers, normalize, frame_stack):
    if n_envs == 1:
        env = DummyVecEnv([make_env(env_id, 0, seed, log_dir, wrappers)])
    else:
        if multiprocessing:
            env = SubprocVecEnv([make_env(env_id, i, seed, log_dir, wrappers) for i in range(n_envs)])
        else:
            env = DummyVecEnv([make_env(env_id, i, seed, log_dir, wrappers) for i in range(n_envs)])

    if normalize:
        if isinstance(normalize, bool):
            env = VecNormalize(env)
        elif isinstance(normalize, dict):
            if 'trained_agent' in normalize:
                env = VecNormalize.load(normalize['trained_agent'], env)
            else:
                env = VecNormalize(env, **normalize)
    if frame_stack:
        env = VecFrameStack(env, **frame_stack)

    return env


def _create_standard_env(env_id, seed, log_dir, wrappers, normalize, frame_stack):
    env_maker = make_env(env_id, 0, seed, log_dir, wrappers)
    env = env_maker()

    if normalize:
        logging.warning("Normalization is not supported for DDPG/DQN methods.")
    if frame_stack:
        env = FrameStack(env, **frame_stack)

    return env

