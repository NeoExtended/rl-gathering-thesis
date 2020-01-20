from collections import deque
import time

import tensorflow as tf

from stable_baselines.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from utils import safe_mean


class TensorboardLogger:
    """
    Logs additional values into the tensorboard log. Can be used as a callback for all learning algorithms.
    :param n_envs: Number of environments that are used in the training process
    :param smoothing: Number of episodes over which the running average for episode length and return
        will be calculated.
    """
    def __init__(self, n_envs, smoothing=100):
        self.ep_len_buffer = deque(maxlen=smoothing)
        self.reward_buffer = deque(maxlen=smoothing)
        self.n_episodes = [0]*n_envs
        self.t_start = time.time()
        self.last_timesteps = 0

    def step(self, locals_, globals_):
        """
        Callback for learning algorithms
        """
        self_ = locals_['self']
        env = self_.env

        self._retrieve_values(env)
        self._write_summary(locals_['writer'], self_.num_timesteps)
        return True

    def _retrieve_values(self, env):
        """
        This method makes use of methods from the Monitor environment wrapper to retrieve episode information
        independent of the used algorithm.
        """
        if isinstance(env, VecEnv) or isinstance(env, VecEnvWrapper):
            self._retrieve_from_vec_env(env)
        else:
            self._retrieve_from_env(env)

    def _retrieve_from_env(self, env):
        episode_rewards = env.get_episode_rewards()
        episode_lengths = env.get_episode_lengths()

        known = self.n_episodes[0]
        self.ep_len_buffer.extend(episode_lengths[known:])
        self.reward_buffer.extend(episode_rewards[known:])
        self.n_episodes[0] = len(episode_rewards)

    def _retrieve_from_vec_env(self, env):
        # Use methods indirectly if we are dealing with a vecotorized environment
        episode_rewards = env.env_method("get_episode_rewards")
        episode_lengths = env.env_method("get_episode_lengths")

        for i, (ep_reward, ep_length) in enumerate(zip(episode_rewards, episode_lengths)):
            known = self.n_episodes[i]
            self.ep_len_buffer.extend(ep_length[known:])
            self.reward_buffer.extend(ep_reward[known:])
            self.n_episodes[i] = len(ep_reward)

    def _write_summary(self, writer, num_timesteps):
        if len(self.ep_len_buffer) > 0:
            reward_summary = tf.Summary(value=[
                    tf.Summary.Value(
                        tag='ep_reward_mean',
                        simple_value=safe_mean(self.ep_len_buffer))
            ])
            length_summary = tf.Summary(value=[
                tf.Summary.Value(
                    tag='ep_length_mean',
                    simple_value=safe_mean(self.reward_buffer))
            ])
            writer.add_summary(reward_summary, num_timesteps)
            writer.add_summary(length_summary, num_timesteps)

        steps = num_timesteps - self.last_timesteps
        t_now = time.time()
        fps = int(steps / (t_now - self.t_start))
        self.t_start = t_now
        self.last_timesteps = num_timesteps

        fps_summary = tf.Summary(value=[
            tf.Summary.Value(
                tag='steps_per_second',
                simple_value=fps
            )
        ])
        writer.add_summary(fps_summary, num_timesteps)
