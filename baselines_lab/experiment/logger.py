import time
from collections import deque

import tensorflow as tf
from stable_baselines.common.vec_env.base_vec_env import VecEnv

from baselines_lab.utils import safe_mean, unwrap_vec_env
from baselines_lab.env.wrappers import CuriosityWrapper


class TensorboardLogger:
    """
    Logs additional values into the tensorboard log. Can be used as a callback for all learning algorithms.
    :param smoothing: (int) Number of episodes over which the running average for episode length and return
        will be calculated.
    :param min_log_delay: (int) Minimum number of timesteps between log entries.
    """
    def __init__(self, smoothing=100, min_log_delay=500):
        self.ep_len_buffer = deque(maxlen=smoothing)
        self.reward_buffer = deque(maxlen=smoothing)
        self.extrinsic_rew_buffer = deque(maxlen=smoothing)
        self.intrinsic_rew_buffer = deque(maxlen=smoothing)
        self.n_episodes = None
        self.t_start = time.time()
        self.last_timesteps = 0
        self.first_step = True
        self.min_log_delay = min_log_delay
        self.curiosity_wrapper = None

    def step(self, locals_, globals_):
        """
        Callback for learning algorithms
        """
        self_ = locals_['self']
        env = self_.env
        timesteps = self_.num_timesteps

        if self.first_step:
            self._initialize(env)
            self.first_step = False

        if timesteps - self.last_timesteps > self.min_log_delay:
            self._retrieve_values(env)
            self._write_summary(locals_['writer'], timesteps)
        return True

    def reset(self):
        """
        Resets the logger.
        """
        self.ep_len_buffer.clear()
        self.reward_buffer.clear()
        self.n_episodes = None
        self.first_step = True
        self.last_timesteps = 0
        self.t_start = time.time()

    def _initialize(self, env):
        """
        Initializes the logger in the first step by retrieving the number of used environments.
        """
        if isinstance(env, VecEnv):
            episode_rewards = env.env_method("get_episode_rewards")
            self.n_episodes = [0] * len(episode_rewards)

            unwrapped_env = unwrap_vec_env(env, CuriosityWrapper)
            if isinstance(unwrapped_env, CuriosityWrapper):
                if unwrapped_env.monitor:
                    self.curiosity_wrapper = unwrapped_env

        else:
            self.n_episodes = [0]

    def _retrieve_values(self, env):
        """
        This method makes use of methods from the Monitor environment wrapper to retrieve episode information
        independent of the used algorithm.
        """
        if isinstance(env, VecEnv):
            if self.curiosity_wrapper:
                self._retrieve_from_vec_env_with_curiosity(env)
            else:
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

    def _retrieve_from_vec_env_with_curiosity(self, env):
        # Use methods indirectly if we are dealing with a vecotorized environment
        episode_rewards = env.env_method("get_episode_rewards")
        episode_lengths = env.env_method("get_episode_lengths")
        extrinsic_rewards = self.curiosity_wrapper.get_extrinsic_rewards()
        intrinsic_rewards = self.curiosity_wrapper.get_intrinsic_rewards()

        for i, (ep_reward, ep_length, ext_rews, int_rews) in enumerate(zip(episode_rewards, episode_lengths, extrinsic_rewards, intrinsic_rewards)):
            known = self.n_episodes[i]
            self.ep_len_buffer.extend(ep_length[known:])
            self.reward_buffer.extend(ep_reward[known:])
            self.intrinsic_rew_buffer.extend(int_rews[known:])
            self.extrinsic_rew_buffer.extend(ext_rews[known:])
            self.n_episodes[i] = len(ep_reward)

    def _write_summary(self, writer, num_timesteps):
        if len(self.ep_len_buffer) > 0:
            length_summary = tf.Summary(value=[
                    tf.Summary.Value(
                        tag='episode_length/ep_length_mean',
                        simple_value=safe_mean(self.ep_len_buffer))
            ])
            writer.add_summary(length_summary, num_timesteps)

            reward_summary = tf.Summary(value=[
                tf.Summary.Value(
                    tag='reward/ep_reward_mean',
                    simple_value=safe_mean(self.reward_buffer))
            ])
            writer.add_summary(reward_summary, num_timesteps)

            if self.curiosity_wrapper:
                ext_rew_summary = tf.Summary(value=[
                    tf.Summary.Value(
                        tag='curiosity/ep_ext_reward_mean',
                        simple_value=safe_mean(self.extrinsic_rew_buffer))
                ])
                writer.add_summary(ext_rew_summary, num_timesteps)

                int_rew_summary = tf.Summary(value=[
                    tf.Summary.Value(
                        tag='curiosity/ep_int_reward_mean',
                        simple_value=safe_mean(self.intrinsic_rew_buffer))
                ])
                writer.add_summary(int_rew_summary, num_timesteps)

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

