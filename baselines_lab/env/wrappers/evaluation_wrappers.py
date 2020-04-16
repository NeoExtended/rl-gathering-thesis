import logging
import os
from collections import deque

import gym
from stable_baselines.common.vec_env import VecEnvWrapper

from baselines_lab.utils import safe_mean, get_timestamp, config_util


class EvaluationWrapper(gym.Wrapper):
    """
    Evaluation class for goal based environments (reaching max_episode_steps is counted as fail)
    :param env: (gym.Env or gym.Wrapper) The environment to wrap.
    :param path: (str) Path to save the evaluation results to.
    """

    def __init__(self, env, path=None):
        gym.Wrapper.__init__(self, env)
        self.aggregator = EpisodeInformationAggregator(num_envs=1, path=path)

    def step(self, action):
        obs, rew, done, info = gym.Wrapper.step(self, action)
        self.aggregator.step([rew], [done], [info])
        return obs, rew, done, info

    def reset_statistics(self):
        self.aggregator.reset_statistics()

    def close(self):
        gym.Wrapper.close(self)
        self.aggregator.close()


class VecEvaluationWrapper(VecEnvWrapper):
    """
    Evaluation class for vectorized goal based environments (reaching max_episode_steps is counted as fail)
    :param env: (VecEnv or VecEnvWrapper) The environment to wrap.
    :param path: (str) Path to save the evaluation results to.
    """

    def __init__(self, env, path=None):
        VecEnvWrapper.__init__(self, env)
        num_envs = env.unwrapped.num_envs
        self.aggregator = EpisodeInformationAggregator(num_envs=num_envs, path=path)

    def reset(self):
        obs = self.venv.reset()
        return obs

    def reset_statistics(self):
        self.aggregator.reset_statistics()

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.aggregator.step(rews, dones, infos)
        return obs, rews, dones, infos

    def close(self):
        VecEnvWrapper.close(self)
        self.aggregator.close()


class EpisodeInformationAggregator:
    """
    Class for aggregating and episode information over the course of an enjoy session. Used for evaluation.
    :param num_envs: (int) Number of environments that will be used.
    :param path: (str) Directory to save the results to.
    :param smoothing: (int) Smoothing factor for running mean averages.
    """

    def __init__(self, num_envs=1, path=None, smoothing=100):
        self.successful_reward = 0
        self.successful_steps = 0
        self.n_failed_episodes = 0
        self.n_successful_episodes = 0
        self.failed_reward = 0
        self.failed_steps = 0

        self.reward_buffer = [0] * num_envs
        self.step_buffer = [0] * num_envs
        self.num_envs = num_envs
        self.path = path
        self.reward_rms_buffer = deque(maxlen=smoothing)
        self.step_rms_buffer = deque(maxlen=smoothing)
        self.smoothing = smoothing

    def step(self, rews, dones, infos):
        for i, (r, done, info) in enumerate(zip(rews, dones, infos)):
            self.reward_buffer[i] += r
            self.step_buffer[i] += 1

            if done:
                # logging.debug("{}; {}".format(self.step_buffer[i], info))
                if info.get('TimeLimit.truncated', False):
                    self.n_failed_episodes += 1
                    self.failed_reward += self.reward_buffer[i]
                    self.failed_steps += self.step_buffer[i]
                else:
                    self.n_successful_episodes += 1
                    self.successful_reward += self.reward_buffer[i]
                    self.successful_steps += self.step_buffer[i]
                self.reward_rms_buffer.append(self.reward_buffer[i])
                self.step_rms_buffer.append(self.step_buffer[i])

                self.reward_buffer[i] = 0
                self.step_buffer[i] = 0

    def reset_statistics(self):
        self.successful_reward = 0
        self.successful_steps = 0
        self.n_failed_episodes = 0
        self.n_successful_episodes = 0
        self.failed_reward = 0
        self.failed_steps = 0

        self.reward_buffer = [0] * self.num_envs
        self.step_buffer = [0] * self.num_envs
        self.reward_rms_buffer = deque(maxlen=self.smoothing)
        self.step_rms_buffer = deque(maxlen=self.smoothing)

    @property
    def total_steps(self):
        return self.successful_steps + self.failed_steps

    @property
    def total_episodes(self):
        return self.n_successful_episodes + self.n_failed_episodes

    @property
    def total_reward(self):
        return float(self.successful_reward + self.failed_reward)

    @property
    def mean_reward(self):
        return float(self.total_reward / self.total_episodes)

    @property
    def mean_steps(self):
        return float(self.total_steps / self.total_episodes)

    @property
    def reward_rms(self):
        return safe_mean(self.reward_rms_buffer)

    @property
    def step_rms(self):
        return safe_mean(self.step_rms_buffer)

    def close(self):
        success_rate = float(self.n_successful_episodes / self.total_episodes)

        logging.info(
            "Performed {} episodes with a success rate of {:.2%} an average reward of {:.4f} and an average of {:.4f} steps.".format(
                self.total_episodes,
                success_rate,
                self.mean_reward,
                self.mean_steps)
        )
        logging.info("Success Rate: {}/{} = {:.2%}".format(self.n_successful_episodes, self.total_episodes, success_rate))
        if self.n_successful_episodes > 0:
            logging.info("Average reward if episode was successful: {:.4f}".format(
                self.successful_reward / self.n_successful_episodes))
            logging.info("Average length if episode was successful: {:.4f}".format(
                self.successful_steps / self.n_successful_episodes))
        if self.n_failed_episodes > 0:
            logging.info("Average reward if episode was not successful: {:.4f}".format(
                self.failed_reward / self.n_failed_episodes))

        if self.path:
            self._save_results(success_rate)

    def _save_results(self, success_rate):
        output = dict()
        output['n_episodes'] = self.total_episodes
        output['n_successful_episodes'] = self.n_successful_episodes
        output['n_failed_episodes'] = self.n_failed_episodes
        output['success_rate'] = round(success_rate, 4)
        output['total_reward'] = self.total_reward
        output['reward_per_episode'] = round(self.mean_reward, 4)
        output['avg_episode_length'] = round(self.mean_steps, 4)

        if self.n_successful_episodes > 0:
            output['reward_on_success'] = round(float(self.successful_reward / self.n_successful_episodes), 4)
            output['episode_length_on_success'] = round(float(self.successful_steps / self.n_successful_episodes))

        if self.n_failed_episodes > 0:
            output['reward_on_fail'] = round(float(self.failed_reward / self.n_failed_episodes), 4)

        output_path = os.path.join(self.path, "evaluation_{}.yml".format(get_timestamp()))
        logging.info("Saving results to {}".format(output_path))
        config_util.save_config(output, output_path)