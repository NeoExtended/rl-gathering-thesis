import logging
import os
from collections import deque
from copy import deepcopy

import gym
from gym.utils.seeding import create_seed
from stable_baselines.common.vec_env import VecNormalize, VecEnvWrapper

from experiment import Runner
from utils import get_timestamp, config_util, safe_mean, unwrap_env, unwrap_vec_env


class Evaluator:
    """
    Class for easy model evaluation. Supports multiple evaluation methods for speed/accuracy tradeoff.
    Evaluates average reward and number of steps.

    :param algorithm_name: (str) Name of the used algorithm (needed for environment creation)
    :param n_eval_episodes: (int) Number of episodes for evaluation.
    :param deterministic: (bool) Weather model actions should be deterministic or stochastic.
    :param render: (bool) Weather or not to render the environment during evaluation.
    :param eval_method: (str) One of the available evaluation types ("fast", "normal", "slow").
        Slow will only use a single env and will be the most accurate.
        Normal uses VecEnvs and fast requires env to be set and wrapped in a Evaluation Wrapper.
    :param env_config: (dict) Config used to create the evaluation environment for normal and slow evaluation mode.
    :param env: (gym.Env or VecEnv) Environment used in case of eval_mode=="fast". Must be wrapped in an evaluation wrapper.
    :param seed: (int) Seed for the evaluation environment. If None a random seed will be generated.
    """
    def __init__(self, algorithm_name, n_eval_episodes=32, deterministic=True, render=False, eval_method="normal",
                 env_config=None, env=None, seed=None):
        self.eval_method = eval_method
        self.config = env_config
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.render = render

        if eval_method in ["normal", "slow"]:
            assert env_config, "You must provide an environment configuration, if the eval_method is not fast!"
            test_env_config = deepcopy(env_config)
            if eval_method == "slow":
                test_env_config['num_envs'] = 1

            if not test_env_config.get('n_envs', None):
                test_env_config['n_envs'] = 8

            from env.environment import create_environment
            if not seed:
                seed = create_seed()
            self.test_env = create_environment(test_env_config,
                                               algorithm_name,
                                               seed,
                                               evaluation=True)
            self.eval_wrapper = unwrap_env(self.test_env, VecEvaluationWrapper, EvaluationWrapper)
        elif eval_method == "fast":
            assert env, "You must provide an environment with an EvaluationWrapper if the eval_method is fast!"
            self.test_env = None
            self.eval_wrapper = unwrap_env(self.test_env, VecEvaluationWrapper, EvaluationWrapper)
        else:
            raise AttributeError("Unknown eval method '{}'".format(eval_method))

    def evaluate(self, model):
        """
        Evaluates the given model on the evaluation environment.
        """
        if self.eval_method == "fast":
            return self._evaluate_fast()
        else:
            return self._evaluate_normal(model)

    def _evaluate_fast(self):
        return self.eval_wrapper.aggregator.reward_rms, self.eval_wrapper.aggregator.step_rms

    def _evaluate_normal(self, model):
        self.eval_wrapper.reset_statistics()

        if self.config.get('normalize', None): # Update normalization running means if necessary
            norm = unwrap_vec_env(self.test_env, VecNormalize)
            model_norm = unwrap_vec_env(model.env, VecNormalize)
            norm.obs_rms = model_norm.obs_rms
            norm.ret_rms = model_norm.ret_rms
            norm.training = False

        runner = Runner(self.test_env, model, render=self.render, deterministic=self.deterministic, close_env=False)
        runner.run(self.n_eval_episodes)
        return self.eval_wrapper.aggregator.mean_reward, self.eval_wrapper.aggregator.mean_steps


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