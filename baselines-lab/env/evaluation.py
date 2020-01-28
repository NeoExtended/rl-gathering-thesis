import logging
import os

from utils import get_timestamp, config_util


class EpisodeInformationAggregator:
    """
    Class for aggregating and episode information over the course of an enjoy session. Used for evaluation.
    :param num_envs: (int) Number of environments that will be used.
    :param path: (str) Directory to save the results to.
    """

    def __init__(self, num_envs=1, path=None):
        self.successful_reward = 0
        self.successful_steps = 0
        self.n_failed_episodes = 0
        self.n_successful_episodes = 0
        self.failed_reward = 0
        self.failed_steps = 0

        self.reward_buffer = [0] * num_envs
        self.step_buffer = [0] * num_envs
        self.path = path

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

                self.reward_buffer[i] = 0
                self.step_buffer[i] = 0

    def close(self):
        num_episodes = self.n_successful_episodes + self.n_failed_episodes
        success_rate = self.n_successful_episodes / num_episodes
        total_reward = self.successful_reward + self.failed_reward
        total_steps = self.successful_steps + self.failed_steps

        logging.info(
            "Performed {} episodes with a success rate of {:.2%} an average reward of {:.4f} and an average of {:.4f} steps.".format(
                num_episodes,
                success_rate,
                total_reward / num_episodes,
                total_steps / num_episodes)
        )
        logging.info("Success Rate: {}/{} = {:.2%}".format(self.n_successful_episodes, num_episodes, success_rate))
        if self.n_successful_episodes > 0:
            logging.info("Average reward if episode was successful: {:.4f}".format(
                self.successful_reward / self.n_successful_episodes))
            logging.info("Average length if episode was successful: {:.4f}".format(
                self.successful_steps / self.n_successful_episodes))
        if self.n_failed_episodes > 0:
            logging.info("Average reward if episode was not successful: {:.4f}".format(
                self.failed_reward / self.n_failed_episodes))

        if self.path:
            self._save_results(num_episodes, success_rate, total_reward, total_steps)

    def _save_results(self, num_episodes, success_rate, total_reward, total_steps):
        output = dict()
        output['n_episodes'] = num_episodes
        output['n_successful_episodes'] = self.n_successful_episodes
        output['n_failed_episodes'] = self.n_failed_episodes
        output['success_rate'] = round(float(success_rate), 4)
        output['total_reward'] = float(total_reward)
        output['reward_per_episode'] = round(float(total_reward / num_episodes), 4)
        output['avg_episode_length'] = round(float(total_steps / num_episodes), 4)

        if self.n_successful_episodes > 0:
            output['reward_on_success'] = round(float(self.successful_reward / self.n_successful_episodes), 4)
            output['episode_length_on_success'] = round(float(self.successful_steps / self.n_successful_episodes))

        if self.n_failed_episodes > 0:
            output['reward_on_fail'] = round(float(self.failed_reward / self.n_failed_episodes), 4)

        output_path = os.path.join(self.path, "evaluation_{}.yml".format(get_timestamp()))
        logging.info("Saving results to {}".format(output_path))
        config_util.save_config(output, output_path)