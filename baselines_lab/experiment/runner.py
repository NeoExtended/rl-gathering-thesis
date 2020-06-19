import numpy as np
from stable_baselines.common.vec_env import VecEnv


class Runner:
    """
    Executes the basic agent - environment interaction loop.
    :param env: (gym.Env or VecEnv) The environment for the agent
    :param agent: The agent to make predictions from observations. If None, random actions will be generated.
    :param render: (bool) Weather or not to render the environment during execution.
    :param deterministic: (bool) Weather or not the agent should make deterministic predictions.
    :param close_env: (bool) Weather or not the environment should be closed after a call to run().
    """
    def __init__(self, env, agent, render=True, deterministic=True, close_env=True):
        self.env = env
        self.agent = agent
        self.deterministic = deterministic
        self.render = render
        self.close_env = close_env

        self.episode_counter = 0
        self.step_counter = 0
        self.total_reward = 0

        if isinstance(env, VecEnv):
            self.vec_env = True
        else:
            self.vec_env = False

    def run(self, n_episodes):
        """
        Simulates at least n_episodes of environment interactions.
        """
        obs = self.env.reset()

        while self.episode_counter < n_episodes:
            if self.agent:
                action, _states = self.agent.predict(obs, deterministic=self.deterministic)
            else:
                action = [self.env.action_space.sample() for i in range(self.env.unwrapped.num_envs)]
            obs, rewards, dones, info = self.env.step(action)
            if self.render:
                self.env.render(mode='human')
            self.update_values(obs, rewards, dones, info)

            # Reset env after done. DummyVecEnv and SubprocVecEnvs have an integrated reset mechanism.
            if not self.vec_env:
                if dones:
                    self.env.reset()

        # logging.info("Performed {} episodes with an avg length of {} and {} avg reward".format(episode_counter, step_counter / episode_counter, total_reward / episode_counter))
        if self.close_env:
            self.env.close()

    def update_values(self, obs, reward, done, info):
        if self.vec_env:
            self.episode_counter += np.sum(done)
            self.step_counter += len(obs)
            self.total_reward += sum(reward)
        else:
            self.episode_counter += done
            self.step_counter += 1
            self.total_reward += reward