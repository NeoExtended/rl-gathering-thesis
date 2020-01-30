import numpy as np

class Runner:
    def __init__(self, env, agent, render=True, deterministic=True, close_env=True):
        self.env = env
        self.agent = agent
        self.deterministic = deterministic
        self.render = render
        self.close_env = close_env

    def run(self, n_episodes):
        """
        Simulates at least n_episodes of environment interactions.
        """
        obs = self.env.reset()
        episode_counter = 0
        step_counter = 0
        while episode_counter < n_episodes:
            action, _states = self.agent.predict(obs, deterministic=self.deterministic)
            obs, rewards, dones, info = self.env.step(action)
            if self.render:
                self.env.render()
            episode_counter += np.sum(dones)
            step_counter += len(obs)

        # logging.info("Performed {} episodes with an avg length of {}".format(episode_counter, step_counter / episode_counter))
        if self.close_env:
            self.env.close()