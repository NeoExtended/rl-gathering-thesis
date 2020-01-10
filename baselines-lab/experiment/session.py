import logging
import os
import numpy as np

from utils import util, config_util
from env.environment import create_environment
from model.model import create_model
from model.saver import ModelSaver

class Session:
    def __init__(self, config, lab_mode):
        self.config = config
        util.set_random_seed(self.config)

        log_dir = self.config['meta'].get('log_dir', None)
        self.log = util.create_log_directory(log_dir)
        if self.log:
            config_util.save_config(self.config, os.path.join(self.log, "config.yml"))

        self.env = create_environment(config=config['env'],
                                      algo_name=config['algorithm']['name'],
                                      seed=self.config['meta']['seed'],
                                      log_dir=self.log)
        self.agent = create_model(config['algorithm'], self.env, seed=self.config['meta']['seed'])
        self.lab_mode = lab_mode

    def run(self):
        if self.lab_mode == 'train':
            self._train()
        elif self.lab_mode == 'enjoy':
            self._enjoy()

    def _enjoy(self):
        obs = self.env.reset()
        episode_counter = 0
        while episode_counter < self.config['env']['n_envs']*4:  # Render at least 4 complete episodes
            action, _states = self.agent.predict(obs)
            obs, rewards, dones, info = self.env.step(action)
            self.env.render()
            episode_counter += np.sum(dones)
        self.env.close()

    def _train(self):
        logging.info("Starting training.")
        save_interval = self.config['meta'].get('save_interval', 250000)
        n_keep = self.config['meta'].get('n_keep', 5)
        keep_best = self.config['meta'].get('keep_best', True)

        saver = ModelSaver(
            model_dir=os.path.join(self.log, "savepoints"),
            save_interval=save_interval,
            n_keep=n_keep,
            keep_best=keep_best,
            config=self.config)

        self.agent.learn(self.config['meta']['n_timesteps'], callback=saver.step)

        # Save model at the end of the learning process
        saver.save(self.agent)
        self.env.close()
