import logging
import os

from utils import util, config_util
from env.environment import create_environment
from model.model import create_model
from model.saver import ModelSaver

class Session:
    def __init__(self, config):
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

    def run(self):
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