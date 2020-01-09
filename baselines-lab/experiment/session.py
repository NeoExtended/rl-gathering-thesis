import logging
import os

from utils import util, config_util
from env.environment import create_environment
from model.model import create_model

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
        self.agent = create_model(config['algorithm'], self.env)

    def run(self):
        logging.info("Starting training.")
        self.agent.learn(self.config['meta']['n_timesteps'])