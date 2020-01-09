import logging

from utils import util
from env.environment import create_environment
from model.model import create_model

class Session:
    def __init__(self, config):
        self.config = config
        util.set_random_seed(self.config)
        self.env = create_environment(config=config['env'],
                                      algo_name=config['algorithm']['name'],
                                      seed=self.config['meta']['seed'],
                                      log_dir='.') # TODO: Logdir
        self.agent = create_model(config['algorithm'], self.env)

    def run(self):
        logging.info("Starting training.")
        self.agent.learn(self.config['meta']['timesteps'])