from utils import util

class Session:
    def __init__(self, config):
        self.config = config
        util.set_random_seed(self.config)
        self.environment = Environment(config['env'])
        self.agent = Agent(config['agent'], environment=self.environment)


    def run(self):
        self.agent.learn()