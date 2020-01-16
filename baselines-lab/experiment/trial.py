from experiment.session import Session

class Trial:
    """
    TODO?
    Runs multiple sessions with the same configuration. Gathers statistics over sessions.
    """
    def __init__(self, config):
        self.config = config

    def run(self):
        for s in self.config['max_sessions']:
            Session(self.config).run()

