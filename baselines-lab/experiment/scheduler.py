import logging

from experiment import Session
from utils import send_email


class Scheduler:
    """
    The scheduler class manages the execution of the lab experiments. If multiple experiments have been specified they are run one by one.
    :param configs: (list) List of lab configs to execute.
    :param args: (namespace) Lab arguments.
    """

    def __init__(self, configs, args):
        self.configs = configs
        self.args = args

    def run(self):
        for config in self.configs:
            success = False
            try:
                session = Session.create_session(config, self.args)
                session.run()
                logging.info("Finished execution of config {}".format(config))
                success = True
            except Exception:
                logging.warning("An exception occurred when executing config {} with args {}".format(config, self.args))

            if self.args.mail:
                if success:
                    send_email(self.args.mail, "Finished Training", "Finished training for config {} with args {}.".format(config, self.args))
                else:
                    send_email(self.args.mail, "Run Failed", "Training for config {} with args {} failed.".format(config, self.args))