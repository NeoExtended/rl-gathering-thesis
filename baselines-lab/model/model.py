import copy
import logging
#from stable_baselines import PPO2, A2C, ACER, ACKTR, DQN, HER, DDPG, TRPO, SAC, TD3
from stable_baselines import PPO2, A2C, ACER, ACKTR, DQN, HER, SAC, TD3

from utils import util

ALGOS = {
    'a2c': A2C,
    'acer': ACER,
    'acktr': ACKTR,
    'dqn': DQN,
#    'ddpg': DDPG, mpi dependency
    'her': HER,
    'sac': SAC,
    'ppo2': PPO2,
#    'trpo': TRPO, mpi dependency
    'td3': TD3
}

def create_model(config, env, seed):
    logging.info("Creating model.")
    config = copy.deepcopy(config)
    name =  config.pop('name')
    tlog = config.pop('tensorboard_log', None)
    verbose = config.pop('verbose', 0)
    policy_config = config.pop('policy')

    tlog_location = _get_tensorflow_log_location(tlog)
    learning_rate = _get_learning_rate(config)

    if 'trained_agent' in config: # Continue training
        logging.info("Loading pretrained model from {}.".format(config['trained_agent']))

        return ALGOS[name].load(
            config['trained_agent'],
            seed=seed,
            env=env,
            tensorboard_log=tlog_location,
            verbose=verbose,
            learning_rate=learning_rate,
            **config)

    else:
        logging.info("Creating new agent.")
        policy_name = policy_config.pop('name')

        return ALGOS[name](
            seed=seed,
            policy=policy_name,
            policy_kwargs=policy_config,
            env=env,
            tensorboard_log=tlog_location,
            verbose=verbose,
            learning_rate=learning_rate,
            **config)


def _get_tensorflow_log_location(tlog):
    if tlog:
        if isinstance(tlog, bool):
            return util.get_log_directory()
        else:
            return tlog.get('path', util.get_log_directory())
    else:
        return None

def _get_learning_rate(config):
    lr = config.pop('learning_rate', 2.5e-4)

    if isinstance(lr, dict):
        name = lr.pop('name')
        if name == 'LinearSchedule':
            return linear_schedule(lr['initial_value'])
        else:
            raise NotImplementedError("Currently only LinearSchedules are supported")
    else:
        return lr


def linear_schedule(initial_value):
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        return progress * initial_value

    return func