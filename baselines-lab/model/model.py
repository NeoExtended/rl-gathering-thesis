import copy
import logging
#from stable_baselines import PPO2, A2C, ACER, ACKTR, DQN, HER, DDPG, TRPO, SAC, TD3
from stable_baselines import PPO2, A2C, ACER, ACKTR, DQN, HER, SAC, TD3

from model.schedules import get_schedule

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

    # Create lr schedules if supported
    if name in ["ppo2", "sac", "td3"]:
        for key in ['learning_rate', "cliprange", "cliprange_vf"]:
            if key in config and isinstance(config[key], dict):
                config[key] = get_schedule(config[key].pop("type"), **config[key])

    if 'trained_agent' in config: # Continue training
        logging.info("Loading pretrained model from {}.".format(config['trained_agent']))

        return ALGOS[name].load(
            config['trained_agent'],
            seed=seed,
            env=env,
            tensorboard_log=tlog_location,
            verbose=verbose,
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
            **config)


def _get_tensorflow_log_location(tlog):
    if tlog:
        if isinstance(tlog, bool):
            return util.get_log_directory()
        else:
            return tlog.get('path', util.get_log_directory())
    else:
        return None
