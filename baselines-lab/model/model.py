import logging
from stable_baselines import PPO2, A2C, ACER, ACKTR, DQN, HER, DDPG, TRPO, SAC, TD3

from utils import util

ALGOS = {
    'a2c': A2C,
    'acer': ACER,
    'acktr': ACKTR,
    'dqn': DQN,
    'ddpg': DDPG,
    'her': HER,
    'sac': SAC,
    'ppo2': PPO2,
    'trpo': TRPO,
    'td3': TD3
}

def create_model(config, env, seed):
    logging.info("Creating model.")
    name =  config.pop('name')
    tlog = config.pop('tensorboard_log', None)
    verbose = config.pop('verbose', 0)
    policy_config = config.pop('policy')

    tlog_location = _get_tensorflow_log_location(tlog)

    if 'trained_agent' in config: # Continue training
        logging.info("Loading pretrained agent.")

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

