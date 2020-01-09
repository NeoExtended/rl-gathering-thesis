import logging
from stable_baselines import PPO2, A2C, ACER, ACKTR, DQN, HER, DDPG, TRPO, SAC, TD3

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

def create_model(config, env):
    logging.info("Creating model.")
    name =  config.pop('name')
    tlog = config.pop('tensorboard_log', None)
    verbose = config.pop('verbose', 0)
    policy_config = config.pop('policy')

    if 'trained_agent' in config: # Continue training
        logging.info("Loading pretrained agent.")

        return ALGOS[name].load(
            config['trained_agent'],
            env=env,
            tensorboard_log=tlog,
            verbose=verbose,
            **config)

    else:
        logging.info("Creating new agent.")
        policy_name = policy_config.pop('name')

        return ALGOS[name](
            policy=policy_name,
            policy_kwargs=policy_config,
            env=env,
            tensorboard_log=tlog,
            verbose=verbose,
            **config)