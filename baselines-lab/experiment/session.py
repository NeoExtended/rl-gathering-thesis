import distutils.spawn
import logging
import os
import numpy as np
from abc import ABC, abstractmethod

from stable_baselines.common.vec_env import VecVideoRecorder
from utils import util, config_util
from env.environment import create_environment
from model.model import create_model
from model.checkpoints import CheckpointManager
from experiment.logger import TensorboardLogger
from env.wrappers import VecGifRecorder


class Session(ABC):
    """
    The main experiment control unit. Creates the environment and model from the lab configuration, runs the experiment
        and controls model saving.
    :param config: (dict) The lab configuration.
    :param args: (dict) Parsed additional command line arguments.
    """
    def __init__(self, config, args):
        self.config = config
        util.set_random_seed(self.config)

        log_dir = self.config['meta'].get('log_dir', None)
        if args.lab_mode == "train":
            self.log = util.create_log_directory(log_dir)
            if self.log:
                config_util.save_config(self.config, os.path.join(self.log, "config.yml"))
        else:
            self.log = None

        self.lab_mode = args.lab_mode
        self.callbacks = []

    @abstractmethod
    def run(self):
        """
        Starts the experiment.
        """
        pass

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def step(self, locals_, globals_):
        for cb in self.callbacks:
            cb.step(locals_, globals_)

    @staticmethod
    def create_session(config, args):
        if args.lab_mode == "train":
            return TrainSession(config, args)
        elif args.lab_mode == "enjoy":
            return ReplaySession(config, args)
        else:
            raise ValueError("Unknown lab mode!")


class ReplaySession(Session):
    """
    Control unit for a replay session (includes enjoy and evaluation lab modes).
    """
    def __init__(self, config, args):
        Session.__init__(self, config, args)

        if args.video or args.obs_video:
            if args.checkpoint_path:
                video_path = args.get("checkpoint_path")
            else:
                video_path = os.path.split(os.path.dirname(config['algorithm']['trained_agent']))[0]
        else:
            video_path = None

        self.env = create_environment(config=config['env'],
                                      algo_name=config['algorithm']['name'],
                                      seed=self.config['meta']['seed'],
                                      log_dir=None,
                                      video_path=video_path)

        self.agent = create_model(config['algorithm'], self.env, seed=self.config['meta']['seed'])
        self.deterministic = not args.stochastic
        if args.video:
            self._setup_video_recorder(video_path)

    def _setup_video_recorder(self, video_path):
        if distutils.spawn.find_executable("avconv") or distutils.spawn.find_executable("ffmpeg"):
            logging.info("Using installed standard video encoder.")
            self.env = VecVideoRecorder(self.env, video_path,
                                        record_video_trigger=lambda x: x == 0,
                                        video_length=10000,
                                        name_prefix=util.get_timestamp())
        else:
            logging.warning("Did not find avconf or ffmpeg - using gif as a video container replacement.")
            self.env = VecGifRecorder(self.env, video_path)

    def run(self):
        obs = self.env.reset()
        episode_counter = 0
        step_counter = 0
        num_episodes = self.config['env']['n_envs'] * 4
        while episode_counter < num_episodes:  # Render about 4 complete episodes per env
            action, _states = self.agent.predict(obs, deterministic=self.deterministic)
            obs, rewards, dones, info = self.env.step(action)
            self.env.render()
            episode_counter += np.sum(dones)
            step_counter += (self.config['env']['n_envs'])

        logging.info("Performed {} episodes with an avg length of {}".format(num_episodes, step_counter / num_episodes))
        self.env.close()


class TrainSession(Session):
    """
    Control unit for the training lab mode.
    """
    def __init__(self, config, args):
        Session.__init__(self, config, args)

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
        n_eval_episodes = self.config['meta'].get('n_eval_episodes', 16)

        saver = CheckpointManager(
            model_dir=os.path.join(self.log, "checkpoints"),
            save_interval=save_interval,
            n_keep=n_keep,
            keep_best=keep_best,
            n_eval_episodes=n_eval_episodes,
            config=self.config,
            env=self.env,
            tb_log=bool(self.log))
        self.add_callback(saver)
        self.add_callback(TensorboardLogger(n_envs=self.config['env']['n_envs']))

        self.agent.learn(self.config['meta']['n_timesteps'], callback=self.step)

        # Save model at the end of the learning process
        saver.save(self.agent)
        self.env.close()