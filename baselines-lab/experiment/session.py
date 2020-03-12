import distutils.spawn
import logging
import os
import numpy as np
from abc import ABC, abstractmethod

from stable_baselines.common.vec_env import VecVideoRecorder
import matplotlib.pyplot as plt

from utils import util, config_util
from env.environment import create_environment
from model.model import create_model
from model.checkpoints import CheckpointManager
from experiment import TensorboardLogger, Runner, HyperparameterOptimizer, Sampler
from env.wrappers import VecGifRecorder
from env.evaluation import EvaluationWrapper, VecEvaluationWrapper


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
        ret_val = True
        for cb in self.callbacks:
            ret = cb.step(locals_, globals_)
            if isinstance(ret, bool) and not ret:
                ret_val = False
        return ret_val

    @staticmethod
    def create_session(config, args):
        if args.lab_mode == "train":
            return TrainSession(config, args)
        elif args.lab_mode == "enjoy":
            return ReplaySession(config, args)
        elif args.lab_mode == "search":
            return SearchSession(config, args)
        else:
            raise ValueError("Unknown lab mode!")

    def _create_log_dir(self):
        log_dir = self.config['meta'].get('log_dir', None)
        self.log = util.create_log_directory(log_dir)
        if self.log:
            config_util.save_config(self.config, os.path.join(self.log, "config.yml"))


class ReplaySession(Session):
    """
    Control unit for a replay session (enjoy lab mode - includes model evaluation).
    """
    def __init__(self, config, args):
        Session.__init__(self, config, args)

        if args.checkpoint_path:
            data_path = args.checkpoint_path
        else:
            data_path = os.path.split(os.path.dirname(config['algorithm']['trained_agent']))[0]

        self.env = create_environment(config=config['env'],
                                      algo_name=config['algorithm']['name'],
                                      seed=self.config['meta']['seed'],
                                      log_dir=None,
                                      video_path=data_path if args.obs_video else None,
                                      evaluation=args.evaluate)

        # TODO: Reenable agent seeding once the stable-baselines bug is fixed
        # self.agent = create_model(config['algorithm'], self.env, seed=self.config['meta']['seed'])
        self.agent = create_model(config['algorithm'], self.env, seed=None)

        obs = self.env.reset()

        self.deterministic = not args.stochastic
        if args.video:
            self._setup_video_recorder(data_path)

        if args.evaluate:
            self.num_episodes = args.evaluate
            eval_wrapper = util.unwrap_env(self.env, VecEvaluationWrapper, EvaluationWrapper)
            eval_wrapper.aggregator.path = data_path
        else:
            # Render about 4 complete episodes per env in enjoy mode without evaluation.
            self.num_episodes = self.config['env']['n_envs'] * 4

        self.runner = Runner(self.env, self.agent, deterministic=self.deterministic)

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
        self.runner.run(self.num_episodes)


class TrainSession(Session):
    """
    Control unit for the training lab mode.
    """
    def __init__(self, config, args):
        Session.__init__(self, config, args)
        self._create_log_dir()
        self.env = create_environment(config=config['env'],
                                      algo_name=config['algorithm']['name'],
                                      seed=self.config['meta']['seed'],
                                      log_dir=self.log)
        # TODO: Reenable agent seeding once the stable-baselines bug is fixed
        # self.agent = create_model(config['algorithm'], self.env, seed=self.config['meta']['seed'])
        self.agent = create_model(config['algorithm'], self.env, seed=None)

    def run(self):
        logging.info("Starting training.")
        save_interval = self.config['meta'].get('save_interval', 250000)
        n_keep = self.config['meta'].get('n_keep', 5)
        keep_best = self.config['meta'].get('keep_best', True)
        n_eval_episodes = self.config['meta'].get('n_eval_episodes', 32)

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
        self.add_callback(TensorboardLogger())

        self.agent.learn(self.config['meta']['n_timesteps'], callback=self.step)

        # Save model at the end of the learning process
        saver.save(self.agent)
        self.env.close()


class SearchSession(Session):
    """
    Control unit for the search lab mode
    """
    def __init__(self, config, args):
        Session.__init__(self, config, args)

        if config['search'].get("resume", False):
            self.log = config['search']['resume']
            util.set_log_directory(self.log)
        else:
            self._create_log_dir()

        self.optimizer = HyperparameterOptimizer(config, self.log, args.mail)
        self.plot = args.plot

    def run(self):
        study = self.optimizer.optimize()
        self._log_study_info(study)

        dataframe = study.trials_dataframe()
        dataframe.to_csv(os.path.join(self.log, "search_history.csv"))
        if self.plot:
            # Suppress find font spam
            logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
            try:
                hist = dataframe.hist()
            except:
                logging.warning("Could not plot data due to infinite values. :(")
            plt.show()

        promising = self._find_promising_trials(study)
        for i, trial in enumerate(promising):
            self._save_config(study.trials[trial[0]], "promising_trial_{}.yml".format(i))
        self._save_config(study.best_trial, "best_trial.yml")

    def _save_config(self, trial, name):
        alg_params = {}
        env_params = {}
        for key, value in trial.params.items():
            if key.startswith("env_"):
                env_params[key[4:]] = value
            elif key.startswith("alg_"):
                alg_params[key[4:]] = value
        sampled_config = Sampler.create_sampler(self.config).update_config(alg_params, env_params)
        config_util.save_config(sampled_config, os.path.join(self.log, name))

    def _find_promising_trials(self, study):
        promising = list()
        for idx, trial in enumerate(study.trials):
            promising.append([idx, trial.value])

        promising = sorted(promising, key=lambda x: x[1])
        return promising[:3]

    def _log_study_info(self, study):
        logging.info('Number of finished trials: {}'.format(len(study.trials)))
        logging.info('Best trial:')
        trial = study.best_trial

        logging.info('Value: {}'.format(trial.value))
        logging.info('Params: ')
        for key, value in trial.params.items():
            logging.info('  {}: {}'.format(key, value))
