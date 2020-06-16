import copy
import distutils.spawn
import logging
import os
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
from stable_baselines.common.vec_env import VecVideoRecorder

# Load mpi4py-dependent algorithms only if mpi is installed.
try:
    import mpi4py
except ImportError:
    mpi4py = None

if mpi4py is not None:
    from stable_baselines.gail import ExpertDataset

from baselines_lab.env import create_environment
from baselines_lab.env.wrappers import EvaluationWrapper, VecEvaluationWrapper, VecGifRecorder
from baselines_lab.experiment import Runner, HyperparameterOptimizer
from baselines_lab.experiment.samplers import Sampler
from baselines_lab.model import create_model
from baselines_lab.model.callbacks import CheckpointManager, TensorboardLogger
from baselines_lab.model.callbacks.obs_logger import ObservationLogger
from baselines_lab.utils import util, config_util
from baselines_lab.utils.tensorboard.plotter import Plotter

PLOT_TAGS = ["curiosity/ep_ext_reward_mean", "curiosity/ep_int_reward_mean", "episode_length/ep_length_mean", "episode_length/eval_ep_length_mean", "episode_reward", "reward/ep_reward_mean", "reward/eval_ep_reward_mean"]
PLOT_NAMES = ["normalized extrinsic reward", "intrinsic reward", "episode length", "eval episode length", "total episode reward", "episode reward", "eval episode reward"]


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

    @abstractmethod
    def run(self):
        """
        Starts the experiment.
        """
        pass

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
            self.config['meta']['timestamp'] = util.get_timestamp()
            config_util.save_config(self.config, os.path.join(self.log, "config.yml"))

    def _plot(self, log_dir):
        file_format = 'pdf'
        if isinstance(self.config['meta']['plot'], dict):
            file_format = self.config['meta']['plot'].get('format', file_format)
            PLOT_TAGS.extend(self.config['meta']['plot'].get('tags'))
            PLOT_NAMES.extend(self.config['meta']['plot'].get('names'))
        plotter = Plotter(file_format, [log_dir])
        plotter.make_plot(PLOT_TAGS, PLOT_NAMES)


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

        self.env = create_environment(config=config,
                                      seed=self.config['meta']['seed'],
                                      log_dir=None,
                                      video_path=data_path if args.obs_video else None,
                                      evaluation=args.evaluate)

        self.agent = create_model(config['algorithm'], self.env, seed=self.config['meta']['seed'])

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

        if args.plot:
            self._plot(config['meta']['session_dir'])

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
        self.env = None
        self.agent = None
        self.saver = None

    def _setup_session(self):
        eval_method = self.config['meta'].get('eval_method', 'normal')

        self.env = create_environment(config=self.config,
                                      seed=self.config['meta']['seed'],
                                      log_dir=util.get_log_directory(),
                                      evaluation='fast' in eval_method)

        self.agent = create_model(self.config['algorithm'], self.env, seed=self.config['meta']['seed'])

        save_interval = self.config['meta'].get('save_interval', 250000)
        n_keep = self.config['meta'].get('n_keep', 5)
        keep_best = self.config['meta'].get('keep_best', True)
        n_eval_episodes = self.config['meta'].get('n_eval_episodes', 32)

        self.saver = CheckpointManager(
            model_dir=os.path.join(util.get_log_directory(), "checkpoints"),
            save_interval=save_interval,
            eval_method=eval_method,
            n_keep=n_keep,
            keep_best=keep_best,
            n_eval_episodes=n_eval_episodes,
            config=self.config,
            env=self.env,
            tb_log=bool(self.log))

    def run(self):
        n_trials = self.config['meta'].get('n_trials', 1)

        if n_trials == 1:
            self._setup_session()
            self._run_experiment()
        else:
            for i in range(n_trials):
                trial_dir = os.path.join(self.log, "trial_{}".format(i))
                os.mkdir(trial_dir)
                util.set_log_directory(trial_dir)

                self._setup_session()
                self._run_experiment()
                del self.env
                del self.agent
                del self.saver

        if self.config['meta'].get('plot', False):
            self._plot(self.log)

    def _run_experiment(self):
        self._pretrain()
        callbacks = [self.saver, TensorboardLogger(config=self.config)]
        if self.config['meta'].get('record_images', False):
            callbacks.append(ObservationLogger())

        logging.info("Starting training.")
        self.agent.learn(self.config['meta']['n_timesteps'], callback=callbacks)

        # Save model at the end of the learning process and do some cleanup.
        self.saver.save(self.agent)
        self.env.close()
        self.saver.close()

    def _pretrain(self):
        if self.config['meta'].get('pretrain', None):
            logging.info("Starting pretraining.")
            pretrain_config = copy.deepcopy(self.config['meta']['pretrain'])
            archive_location = pretrain_config.get('expert_path')
            n_epochs = pretrain_config.pop('n_epochs', 1000)
            assert os.path.exists(archive_location), "Could not find archive with pretraining data at {}".format(archive_location)
            dataset = ExpertDataset(**pretrain_config)
            self.agent.pretrain(dataset, n_epochs=n_epochs)


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
