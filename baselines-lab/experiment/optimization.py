import logging
import optuna
from copy import deepcopy
import numpy as np
from optuna.pruners import SuccessiveHalvingPruner, MedianPruner, NopPruner
from optuna.samplers import RandomSampler, TPESampler
from stable_baselines.common.vec_env import VecNormalize, VecEnv

from model.model import create_model
from env.environment import create_environment
from env.wrappers import VecEvaluationWrapper, EvaluationWrapper
from utils import unwrap_env, unwrap_vec_env
from experiment import Runner, TensorboardLogger, Sampler


class HyperparameterOptimizer:
    """
    Class for automated hyperparameter optimization with optuna.
    :param config: (dict) Lab config.
    :param log_dir: (str) Global log directory.
    """
    def __init__(self, config, log_dir):
        search_config = config['search']
        self.config = config

        # Number of test episodes per evaluation
        self.n_test_episodes = search_config.get('n_test_episodes', 10)
        # Number of evaluations per trial
        self.n_evaluations = search_config.get('n_evaluations', 15)
        # Timesteps per trial
        self.n_timesteps = search_config.get('n_timesteps', 10000)
        self.evaluation_interval = int(self.n_timesteps / self.n_evaluations)
        self.n_trials = search_config.get('n_trials', 10)
        self.n_jobs = search_config.get('n_jobs', 1)
        self.seed = config['meta']['seed']
        self.sampler_method = search_config.get('sampler', 'random')
        self.pruner_method = search_config.get('pruner', 'median')
        self.eval_method = search_config.get('eval_method', 'normal')
        self.deterministic_evaluation = search_config.get('deterministic', False)
        self.train_env = None
        self.test_env = None
        self.log_dir = log_dir
        self.logger = TensorboardLogger()
        self.callbacks = [self._evaluation_callback, self.logger.step]
        self.integrated_evaluation = True if self.eval_method == "fast" else False

        self._initialize_test_env()

    def optimize(self):
        """
        Starts the optimization process. This function will return even if the program receives a keyboard interrupt.
        :return (optuna.study.Study) An optuna study object containing all information about each trial that was run.
        """
        sampler = self._make_sampler()
        pruner = self._make_pruner()
        logging.info("Starting optimization process.")
        logging.info("Sampler: {} - Pruner: {}".format(self.sampler_method, self.pruner_method))

        study = optuna.create_study(sampler=sampler, pruner=pruner)
        objective = self._create_objective_function()

        try:
            study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
        except KeyboardInterrupt:
            # Still save results after keyboard interrupt!
            pass

        return study

    def _initialize_test_env(self):
        if not self.integrated_evaluation:
            test_env_config = deepcopy(self.config)
            if self.eval_method == "slow":
                test_env_config['env']['num_envs'] = 1

            if not test_env_config['env'].get('n_envs', None):
                test_env_config['env']['n_envs'] = 8

            self.test_env = create_environment(test_env_config['env'],
                                               self.config['algorithm']['name'],
                                               test_env_config['meta']['seed'],
                                               evaluation=True)

    def _make_pruner(self):
        if isinstance(self.pruner_method, str):
            if self.pruner_method == 'halving':
                pruner = SuccessiveHalvingPruner(min_resource=self.n_timesteps // 6, reduction_factor=4,  min_early_stopping_rate=0)
            elif self.pruner_method == 'median':
                pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=self.n_timesteps // 6)
            elif self.pruner_method == 'none':
                # Do not prune
                pruner = NopPruner()
            else:
                raise ValueError('Unknown pruner: {}'.format(self.pruner_method))
        elif isinstance(self.pruner_method, dict):
            method_copy = deepcopy(self.pruner_method)
            method = method_copy.pop('method')
            if method == 'halving':
                pruner = SuccessiveHalvingPruner(**method_copy)
            elif method == 'median':
                pruner = MedianPruner(**method_copy)
            elif method == 'none':
                # Do not prune
                pruner = NopPruner()
            else:
                raise ValueError('Unknown pruner: {}'.format(self.pruner_method))
        else:
            raise ValueError("Wrong type for pruner settings!")
        return pruner

    def _make_sampler(self):
        if self.sampler_method == 'random':
            sampler = RandomSampler(seed=self.seed)
        elif self.sampler_method == 'tpe':
            sampler = TPESampler(n_startup_trials=5, seed=self.seed)
        else:
            raise ValueError('Unknown sampler: {}'.format(self.sampler_method))
        return sampler

    def _evaluation_callback(self, locals_, globals_):
        self_ = locals_['self']
        trial = self_.trial

        # Initialize variables
        if not hasattr(self_, 'is_pruned'):
            self_.is_pruned = False
            self_.last_mean_test_reward = -np.inf
            self_.last_time_evaluated = 0
            self_.eval_idx = 0

        if (self_.num_timesteps - self_.last_time_evaluated) < self.evaluation_interval:
            return True

        self_.last_time_evaluated = self_.num_timesteps
        logging.debug("Evaluating model at {} timesteps".format(self_.num_timesteps))

        if self.eval_method == "fast":
            mean_reward = self._evaluate_fast()
        elif self.eval_method == "normal":
            mean_reward = self._evaluate_normal(self_)
        elif self.eval_method == "slow":
            # Slow means only a single env is used in parallel.
            mean_reward = self._evaluate_normal(self_)
        else:
            raise NotImplementedError("Unknown evaluation method '{}'".format(self.eval_method))

        logging.info("Evaluated model at {} timesteps. Reached a mean reward of {}".format(self_.num_timesteps, mean_reward))
        self_.last_mean_test_reward = mean_reward
        self_.eval_idx += 1

        # report best or report current ?
        # report num_timesteps or elasped time ?
        trial.report(-1 * mean_reward, self_.num_timesteps) #TODO: Change to num_timesteps?
        # Prune trial if need
        if trial.should_prune(self_.num_timesteps):
            logging.debug("Pruning - aborting training...")
            self_.is_pruned = True
            return False

        return True

    def _evaluate_fast(self):
        return self.test_env.aggregator.reward_rms

    def _evaluate_normal(self, model):
        eval = unwrap_env(self.test_env, VecEvaluationWrapper, EvaluationWrapper)
        eval.reset_statistics()

        if self.config['env'].get('normalize', None):
            norm = unwrap_vec_env(self.test_env, VecNormalize)
            model_norm = unwrap_vec_env(model.env, VecNormalize)
            norm.obs_rms = model_norm.obs_rms
            norm.ret_rms = model_norm.ret_rms
            norm.training = False

        runner = Runner(self.test_env, model, render=False, deterministic=self.deterministic_evaluation, close_env=False)
        runner.run(self.n_test_episodes)
        return eval.aggregator.mean_reward

    def _get_train_env(self, config):
        if self.train_env:
            # Create new environments if normalization layer is learned.
            if config['env'].get('normalize', None):
                if not config['env']['normalize'].get('precompute', False):
                    self.train_env.close()
                    self._make_train_env(config)
            # Create new environments if num_envs changed.
            if isinstance(self.train_env, VecEnv):
                if self.train_env.unwrapped.num_envs != config['env']['n_envs']:
                    self.train_env.close()
                    self._make_train_env(config)
        else:
            self._make_train_env(config)

        return self.train_env

    def _make_train_env(self, config):
        self.train_env = create_environment(config['env'],
                                            config['algorithm']['name'],
                                            config['meta']['seed'],
                                            evaluation=self.integrated_evaluation,
                                            log_dir=self.log_dir)
        if self.integrated_evaluation:
            self.test_env = unwrap_env(self.train_env, VecEvaluationWrapper, EvaluationWrapper)

    def _create_objective_function(self):
        sampler = Sampler.create_sampler(self.config)

        def objective(trial):
            trial_config = sampler.sample(trial)
            trial_config['algorithm']['verbose'] = 0
            alg_sample, env_sample = sampler.last_sample
            logging.info("Sampled new configuration: algorithm: {} env: {}".format(alg_sample, env_sample))

            train_env = self._get_train_env(trial_config)
            model = create_model(trial_config['algorithm'], train_env, None)
            # model = create_model(trial_config['algorithm'], train_env, trial_config['meta']['seed'])
            # TODO: Reenable once stable-baselines bug is fixed

            self.logger.reset()
            model.trial = trial
            try:
                logging.debug("Training model...")
                model.learn(trial_config['search']['n_timesteps'], callback=self.callback_step)
            except:
                # Random hyperparams may be invalid
                logging.debug("Something went wrong - stopping trial.")
                raise
            is_pruned = False
            cost = np.inf
            if hasattr(model, 'is_pruned'):
                is_pruned = model.is_pruned
                cost = -1 * model.last_mean_test_reward
            del model

            if is_pruned:
                logging.info("Pruned trial.")
                raise optuna.exceptions.TrialPruned()

            return cost
        return objective

    def callback_step(self, locals_, globals_):
        ret_val = True
        for cb in self.callbacks:
            if not cb(locals_, globals_):
                ret_val = False
        return ret_val
