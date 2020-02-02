import logging
import optuna
from copy import deepcopy
import numpy as np
from optuna.pruners import SuccessiveHalvingPruner, MedianPruner, NopPruner
from optuna.samplers import RandomSampler, TPESampler
from stable_baselines.common.vec_env import VecNormalize

from model.model import create_model
from env.environment import create_environment
from env.wrappers import VecEvaluationWrapper, EvaluationWrapper
from utils import unwrap_env, unwrap_vec_env
from experiment.runner import Runner
from experiment.logger import TensorboardLogger

class HyperparameterOptimizer:
    def __init__(self, config, log_dir):
        # TODO: Investigate wrong tensorboard outputs
        # TODO: Do not create a new env for each run - just reset - what about normalization?
        # TODO: Extend reporting to also report parameters with similar results
        # TODO: Automated parameter/result saving
        # TODO: Code cleanup

        search_config = config['search']
        self.config = config

        # test during 5 episodes
        self.n_test_episodes = search_config.get('n_test_episodes', 10)
        # evaluate every 20th of the maximum budget per iteration
        self.n_evaluations = 20
        self.n_timesteps = search_config.get('n_timesteps', 10000)
        self.evaluation_interval = int(self.n_timesteps / self.n_evaluations)
        self.n_trials = search_config.get('n_trials', 10)
        self.n_jobs = search_config.get('n_jobs', 1)
        self.seed = config['meta']['seed']
        self.sampler_method = search_config.get('sampler', 'random')
        self.pruner_method = search_config.get('pruner', 'halving')
        self.eval_method = search_config.get('eval_method', 'normal')
        self.deterministic_evaluation = search_config.get('deterministic', False)
        self.test_env = None
        self.log_dir = log_dir
        self.logger = TensorboardLogger(config['env']['n_envs'])
        self.callbacks = [self.evaluation_callback, self.logger.step]

    def optimize(self):
        sampler = self._make_sampler()
        pruner = self._make_pruner()
        logging.info("Sampler: {} - Pruner: {}".format(self.sampler_method, self.pruner_method))

        study = optuna.create_study(sampler=sampler, pruner=pruner)
        objective = self.create_objective_function()

        try:
            study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
        except KeyboardInterrupt:
            pass

        logging.info('Number of finished trials: {}'.format(len(study.trials)))
        logging.info('Best trial:')
        trial = study.best_trial

        logging.info('Value: {}'.format(trial.value))
        logging.info('Params: ')
        for key, value in trial.params.items():
            logging.info('    {}: {}'.format(key, value))

        return study.trials_dataframe()

    def _make_pruner(self):
        # n_warmup_steps: Disable pruner until the trial reaches the given number of step.
        if self.pruner_method == 'halving':
            pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
        elif self.pruner_method == 'median':
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=self.n_evaluations // 3)
        elif self.pruner_method == 'none':
            # Do not prune
            pruner = NopPruner()
        else:
            raise ValueError('Unknown pruner: {}'.format(self.pruner_method))
        return pruner

    def _make_sampler(self):
        if self.sampler_method == 'random':
            sampler = RandomSampler(seed=self.seed)
        elif self.sampler_method == 'tpe':
            sampler = TPESampler(n_startup_trials=5, seed=self.seed)
        # elif sampler_method == 'skopt':
        #     # cf https://scikit-optimize.github.io/#skopt.Optimizer
        #     # GP: gaussian process
        #     # Gradient boosted regression: GBRT
        #     sampler = SkoptSampler(skopt_kwargs={'base_estimator': "GP", 'acq_func': 'gp_hedge'})
        else:
            raise ValueError('Unknown sampler: {}'.format(self.sampler_method))
        return sampler

    def evaluation_callback(self, locals_, globals_):
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

    def create_objective_function(self):
        algo_name = self.config['algorithm']['name']
        sampler = SAMPLER[algo_name]

        def objective(trial):
            trial_config = deepcopy(self.config)
            sample = sampler(trial)
            trial_config['algorithm'].update(sample)
            logging.info("Sampled new configuration: {}".format(sample))

            integrated_evaluation = True if self.eval_method == "fast" else False

            train_env = create_environment(trial_config['env'], algo_name, trial_config['meta']['seed'], evaluation=integrated_evaluation, log_dir=self.log_dir)
            model = create_model(trial_config['algorithm'], train_env, trial_config['meta']['seed'])

            if not integrated_evaluation:
                if self.eval_method == "slow":
                    trial_config['env']['num_envs'] = 1

                self.test_env = create_environment(trial_config['env'], algo_name, trial_config['meta']['seed'], evaluation=True)
            else:
                self.test_env = unwrap_env(train_env, VecEvaluationWrapper, EvaluationWrapper)

            model.trial = trial

            try:
                logging.debug("Training model...")
                model.learn(trial_config['search']['n_timesteps'], callback=self.callback_step)
                # Free memory
                model.env.close()
                self.test_env.close()
            except AssertionError:
                # Sometimes, random hyperparams can generate NaN
                # Free memory
                logging.debug("Something went wrong - stopping trial.")
                model.env.close()
                self.test_env.close()
                raise
            is_pruned = False
            cost = np.inf
            if hasattr(model, 'is_pruned'):
                is_pruned = model.is_pruned
                cost = -1 * model.last_mean_test_reward
            del model.env, self.test_env
            del model

            if is_pruned:
                logging.info("Pruned trial.")
                raise optuna.exceptions.TrialPruned()

            return cost
        return objective

    def callback_step(self, locals_, globals_):
        for cb in self.callbacks:
            cb(locals_, globals_)


def sample_ppo2_params(trial):
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    n_steps = trial.suggest_categorical('n_steps', [16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical('gamma', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1)
    ent_coef = trial.suggest_loguniform('ent_coef', 0.00000001, 0.1)
    cliprange = trial.suggest_categorical('cliprange', [0.1, 0.2, 0.3, 0.4])
    noptepochs = trial.suggest_categorical('noptepochs', [1, 5, 10, 20, 30, 50])
    lam = trial.suggest_categorical('lamdba', [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])

    if n_steps < batch_size:
        nminibatches = 1
    else:
        nminibatches = int(n_steps / batch_size)

    return {
        'n_steps': n_steps,
        'nminibatches': nminibatches,
        'gamma': gamma,
        'learning_rate': learning_rate,
        'ent_coef': ent_coef,
        'cliprange': cliprange,
        'noptepochs': noptepochs,
        'lam': lam
    }

def sample_dqn_params(trial):
    pass

SAMPLER = {
    'ppo2' : sample_ppo2_params,
    'dqn' : sample_ppo2_params
}