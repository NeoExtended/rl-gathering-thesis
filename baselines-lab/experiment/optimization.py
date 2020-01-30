import optuna
from copy import deepcopy
import numpy as np

from model.model import create_model
from env.environment import create_environment


def evaluation_callback(self, locals_, globals_):
    self_ = locals_['self']
    trial = self_.trial

    # Initialize variables
    if not hasattr(self_, 'is_pruned'):
        self_.is_pruned = False
        self_.last_mean_test_reward = -np.inf
        self_.last_time_evaluated = 0
        self_.eval_idx = 0

    if (self_.num_timesteps - self_.last_time_evaluated) < self.evalation_interval:
        return True

    self_.last_time_evaluated = self_.num_timesteps

    if self.eval_method == "fast":
        mean_reward = self._evaluate_fast()
    elif self.eval_method == "normal":
        mean_reward = self._evaluate_normal()
    elif self.eval_method == "slow":
        mean_reward = self._evaluate_slow()
    else:
        raise NotImplementedError("Unknown evaluation method '{}'".format(self.eval_method))

    self_.last_mean_test_reward = mean_reward
    self_.eval_idx += 1


def _evaluate_fast(self):
    pass


def _evaluate_normal(self):
    pass


def _evaluate_slow(self):
    pass


def create_objective_function(config):
    algo_name = config['algorithm']['name']
    sampler = SAMPLER[algo_name]

    def objective(trial):
        trial_config = deepcopy(config)
        config['algorithm'].update(sampler(trial))

        train_env = create_environment(trial_config['env'], algo_name, trial_config['meta']['seed'])
        model = create_model(trial_config['algorithm'], train_env, trial_config['meta']['seed'])

        model.trial = trial

        try:
            model.learn(trial_config['search']['n_timesteps'], callback=evaluation_callback)
            # Free memory
            model.env.close()
            model.test_env.close()
        except AssertionError:
            # Sometimes, random hyperparams can generate NaN
            # Free memory
            model.env.close()
            model.test_env.close()
            raise
        is_pruned = False
        cost = np.inf
        if hasattr(model, 'is_pruned'):
            is_pruned = model.is_pruned
            cost = -1 * model.last_mean_test_reward
        del model.env, model.test_env
        del model

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        return cost




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