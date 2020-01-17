import logging
import os
import copy
from stable_baselines.common.evaluation import evaluate_policy

from utils import util
from env.environment import create_environment


class ModelSaver:
    """
    Class to manage model savepoints.
    :param model_dir: (str) Target directory for all the savepoints
    :param save_interval: (int) Interval at which models will be saved. Note that the real interval may depend on the
        frequency of calls to the step() function.
    :param n_keep: Number of models to keep. when saving the newest model n+1 the oldest will be deleted automatically.
    :param keep_best: Whether or not to also save the best model. The best model is determined by running a test each
        time a new model is saved. This may take some time.
    :param config: Current lab configuration. Needed to create an evaluation environment if keep_best=True
    :param env: If the lab environment uses a running average normalization like VecNormalize, the running averages of
        the given env will be saved along with the model.
    """
    def __init__(self, model_dir, save_interval=250000, n_keep=5, keep_best=True, config=None, env=None):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.save_interval = save_interval
        self.n_keep = n_keep
        self.keep_best = keep_best

        if keep_best or env:
            assert config, "You must provide an environment configuration to evaluate the model!"

        self.last_models = []
        self.all_models = set()
        self.best = None
        self.best_score = float('-inf')
        self.last_save = 0
        self.update_counter = 0
        self.env = None

        if config:
            env_desc = copy.deepcopy(config['env'])
            env_desc['n_envs'] = 1
            normalization = env_desc.get('normalize', False)

            if normalization:
                self.env = env
                # Remove unnecessary keys
                normalization.pop('precompute', None)
                normalization.pop('samples', None)
            self.eval_env = create_environment(env_desc, config['algorithm']['name'], config['meta']['seed'])

    def step(self, locals_, globals_):
        """
        Step function that can be used as a callback in stable-baselines models. Saves models if necessary.
        :param locals_: (dict) The callers local variables at call time.
        :param globals_: (dict) The callers global variables at call time
        """
        model = locals_['self']
        self.update_counter = model.num_timesteps

        if self.update_counter >= self.last_save + self.save_interval:
            self.last_save = self.update_counter
            self.save(model)

    def save(self, model):
        """
        Explicitly saves the given model.
        :param model: stable-baselines model.
        """
        if self.n_keep > 0:
            self._save_model(model)
        if self.keep_best:
            self._save_best_model(model)

    def _save_model(self, model):
        logging.info("Saving last model at timestep {}".format(str(self.update_counter)))
        savepoint = (self.update_counter, util.get_timestamp())
        self._create_savepoint(savepoint, model)

        self.last_models.append(savepoint)
        self.all_models.add(savepoint)

        if len(self.last_models) > self.n_keep:
            old_savepoint = self.last_models[0]
            del self.last_models[0]
            self._remove_savepoint(old_savepoint)
            self.all_models.remove(old_savepoint)

    def _get_model_path(self, savepoint, postfix=None):
        if postfix:
            return os.path.join(self.model_dir, "model_{}_{}_{}.zip".format(savepoint[0], savepoint[1], postfix))
        else:
            return os.path.join(self.model_dir, "model_{}_{}.zip".format(savepoint[0], savepoint[1]))

    def _get_vecnorm_path(self, savepoint, postfix=None):
        if postfix:
            return os.path.join(self.model_dir, "normalization_{}_{}_{}.pkl".format(savepoint[0], savepoint[1], postfix))
        else:
            return os.path.join(self.model_dir, "normalization_{}_{}.pkl".format(savepoint[0], savepoint[1]))

    def _save_best_model(self, model):
        logging.debug("Evaluating model.")
        if self.env: # Update normalization running means if necessary
            self.eval_env.obs_rms = self.env.obs_rms
            self.eval_env.ret_rms = self.env.ret_rms

        reward, steps = evaluate_policy(model, self.eval_env, n_eval_episodes=10)
        logging.debug("Evaluation result: Avg reward: {:.4f}, Avg Episode Length: {:.2f}".format(reward, steps/10))

        if reward > self.best_score:
            logging.info("Found new best model with a mean reward of {:4f}".format(str(reward)))
            self.best_score = reward
            if self.best:
                self._remove_savepoint(self.best, postfix="best")

            self.best = (self.update_counter, util.get_timestamp())
            self._create_savepoint(self.best, model, postfix="best")

    def _remove_savepoint(self, savepoint, postfix=None):
        model_path = self._get_model_path(savepoint, postfix)
        os.remove(model_path)
        if self.env:
            env_path = self._get_vecnorm_path(savepoint, postfix)
            os.remove(env_path)

    def _create_savepoint(self, savepoint, model, postfix=None):
        model_path = self._get_model_path(savepoint, postfix)
        model.save(model_path)
        if self.env:
            env_path = self._get_vecnorm_path(savepoint, postfix)
            self.env.save(env_path)