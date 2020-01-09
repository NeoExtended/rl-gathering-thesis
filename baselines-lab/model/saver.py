import logging
import os
import copy
from stable_baselines.common.evaluation import evaluate_policy

from utils import util
from env.environment import create_environment


class ModelSaver:
    def __init__(self, model_dir, save_interval=250000, n_keep=5, keep_best=True, config=None):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.save_interval = save_interval
        self.n_keep = n_keep
        self.keep_best = keep_best

        if keep_best:
            assert config, "You must provide an environment configuration to evaluate the model!"

        env_desc = copy.deepcopy(config['env'])
        env_desc['n_envs'] = 1
        self.env = create_environment(env_desc, config['algorithm']['name'], config['meta']['seed'])

        self.last_models = []
        self.all_models = set()
        self.best = None
        self.best_score = 0
        self.last_save = 0
        self.update_counter = 0

    def step(self, locals_, globals_):
        model = locals_['self']
        self.update_counter += model.num_timesteps

        if self.update_counter >= self.last_save + self.save_interval:
            self.last_save = self.update_counter
            self.save(model)

    def save(self, model):
        if self.n_keep > 0:
            self._save_model(model)
        if self.keep_best:
            self._save_best_model(model)

    def _save_model(self, model):
        logging.debug("Saving last model at timestep {}".format(str(self.update_counter)))
        timestamp = util.get_timestamp()
        path = os.path.join(self.model_dir, "model_{}_{}.zip".format(self.update_counter, timestamp))
        model.save(path)
        self.last_models.append(path)
        self.all_models.add(path)

        if len(self.last_models) > self.n_keep:
            rem_model = self.last_models[0]
            del self.last_models[0]
            os.remove(rem_model)
            self.all_models.remove(rem_model)

    def _save_best_model(self, model):
        logging.debug("Evaluating model.")
        reward, steps = evaluate_policy(model, self.env, n_eval_episodes=10)

        if reward > self.best_score:
            logging.info("Found new best model with a mean reward of {}".format(str(reward)))
            self.best_score = reward
            if self.best:
                os.remove(self.best)

            timestamp = util.get_timestamp()
            path = os.path.join(self.model_dir, "model_{}_{}".format(self.update_counter, timestamp))
            self.best = path + "_best.zip"
            model.save(self.best)