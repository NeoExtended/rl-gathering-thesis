from datetime import datetime
import logging
import os
import copy
import tensorflow as tf
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import VecNormalize

from utils import util
from env.environment import create_environment


class CheckpointManager:
    """
    Class to manage model checkpoints.
    :param model_dir: (str) Target directory for all the checkpoints.
    :param save_interval: (int) Interval at which models will be saved. Note that the real interval may depend on the
        frequency of calls to the step() function.
    :param n_keep: Number of models to keep. when saving the newest model n+1 the oldest will be deleted automatically.
    :param n_eval_episodes: Number of episodes used for model evaluation.
    :param keep_best: Whether or not to also save the best model. The best model is determined by running a test each
        time a new model is saved. This may take some time.
    :param config: Current lab configuration. Needed to create an evaluation environment if keep_best=True
    :param env: If the lab environment uses a running average normalization like VecNormalize, the running averages of
        the given env will be saved along with the model.
    :param tb_log: Set to true if the evaluation results should be logged. (Only works with keep_best=True)
    """
    def __init__(self, model_dir, save_interval=250000, n_keep=5, keep_best=True, n_eval_episodes=16, config=None,
                 env=None, tb_log=False):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.save_interval = save_interval
        self.n_keep = n_keep
        self.keep_best = keep_best
        self.n_eval_episodes = n_eval_episodes

        if keep_best or env:
            assert config, "You must provide an environment configuration to evaluate the model!"

        self.last_models = []
        self.all_models = set()
        self.best = None
        self.best_score = float('-inf')
        self.last_save = 0
        self.update_counter = 0
        self.env = None
        self.tb_log = tb_log
        self.writer = None

        if config:
            env_desc = copy.deepcopy(config['env'])
            env_desc['n_envs'] = 1
            normalization = env_desc.get('normalize', False)

            if normalization:
                self.env = util.unwrap_vec_env(env, VecNormalize)
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

        if not self.writer:
            self.writer = locals_['writer']

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
            reward, steps = self._save_best_model(model)
            self._log(reward, steps)

    def _save_model(self, model):
        logging.info("Saving last model at timestep {}".format(str(self.update_counter)))
        savepoint = (self.update_counter, util.get_timestamp())
        self._create_checkpoint(savepoint, model)

        self.last_models.append(savepoint)
        self.all_models.add(savepoint)

        if len(self.last_models) > self.n_keep:
            old_savepoint = self.last_models[0]
            del self.last_models[0]
            self._remove_checkpoint(old_savepoint)
            self.all_models.remove(old_savepoint)

    def _get_model_path(self, checkpoint, suffix=""):
        return os.path.join(self.model_dir,
                            self._build_filename(checkpoint, "model", suffix=suffix, extension="zip"))

    def _get_vecnorm_path(self, checkpoint, suffix=""):
        return os.path.join(self.model_dir,
                            self._build_filename(checkpoint, "normalization", suffix=suffix, extension="pkl"))

    @staticmethod
    def _build_filename(checkpoint, prefix, suffix="", extension="zip"):
        if len(suffix) > 0:
            return "{}_{}_{}_{}.{}".format(prefix, checkpoint[0], checkpoint[1], suffix, extension)
        else:
            return "{}_{}_{}.{}".format(prefix, checkpoint[0], checkpoint[1], extension)

    def _save_best_model(self, model):
        logging.debug("Evaluating model.")
        if self.env: # Update normalization running means if necessary
            norm_wrapper = util.unwrap_vec_env(self.eval_env, VecNormalize)
            norm_wrapper.obs_rms = self.env.obs_rms
            norm_wrapper.ret_rms = self.env.ret_rms

        reward, steps = evaluate_policy(model, self.eval_env, n_eval_episodes=self.n_eval_episodes)
        logging.debug("Evaluation result: Avg reward: {:.4f}, Avg Episode Length: {:.2f}".format(reward, steps/self.n_eval_episodes))

        if reward > self.best_score:
            logging.info("Found new best model with a mean reward of {:.4f}".format(reward))
            self.best_score = reward
            if self.best:
                self._remove_checkpoint(self.best, postfix="best")

            self.best = (self.update_counter, util.get_timestamp())
            self._create_checkpoint(self.best, model, postfix="best")

        return reward, steps

    def _log(self, reward, steps):
        if not self.tb_log:
            return

        length_summary = tf.Summary(value=[
            tf.Summary.Value(
                tag='episode_length/eval_ep_length_mean',
                simple_value=steps/self.n_eval_episodes)
        ])
        self.writer.add_summary(length_summary, self.update_counter)


        reward_summary = tf.Summary(value=[
            tf.Summary.Value(
                tag='reward/eval_ep_reward_mean',
                simple_value=reward)
        ])
        self.writer.add_summary(reward_summary, self.update_counter)

    def _remove_checkpoint(self, savepoint, postfix=""):
        model_path = self._get_model_path(savepoint, postfix)
        os.remove(model_path)
        if self.env:
            env_path = self._get_vecnorm_path(savepoint, postfix)
            os.remove(env_path)

    def _create_checkpoint(self, savepoint, model, postfix=""):
        model_path = self._get_model_path(savepoint, postfix)
        model.save(model_path)
        if self.env:
            env_path = self._get_vecnorm_path(savepoint, postfix)
            self.env.save(env_path)

    @classmethod
    def get_checkpoint(cls, path, type="best", normalization=False):
        """
        Returns paths to the files of the latest checkpoint saved in a given log directory.
        :param path: (str) Path to a log directory (should contain subdirectories for each run).
        :param type: (str) Type of the checkpoint ("last" or "best").
        :param normalization: (bool) Weather or not to load normalization parameters.
        :return: (str, None) or (str, str) depending on the normalization parameter, containing paths to the
            latest checkpoint files.
        """
        sp_path = os.path.join(path, "checkpoints")
        assert os.path.exists(sp_path), "No checkpoints directory found in {}".format(path)

        if type == "best":
            model_suffix = "best"
        else:
            model_suffix = ""

        checkpoint = cls._get_latest_checkpoint(sp_path, prefix="model", suffix=model_suffix)
        model_name = cls._build_filename(checkpoint, "model", suffix=model_suffix, extension="zip")
        model_path = os.path.join(sp_path, model_name)
        assert os.path.exists(model_path), "Could not find model checkpoint {} in {}".format(model_name, sp_path)

        if normalization:
            norm_name = cls._build_filename(checkpoint, "normalization", suffix=model_suffix, extension="pkl")
            norm_path = os.path.join(sp_path, norm_name)
            assert os.path.exists(norm_path), \
                "Could not find normalization parameter checkpoint {} in {}".format(norm_name, sp_path)
            return model_path, norm_path
        else:
            return model_path, None

    @staticmethod
    def get_latest_run(path):
        runs = os.listdir(path)
        runs.sort()
        return os.path.join(path, runs[-1])  # Return latest run

    @staticmethod
    def _get_latest_checkpoint(dir, prefix="", suffix=""):
        files = os.listdir(dir)

        latest = datetime.fromisoformat('1970-01-01')
        counter = None
        for savepoint in files:
            datestring = os.path.splitext(savepoint)[0]
            if not (datestring.startswith(prefix) and datestring.endswith(suffix)):
                continue

            if len(prefix) > 0:
                datestring = datestring[len(prefix) + 1:]
            if len(suffix) > 0:
                datestring = datestring[:-(len(suffix) + 1)]

            step, datestring = datestring.split("_", maxsplit=1)
            # If no suffix is given the datestring may contain invalid data.
            if len(datestring) > 17:
                continue

            date = datetime.strptime(datestring, util.TIMESTAMP_FORMAT)
            if date > latest:
                latest = date
                counter = step

        return (counter, latest.strftime(util.TIMESTAMP_FORMAT))