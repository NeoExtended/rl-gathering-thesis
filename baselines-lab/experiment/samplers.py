from abc import ABC, abstractmethod
from copy import deepcopy


class Sampler(ABC):
    def __init__(self, config):
        self.config = config
        self.alg_parameters = {}
        self.env_parameters = {}
        self.alg_sample = None
        self.env_sample = None

        if config.get('search', None):
            search_config = config['search']
            if search_config.get('algorithm', None):
                for sample_name in search_config['algorithm']:
                    self.alg_parameters[sample_name] = self._parse_config(search_config['algorithm'][sample_name])

            if search_config.get('env', None):
                for sample_name in search_config['env']:
                    self.env_parameters[sample_name] = self._parse_config(search_config['env'][sample_name])

    def __call__(self, *args, **kwargs):
        return self.sample(**kwargs)

    def sample(self, trial):
        alg_sample = self.sample_parameters(trial, self.alg_parameters, 'alg_')
        env_sample = self.sample_parameters(trial, self.env_parameters, 'env_')
        return self.update_config(alg_sample, env_sample)

    def sample_parameters(self, trial, parameters, prefix=""):
        sample = {}
        for name, (method, data) in parameters.items():
            p_name = prefix + name
            if method == 'categorial':
                sample[name] = trial.suggest_categorical(p_name, data)
            elif method == 'loguniform':
                low, high = data
                sample[name] = trial.suggest_loguniform(p_name, low, high)
            elif method == 'int':
                low, high = data
                sample[name] = trial.suggest_int(p_name, low, high)
            elif method == 'uniform':
                low, high = data
                sample[name] = trial.suggest_uniform(p_name, low, high)
            elif method == 'discrete_uniform':
                low, high, q = data
                sample[name] = trial.suggest_discrete_uniform(p_name, low, high, q)
        return sample

    def update_config(self, alg_sample, env_sample):
        alg_sample, env_sample = self.transform_samples(alg_sample, env_sample)
        sampled_config = deepcopy(self.config)

        alg_sample.update(self.config['algorithm'])
        sampled_config['algorithm'].update(alg_sample)

        env_sample.update(self.config['env'])
        sampled_config['env'].update(env_sample)
        self.alg_sample, self.env_sample = alg_sample, env_sample
        return sampled_config

    @property
    def last_sample(self):
        return self.alg_sample, self.env_sample

    @abstractmethod
    def transform_samples(self, alg_sample, env_sample):
        pass

    def _parse_config(self, parameter_config):
        method = parameter_config.get('method')
        if method == 'categorial':
            data = parameter_config.get('choices')
        elif method == 'loguniform':
            data = (float(parameter_config.get('low')), float(parameter_config.get('high')))
        elif method == 'int':
            data = (int(parameter_config.get('low')), int(parameter_config.get('high')))
        elif method == 'uniform':
            data = (float(parameter_config.get('low')), float(parameter_config.get('high')))
        elif method == 'discrete_uniform':
            data = (float(parameter_config.get('low')), float(parameter_config.get('high')), float(parameter_config.get('q')))
        else:
            raise NotImplementedError("Unknown sampling method {}".format(method))

        return method, data

    @staticmethod
    def create_sampler(config):
        alg_name = config['algorithm']['name']
        if alg_name == 'ppo2':
            return PPO2Sampler(config)
        else:
            raise NotImplementedError("There is currently no parameter sampler available for {}".format(alg_name))


class PPO2Sampler(Sampler):
    def __init__(self, config):
        super().__init__(config)
        parameters = {'batch_size': ('categorial', [32, 64, 128, 256]),
                      'n_steps': ('categorial', [16, 32, 64, 128, 256, 512, 1024, 2048]),
                      'gamma': ('categorial', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]),
                      'learning_rate': ('loguniform', (0.5e-5, 0.2)),
                      'ent_coef': ('loguniform', (1e-8, 0.1)),
                      'cliprange': ('categorial', [0.1, 0.2, 0.3, 0.4]),
                      'cliprange_vf': ('categorial', [-1, None]),
                      'noptepochs': ('categorial', [1, 5, 10, 20, 30, 50]),
                      'lam': ('categorial', [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])}
        parameters.update(self.alg_parameters)
        self.alg_parameters.update(parameters)

    def transform_samples(self, alg_sample, env_sample):
        batch_size = alg_sample.pop('batch_size')
        if alg_sample['n_steps'] < batch_size:
            alg_sample['nminibatches'] = 1
        else:
            alg_sample['nminibatches'] = int(alg_sample['n_steps'] / batch_size)

        return alg_sample, env_sample