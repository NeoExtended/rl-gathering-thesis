import tensorflow as tf
from stable_baselines.common import tf_layers
from stable_baselines.common.policies import ActorCriticPolicy, mlp_extractor

from baselines_lab.utils.tf_utils import build_cnn, build_dynamic_cnn
import numpy as np

class SimpleMazeCnnPolicy(ActorCriticPolicy):
    """
    Simple CNN policy with Leaky RELU activations.
    """
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, scale=True, **kwargs):
        super(SimpleMazeCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=scale)

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.leaky_relu
            extracted_features = build_cnn(self.processed_obs, **kwargs)
            pi_latent = vf_latent = activ(tf_layers.linear(extracted_features, "fc_1", 512, init_scale=np.sqrt(2)))

            value_fn = tf_layers.linear(vf_latent, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._value_fn = value_fn
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


class GeneralCnnPolicy(ActorCriticPolicy):
    """
    Configurable policy with flexible CNN extractor combined with flexible MLP.
    :param extractor_arch: Architecture description for the CNN extractor. Consists out of tuples of (n_filters, filter_size, stride).
    :param mlp_arch: Mlp architecture. Contains integers denoting the number of shared fully connected layers and an optional dict with the keys 'pi' and
                    'vf' to define individual networks.
    """
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, scale=True, extractor_arch=None, mlp_arch=None,
                 extractor_act=tf.nn.leaky_relu, mlp_act=tf.nn.leaky_relu, **kwargs):
        super(GeneralCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=scale)

        if extractor_arch is None:
            extractor_arch = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]

        if mlp_arch is None:
            mlp_arch = [512]

        with tf.variable_scope("model", reuse=reuse):
            extracted_features = build_dynamic_cnn(self.processed_obs, extractor_arch, extractor_act, **kwargs)
            pi_latent, vf_latent = mlp_extractor(extracted_features, mlp_arch, mlp_act)

            value_fn = tf.layers.dense(vf_latent, 1, name='vf')

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._value_fn = value_fn
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})