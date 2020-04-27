import tensorflow as tf
from stable_baselines.common import tf_layers
from stable_baselines.common.policies import ActorCriticPolicy

from utils.tf_utils import build_cnn


class SimpleMazeCnnPolicy(ActorCriticPolicy):
    """
    Simple CNN policy with Leaky RELU activations.
    """
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(SimpleMazeCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.leaky_relu
            extracted_features = build_cnn(self.processed_obs, **kwargs)
            pi_latent = vf_latent = activ(tf_layers.linear(extracted_features, "fc_1", 512))

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