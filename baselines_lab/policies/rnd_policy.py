import numpy as np
import tensorflow as tf
from stable_baselines.common import tf_layers
from stable_baselines.common.policies import ActorCriticPolicy


def build_cnn(scaled_images, activ=tf.nn.leaky_relu, **kwargs):
    """
    CNN from Nature paper. #TODO: Put this into utils and combine with RND cnn builder

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    layer_1 = activ(tf_layers.conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(tf_layers.conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(tf_layers.conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = tf_layers.conv_to_fc(layer_3)
    return layer_3


class RndPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(RndPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu

            #extracted_features1 = nature_cnn(self.processed_obs, **kwargs)
            #extracted_features1 = tf.layers.flatten(extracted_features1)
            extracted_features = build_cnn(self.processed_obs, **kwargs)

            shared_layer = activ(tf_layers.linear(extracted_features, "fc_shared_1", 256))
            shared_layer = activ(tf_layers.linear(shared_layer, "fc_shared_2", 448))

            pi_latent = activ(tf_layers.linear(shared_layer, "fc_pi_1", 448))

            vf_latent = activ(tf_layers.linear(shared_layer, "fc_vf_1", 448))
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