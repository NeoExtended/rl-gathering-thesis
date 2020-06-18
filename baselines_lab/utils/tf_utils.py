import numpy as np
import tensorflow as tf
from stable_baselines.common import tf_layers


def build_dynamic_cnn(input, arch, activ=tf.nn.leaky_relu, **kwargs):
    current_layer = input
    for idx, layer in enumerate(arch):
        filters, filter_size, stride = layer
        current_layer = activ(tf_layers.conv(current_layer, "c{}".format(idx), n_filters=filters, filter_size=filter_size, stride=stride, init_scale=np.sqrt(2)), **kwargs)
    to_fc = tf_layers.conv_to_fc(current_layer)
    return to_fc


def build_cnn(scaled_images, activ=tf.nn.leaky_relu, **kwargs):
    """
    CNN from Nature paper.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    layer_1 = activ(tf_layers.conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(tf_layers.conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(tf_layers.conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = tf_layers.conv_to_fc(layer_3)
    return layer_3