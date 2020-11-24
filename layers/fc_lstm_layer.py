# encoding: utf-8
from layers.base_layer import BaseLayer

import tensorflow as tf
import numpy as np


class FullConnectedLayer(BaseLayer):
    def __init__(self, weight_dict=None, regularizer_fc=None, is_musked=False):
        super(FullConnectedLayer, self).__init__()
        self.layer_type = 'F'
        self.build(weight_dict, regularizer_fc, is_musked)

    def build(self, weight_dict=None, regularizer_fc=None, is_musked=False):
        weights, biases = self.get_fc_param(weight_dict, regularizer_fc)
        self.weight_tensors = [weights, biases]

    def forward(self, x):
        # x: [batch_size, hidden_size]
        # outputs: [n_steps, batch_size, dim_out]
        outputs = tf.map_fn(
            lambda x_step: tf.nn.bias_add(tf.matmul(x_step, self.weight_tensors[0]), self.weight_tensors[1]), x)

        return outputs

    def get_fc_param(self, weight_dict, regularizer_fc):
        weights = tf.get_variable(name="weights",
                                  initializer=weight_dict[self.layer_name + '/weights'].astype(np.float32),
                                  regularizer=regularizer_fc, trainable=True, dtype=tf.float32)
        biases = tf.get_variable(name="biases", initializer=weight_dict[self.layer_name + '/biases'].astype(np.float32),
                                 regularizer=regularizer_fc, trainable=True, dtype=tf.float32)

        return weights, biases
