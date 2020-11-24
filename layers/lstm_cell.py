# encoding: utf-8
from tensorflow.nn.rnn_cell import BasicLSTMCell
from tensorflow.nn.rnn_cell import LSTMStateTuple
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from layers.base_layer import BaseLayer
from utils.logger import log_t

import tensorflow as tf
import numpy as np


class LSTMCell(BasicLSTMCell, BaseLayer):
    def __init__(self, weight_dict, is_training, pruning, batch_size, layer_type='L', *args, **kwargs):
        BasicLSTMCell.__init__(self, *args, **kwargs)
        BaseLayer.__init__(self)

        self.layer_type = layer_type
        self.weight_dict = weight_dict
        self.is_training = is_training
        self.weight_tensors = None
        self.pruning = pruning
        self.batch_size = batch_size

    def call(self, inputs, state):
        sigmoid = math_ops.sigmoid
        one = constant_op.constant(1, dtype=dtypes.int32)

        if self._state_is_tuple:
            c, h = state
        else:
            c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)

        gate_inputs = math_ops.matmul(array_ops.concat([inputs, h], 1), self._kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        i, j, f, o = array_ops.split(value=gate_inputs, num_or_size_splits=4, axis=one)

        forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)

        add = math_ops.add
        multiply = math_ops.multiply

        new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))), multiply(sigmoid(i), self._activation(j)))
        new_h = multiply(self._activation(new_c), sigmoid(o))

        # vib
        if self.pruning:
            std = tf.exp(self._logD * 0.5)
            dim = tf.shape(self._logD)[0]
            # eps = tf.random.normal(shape=[self.batch_size, dim])
            z_scale = tf.cond(self.is_training, lambda: tf.reshape(self._mu, shape=[1, -1]) + tf.random.normal(
                shape=[self.batch_size, dim]) * tf.reshape(std, shape=[1, -1]), lambda: (tf.reshape(self._mu, shape=[1,
                                                                                                                     -1]) + tf.zeros(
                shape=[self.batch_size, dim])) * self.get_mask())
            new_h = new_h * z_scale

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)

        return new_h, new_state

    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % str(inputs_shape))
        _check_supported_dtypes(self.dtype)
        input_depth = inputs_shape[-1]
        h_depth = np.int(self._num_units)

        name_prefix = '/'.join(self.scope_name.split('/')[1:])

        name_weights = '%s/weights' % name_prefix
        name_bias = '%s/biases' % name_prefix
        name_mu = '%s/mu' % name_prefix
        name_logD = '%s/logD' % name_prefix

        if name_weights in self.weight_dict.keys() and name_bias in self.weight_dict.keys():
            log_t('Load weights/bias for %s ...' % self.scope_name)
            self._kernel = tf.get_variable(name="weights",
                                           initializer=self.weight_dict[name_weights].astype(np.float32),
                                           trainable=True, dtype=tf.float32)
            self._bias = tf.get_variable(name="biases", initializer=self.weight_dict[name_bias].astype(np.float32),
                                         trainable=True, dtype=tf.float32)
        else:
            log_t('Create weights/bias for %s ...' % self.scope_name)
            self._kernel = self.add_variable('weights', shape=[input_depth + h_depth, 4 * np.int(self._num_units)])
            self._bias = self.add_variable('biases', shape=[4 * self._num_units],
                                           initializer=init_ops.zeros_initializer(dtype=self.dtype))

        if self.pruning:
            if name_mu in self.weight_dict.keys() and name_logD in self.weight_dict.keys():
                log_t('Load mu/logD for %s ...' % self.scope_name)
                self._mu = tf.get_variable(name='mu', initializer=self.weight_dict[name_mu].astype(np.float32),
                                           trainable=True, dtype=tf.float32)
                self._logD = tf.get_variable(name='logD', initializer=self.weight_dict[name_logD].astype(np.float32),
                                             trainable=True, dtype=tf.float32)
            else:
                log_t('Create mu/logD for %s ...' % self.scope_name)
                self._mu = tf.get_variable(name='mu', initializer=np.random.normal(loc=1, scale=0.01,
                                                                                   size=[int(self._num_units)]).astype(
                    np.float32))
                self._logD = tf.get_variable(name='logD', initializer=np.random.normal(loc=-9, scale=0.01,
                                                                                       size=[int(
                                                                                           self._num_units)]).astype(
                    np.float32))

        if self.pruning:
            self.weight_tensors = [self._kernel, self._bias, self._mu, self._logD]
        else:
            self.weight_tensors = [self._kernel, self._bias]

        self.built = True

    def get_kld(self):
        """
        return kl divergence of this layer with respect to x
        :param x: [batch_size, h, w, channel_size]
        :return: kl divergence
        """
        new_shape = tf.convert_to_tensor([1, tf.shape(self._mu)[0]])

        h_D = tf.exp(tf.reshape(self._logD, shape=new_shape))
        h_mu = tf.reshape(self._mu, shape=new_shape)

        KLD = tf.reduce_sum(tf.log(1 + tf.pow(h_mu, 2) / (h_D + 1e-8))) * self._num_units / tf.cast(
            tf.shape(h_D)[1], dtype=tf.float32)

        return KLD * 0.5 * 1

    def get_mask(self, threshold=0.01):
        logalpha = self._logD - tf.log(tf.pow(self._mu, 2) + 1e-8)
        mask = tf.cast(logalpha < threshold, dtype=tf.float32)
        return mask


def _check_supported_dtypes(dtype):
    if dtype is None:
        return
    dtype = dtypes.as_dtype(dtype)
    if not (dtype.is_floating or dtype.is_complex):
        raise ValueError("RNN cell only supports floating point inputs, but saw dtype: %s" % dtype)
