# encoding: utf-8
from layers.base_layer import BaseLayer

import tensorflow as tf


class InformationBottleneckLayer(BaseLayer):
    def __init__(self, layer_type, weight_dict, is_training, batch_size, dimension, kl_mult=2, mask_threshold=0):
        super(InformationBottleneckLayer, self).__init__()

        self.layer_type = layer_type
        self.is_training = is_training

        self.kl_mult = kl_mult
        self.mask_threshold = mask_threshold

        self.batch_size = batch_size
        self.dimension = dimension

        self.build(weight_dict)

    def build(self, weight_dict):
        mu, logD = self.get_ib_param(weight_dict=weight_dict)
        self.weight_tensors = [mu, logD]
        self.z_scale = self.reparameterize(mu, logD, self.is_training)
        self.z_scale = tf.cond(self.is_training, lambda: self.z_scale,
                               lambda: self.z_scale * self.get_mask(self.mask_threshold))
        self.kld = self.get_kld()

    def forward(self, x):
        """
        :param x: [n_step, batch_size, hidden_size]
        :return: [n_step, batch_szie, hidden_size]
        """
        return x * self.z_scale, self.kld

    def adapt_shape(self, src_shape, x_shape):
        """
        if dimension of src_shape = 2:
            new_shape = src_shape
        else：
            new_shape = [1, src_shape[0]]
        if dimension of x_shape > 2:
            new_shape += [1, 1]
        :param src_shape:
        :param x_shape:
        :return:
        """
        new_shape = tf.cond(tf.equal(tf.shape(src_shape)[0], 2), lambda: src_shape,
                            lambda: tf.convert_to_tensor([1, src_shape[0]]))

        new_shape = tf.cond(tf.greater(tf.shape(x_shape)[0], 2),
                            lambda: tf.concat([new_shape, tf.constant([1, 1])], axis=0),
                            lambda: new_shape)

        return new_shape

    def reparameterize(self, mu, logD, is_training):
        # std dev
        std = tf.exp(logD * 0.5)

        # num of in_channels for conv
        # num of dim of fc
        dim = tf.shape(logD)[0]

        # the random epsilon
        eps = tf.random.normal(shape=[self.batch_size, dim])
        return tf.cond(is_training, lambda: tf.reshape(mu, shape=[1, -1]) + eps * tf.reshape(std, shape=[1, -1]),
                       lambda: tf.reshape(mu, shape=[1, -1]) + tf.zeros(shape=[self.batch_size, dim]))

    def get_mask(self, threshold=0):
        logalpha = self.weight_tensors[1] - tf.log(tf.pow(self.weight_tensors[0], 2) + 1e-8)
        mask = tf.cast(logalpha < threshold, dtype=tf.float32)
        return mask

    def get_kld(self):
        """
        return kl divergence of this layer with respect to x
        :param x: [batch_size, h, w, channel_size]
        :return: kl divergence
        """
        x_shape = [self.batch_size, self.dimension]

        mu, logD = self.weight_tensors
        new_shape = self.adapt_shape(tf.shape(mu), x_shape)

        h_D = tf.exp(tf.reshape(logD, shape=new_shape))
        h_mu = tf.reshape(mu, shape=new_shape)

        KLD = tf.reduce_sum(tf.log(1 + tf.pow(h_mu, 2) / (h_D + 1e-8))) * tf.cast(x_shape[1],
                                                                                  dtype=tf.float32) / tf.cast(
            tf.shape(h_D)[1], dtype=tf.float32)

        return KLD * 0.5 * self.kl_mult

    def get_ib_param(self, weight_dict):
        mu = tf.get_variable(name='mu', initializer=weight_dict[self.layer_name + '/mu'],
                             trainable=True)
        logD = tf.get_variable(name='logD', initializer=weight_dict[self.layer_name + '/logD'],
                               trainable=True)

        return mu, logD
