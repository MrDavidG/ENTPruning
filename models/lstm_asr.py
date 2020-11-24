# encoding: utf-8
import sys

sys.path.append('..')
from models.base_model import BaseModel
from layers.fc_lstm_layer import FullConnectedLayer
from layers.lstm_cell import LSTMCell
from utils.configer import get_cfg, load_cfg
from utils.logger import *

import tensorflow as tf
import pandas as pd
import numpy as np

import json
import pickle
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class LstmNet(BaseModel):
    def __init__(self, config, load_train=True, load_test=True):
        super(LstmNet, self).__init__(config, load_train, load_test, data_type='seq')

        self.X_ = tf.placeholder(dtype=tf.float32, shape=[None, None, self.cfg['data'].getint('dimension')])
        # [batch_size, n_steps]
        self.Y_ = tf.placeholder(dtype=tf.int32)
        self.Y = tf.one_hot(self.Y_, depth=self.cfg['data'].getint('n_classes'))

        model_path = config['path']['path_load']

        if model_path and os.path.exists(model_path):
            log_t('Loading weight from %s' % model_path)
            # Load weight
            self.load_model(model_path)
        else:
            log_l('Initialize weight matrix')
            self.weight_dict = self.construct_initial_weights()

        if self.cfg['basic'].getfloat('entropy_factor') is not None:
            self.entropy_factor = self.cfg['basic'].getfloat('entropy_factor')

        self.build()

    def load_model(self, model_path):
        self.weight_dict = pickle.load(open(model_path, 'rb'), encoding='bytes')
        # reset
        dim_list = list()
        for i in range(self.cfg['structure'].getint('n_layers_lstm')):
            hidden_size = np.shape(self.weight_dict['rnn/multi_rnn_cell/cell_%d/lstm_cell/weights' % i])[1] / 4
            dim_list.append(hidden_size)
        self.cfg['structure']['layers'] = str([self.cfg['data'].getint('dimension')] + dim_list + [
            self.cfg['data'].getint('n_classes')])
        log_t('Loading models as %s' % self.cfg['structure']['layers'])

    def construct_initial_weights(self):
        # only create tensors for fc layers
        def bias_variable(shape):
            return (np.zeros(shape=shape, dtype=np.float32)).astype(dtype=np.float32)

        def weight_variable(shape):
            return np.random.normal(loc=0, scale=np.sqrt(1. / shape[0]), size=shape).astype(np.float32)

        dim_layers = json.loads(self.cfg['structure']['layers'])

        weight_dict = dict()

        n_lstm = self.cfg['structure'].getint('n_layers_lstm')
        n_fc = self.cfg['structure'].getint('n_layers_fc')

        for index_fc in range(n_fc):
            index_layer = index_fc + n_lstm + 1

            shape_weights = [dim_layers[index_layer - 1], dim_layers[index_layer]]

            weight_dict['fc%d/weights' % (index_layer)] = weight_variable(shape_weights)
            weight_dict['fc%d/biases' % (index_layer)] = bias_variable(dim_layers[index_layer])

        return weight_dict

    def inference(self):
        self.layers.clear()
        with tf.variable_scope(self.task_name, reuse=tf.AUTO_REUSE):
            if self.cfg['basic']['pruning_method'] == 'info_bottle':
                self.kl_total = 0.
            y = self.X_

            n_lstm = self.cfg['structure'].getint('n_layers_lstm')
            n_fc = self.cfg['structure'].getint('n_layers_fc')

            hidden_size = json.loads(self.cfg['structure']['layers'])[1:-1]

            # lstm
            flag = self.cfg['basic']['pruning_method'] == 'info_bottle'
            batch_size = self.cfg['basic'].getint('batch_size')

            cell_list = [
                LSTMCell(self.weight_dict, is_training=self.is_training, pruning=flag, batch_size=batch_size,
                         num_units=hidden_size[0]),
                LSTMCell(self.weight_dict, is_training=self.is_training, pruning=flag, batch_size=batch_size,
                         num_units=hidden_size[1]),
                LSTMCell(self.weight_dict, is_training=self.is_training, pruning=flag, batch_size=batch_size,
                         num_units=hidden_size[2]),
                LSTMCell(self.weight_dict, is_training=self.is_training, pruning=flag, batch_size=batch_size,
                         num_units=hidden_size[3])]

            lstm_cell = tf.contrib.rnn.MultiRNNCell(cell_list)

            _init_state = lstm_cell.zero_state(self.cfg['basic'].getint('batch_size'), dtype=tf.float32)
            # outputs: [step, batch_size, hidden_size]
            y, _ = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=y, initial_state=_init_state, dtype=tf.float32,
                                     time_major=True)
            y = tf.nn.relu(y)

            self.layers = [cell_list[i] for i in range(4)]

            if self.cfg['basic']['pruning_method'] == 'info_bottle':
                for cell in cell_list:
                    self.kl_total += cell.get_kld()

            # output
            with tf.variable_scope('fc%d' % (n_lstm + n_fc)):
                fc_layer = FullConnectedLayer(self.weight_dict, regularizer_fc=self.regularizer_fc,
                                              is_musked=self.is_musked)
                self.layers.append(fc_layer)

                # [n_step, batch_size, dim_output]
                y = fc_layer.forward(y)

                self.op_logits = tf.reshape(y, shape=[-1, self.cfg['data'].getint('n_classes')])

    def loss(self):
        # self.Y: [n_step*batch_size, n_classes]
        # self.op_logits: [n_step*batch_size, n_classes]
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=self.op_logits)
        l2_loss = tf.losses.get_regularization_loss()

        if self.cfg['basic']['pruning_method'] == 'info_bottle':
            self.op_loss = tf.reduce_mean(entropy,
                                          name='loss') + l2_loss + self.kl_factor * self.kl_total + self.entropy_factor * self.op_entropy
        else:
            self.op_loss = tf.reduce_mean(entropy, name='loss') + l2_loss + self.entropy_factor * self.op_entropy

    def optimize(self, lr):
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            self.opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9, use_nesterov=True)
            self.op_opt = self.opt.minimize(self.op_loss)

    def evaluate(self):
        with tf.name_scope('predict'):
            correct_preds = tf.equal(tf.argmax(self.op_logits, 1), tf.argmax(self.Y, 1))
            self.op_accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

    def entropy(self):
        with tf.name_scope('entropy'):
            reduce_mean = self.op_logits - tf.reshape(tf.reduce_max(self.op_logits, axis=1), shape=(-1, 1))
            score = tf.exp(reduce_mean) / tf.reshape(tf.reduce_sum(tf.exp(reduce_mean), axis=1), shape=(-1, 1))
            # 得到的是第一个batch内每个sample平均的结果
            self.op_entropy = - tf.reduce_mean(tf.reduce_sum(score * tf.log(score + 1e-30), axis=1))

    def train_one_epoch(self, sess, epoch):
        avg_acc = 0
        avg_loss = 0
        avg_kl_loss = 0
        avg_entropy = 0
        n_batches = 1

        time_last = time.time()

        data_loader = self.loader_train.get_data_in_batch(shuffle=True)
        try:
            while True:
                x, y = data_loader.__next__()

                if self.cfg['basic']['pruning_method'] == 'info_bottle':
                    _, loss, loss_kl, accuracy_batch, entropy_batch = sess.run(
                        [self.op_opt, self.op_loss, self.kl_total, self.op_accuracy, self.op_entropy],
                        feed_dict={self.is_training: True, self.X_: x, self.Y_: y})

                    avg_kl_loss += (loss_kl * self.kl_factor - avg_kl_loss) / n_batches
                    avg_entropy += (entropy_batch - avg_entropy) / n_batches
                else:
                    _, loss, accuracy_batch = sess.run([self.op_opt, self.op_loss, self.op_accuracy],
                                                       feed_dict={self.is_training: True, self.X_: x, self.Y_: y})

                avg_loss += (loss - avg_loss) / n_batches
                avg_acc += (accuracy_batch - avg_acc) / n_batches

                n_batches += 1

                if n_batches % 5 == 0:

                    if self.cfg['basic']['pruning_method'] == 'info_bottle':
                        str_ = 'epoch={:d}, batch={:d}/{:d}, curr_loss={:f}, kl_loss={:f}, curr_acc={:%}, entropy={:.4f}, used_time:{:.4f}s'.format(
                            epoch + 1,
                            n_batches,
                            self.loader_train.get_N() // self.loader_train.batch_size + 1,
                            avg_loss,
                            avg_kl_loss,
                            avg_acc,
                            avg_entropy,
                            time.time() - time_last)
                    else:
                        str_ = 'epoch={:d}, batch={:d}/{:d}, curr_loss={:f}, curr_acc={:%}, used_time:{:.4f}s'.format(
                            epoch + 1,
                            n_batches,
                            self.loader_train.get_N() // self.loader_train.batch_size + 1,
                            avg_loss,
                            avg_acc,
                            time.time() - time_last)

                    print('\r' + str_, end=' ')

                    time_last = time.time()

                    if n_batches % 5000 == 0 and self.cfg['basic']['pruning_method'] == 'info_bottle':
                        cr = self.get_CR(sess)
                        self.save_weight_clean(sess, save_path='%s/tr%.2d-bat%.3d-cr%.4f' % (
                            self.cfg['path']['path_save'], self.cnt_train, n_batches, cr))

        except IndexError:
            pass

        log(str_, need_print=False)

    def eval_once(self, sess, epoch):
        total_loss = 0
        total_kl_loss = 0
        total_correct_preds = 0
        total_entropy = 0
        n_batches = 0

        data_loader = self.loader_test.get_data_in_batch(shuffle=False)
        try:
            while True:
                x, y = data_loader.__next__()
                if self.cfg['basic']['pruning_method'] == 'info_bottle':
                    loss_batch, loss_kl, accuracy_batch, entropy_batch = sess.run(
                        [self.op_loss, self.kl_total, self.op_accuracy, self.op_entropy],
                        feed_dict={self.is_training: False, self.X_: x,
                                   self.Y_: y})
                    total_kl_loss += loss_kl
                else:
                    loss_batch, accuracy_batch, entropy_batch = sess.run(
                        [self.op_loss, self.op_accuracy, self.op_entropy],
                        feed_dict={self.is_training: False, self.X_: x, self.Y_: y})

                total_loss += loss_batch
                total_entropy += entropy_batch
                total_correct_preds += accuracy_batch
                n_batches += 1

        except IndexError:
            pass

        acc = np.around(total_correct_preds / n_batches, decimals=4)
        if self.cfg['basic']['pruning_method'] == 'info_bottle':
            log('\nEpoch:{:d}, val_acc={:%}, val_loss={:f}, kl_loss={:f}, entropy={:f}'.format(epoch + 1,
                                                                                               acc,
                                                                                               total_loss / n_batches,
                                                                                               total_kl_loss * self.kl_factor / n_batches,
                                                                                               total_entropy / n_batches))
        else:
            log('\nEpoch:{:d}, val_acc={:%}, val_loss={:f}, entropy={:f}'.format(epoch + 1, acc, total_loss / n_batches,
                                                                                 total_entropy / n_batches))
        return acc, entropy_batch / n_batches

    def train(self, sess, n_epochs, lr, save=False):
        self.optimize(lr)

        name = None

        sess.run(tf.variables_initializer(self.opt.variables()))
        for epoch in range(n_epochs):
            self.train_one_epoch(sess, epoch)
            acc, ent = self.eval_once(sess, epoch)

            if self.cfg['basic']['pruning_method'] == 'info_bottle':
                cr = self.get_CR(sess)

            if self.is_evaluated(epoch + 1) or save:
                if self.cfg['basic']['pruning_method'] != 'info_bottle':
                    name = '%s/tr%.2d-epo%.3d-acc%.4f-ent%.4f' % (
                        self.cfg['path']['path_save'], self.cnt_train, epoch + 1, acc, ent)
                else:
                    name = '%s/tr%.2d-epo%.3d-cr%.4f-acc%.4f-ent%.4f' % (
                        self.cfg['path']['path_save'], self.cnt_train, epoch + 1, cr, acc, ent)

                # Save mode: weather to clean musked weights
                if self.cfg['basic']['pruning_method'] == 'info_bottle' and self.cfg['pruning'].getboolean(
                        'weight_save_clean'):
                    self.save_weight_clean(sess, name)
                else:
                    self.save_weight(sess, name)

        name_train = 'train%d' % self.cnt_train
        self.cfg.add_section(name_train)
        self.cfg.set(name_train, 'n_epochs', str(n_epochs))
        self.cfg.set(name_train, 'lr', str(lr))
        self.cfg.set(name_train, 'acc', str(acc))

        if self.cfg['basic']['pruning_method'] == 'info_bottle':
            self.cfg.set(name_train, 'cr', str(cr))

        self.cnt_train += 1

        return name

    def forward(self, sess, name, save_res=True):
        # 删除之前已经存在的目录
        if os.path.exists(name) and save_res:
            log_t('Remove existed %s' % name)
            os.remove(name)
        if save_res:
            log_t('Save forward results into %s' % name)
        data_loader = self.loader_test.get_data_in_batch(shuffle=False)

        time_sum = 0
        total_entropy = 0
        n_batch = 1

        try:
            while True:
                x, y = data_loader.__next__()
                time_start = time.time()
                logits = sess.run(self.op_logits, feed_dict={self.is_training: False, self.X_: x, self.Y_: y})
                time_sum = time_sum + time.time() - time_start

                entropy = sess.run(self.op_entropy, feed_dict={self.is_training: False, self.X_: x, self.Y_: y})
                total_entropy += entropy

                n_batch += 1

                if save_res:
                    pd.DataFrame(logits).to_csv(name, mode='a', header=False, index=False, sep=' ')
                    break

        except IndexError:
            pass

        log_t('Done with %.4fs, Entropy: %.4f' % (time_sum, total_entropy / n_batch))
        return time_sum

    def save_weight_clean(self, sess, save_path):
        weight_dict = self.fetch_weight(sess)

        n_lstm = self.cfg['structure'].getint('n_layers_lstm')
        n_fc = self.cfg['structure'].getint('n_layers_fc')

        structure = json.loads(self.cfg['structure']['layers'])

        mask_last = [True for _ in range(self.cfg['data'].getint('dimension'))]

        for index_layer, layer in enumerate(self.layers[:n_lstm]):
            name_prefix = '/'.join(layer.scope_name.split('/')[1:])

            mask = sess.run(layer.get_mask(self.cfg['pruning'].getfloat('threshold'))).astype(bool)

            mask_kernel = np.concatenate([mask, mask, mask, mask]).astype(bool)
            mask_input = np.concatenate([mask_last, mask]).astype(bool)

            weight_dict[name_prefix + '/weights'] = weight_dict[name_prefix + '/weights'][mask_input, :][:, mask_kernel]
            weight_dict[name_prefix + '/biases'] = weight_dict[name_prefix + '/biases'][mask_kernel]

            weight_dict[name_prefix + '/mu'] = weight_dict[name_prefix + '/mu'][mask]
            weight_dict[name_prefix + '/logD'] = weight_dict[name_prefix + '/logD'][mask]

            mask_last = mask

            structure[index_layer] = np.int(np.sum(mask))

        # save structure
        self.cfg['structure']['layers'] = str(structure)

        for layer_index in range(n_lstm, n_lstm + n_fc):
            layer_name = 'fc%d' % (layer_index + 1)

            # output layer
            weight_dict[layer_name + '/weights'] = weight_dict[layer_name + '/weights'][mask_last, :]

        file_handler = open(save_path, 'wb')
        pickle.dump(weight_dict, file_handler)
        file_handler.close()

    def get_CR(self, sess):
        # Obtain all masks
        masks = list()
        for layer in self.layers:
            if layer.layer_type == 'C_ib' or layer.layer_type == 'F_ib' or layer.layer_type == 'L':
                masks += [layer.get_mask(threshold=self.cfg['pruning'].getfloat('threshold'))]

        masks = sess.run(masks)
        n_classes = self.cfg['data'].getint('n_classes')

        # how many channels/dims are reserved in each layer
        reserve_state = [np.sum(mask == 1) for mask in masks]

        dim_lstm = self.cfg['structure'].getint('dim_lstm')

        total_params, pruned_params, remain_params = 0, 0, 0

        # for conv layers
        in_channels, in_reserve = self.cfg['data'].getint('dimension'), self.cfg['data'].getint('dimension')
        for n, n_out in enumerate(json.loads(self.cfg['structure']['layers'])[1:-1]):
            # lstm
            total_params += (in_channels + dim_lstm) * 4 * dim_lstm
            remain_params += (in_reserve + reserve_state[n]) * 4 * reserve_state[n]

            # For next layer
            in_channels = dim_lstm
            in_reserve = reserve_state[n]

        # Output layer
        total_params += in_channels * n_classes
        remain_params += in_reserve * n_classes
        pruned_params = total_params - remain_params

        cr = np.around(float(total_params - pruned_params) / total_params, decimals=4)

        log(
            'Total parameters: {}, Pruned parameters: {}, Remaining params:{}, CR: {}, Each layer reserved: {}'.format(
                total_params, pruned_params, remain_params, cr, reserve_state))

        return cr


def exp(task_name, plan_train, pruning=False, pruning_set=None, path_load=None):
    """
    train and prune the network
    Args:
        task_name: dataset
        plan_train: training plan
        pruning: false to train a normal model
        pruning_set: vib pruning settings
        path_load:

    Returns:

    """
    # Specific dataset and model
    cfg = get_cfg(dataset_name=task_name, file_cfg_model='net_lstm.cfg')

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    cfg['basic']['task_name'] = task_name
    cfg['basic']['time_stamp'] = time_stamp

    if pruning:
        cfg['basic']['pruning_method'] = 'info_bottle'

        cfg.add_section('pruning')
        for key in pruning_set.keys():
            cfg.set('pruning', key, str(pruning_set[key]))
    else:
        cfg['basic']['pruning_method'] = str(None)

    cfg['path']['path_load'] = str(path_load)
    cfg['path']['path_save'] += '%s-%s' % (task_name, time_stamp)
    cfg['path']['path_cfg'] = cfg['path']['path_save'] + '/cfg.ini'
    cfg['path']['path_log'] = cfg['path']['path_save'] + '/log.log'
    cfg['path']['path_dataset'] = cfg['path']['path_dataset'] + cfg['basic']['task_name'] + '/'

    logger(cfg['path']['path_log'])

    # save dir
    if not os.path.exists(cfg['path']['path_save']):
        os.mkdir(cfg['path']['path_save'])

    # init tf
    gpu_config = tf.ConfigProto(log_device_placement=False)
    gpu_config.gpu_options.allow_growth = True
    tf.reset_default_graph()
    sess = tf.Session(config=gpu_config)

    model = LstmNet(cfg)

    sess.run(tf.global_variables_initializer())

    log_l('Pre test')
    model.eval_once(sess, epoch=0)
    if cfg['basic']['pruning_method'] == 'info_bottle':
        model.get_CR(sess)
    log_l('')

    model.save_cfg()
    for plan in plan_train:
        model.train(sess=sess, n_epochs=plan['n_epochs'], lr=plan['lr'])
        model.save_cfg()


def exp_forward(task_name, path_model, save_res, batch_size):
    """
    used to test inference time
    Args:
        task_name:
        path_model:
        save_res:
        batch_size:

    Returns:

    """
    cfg = load_cfg('/'.join(path_model.split('/')[:-1]) + '/cfg.ini', localize=True, task_name=task_name)
    # Change the path of loading model
    cfg['path']['path_load'] = path_model
    cfg['basic']['batch_size'] = str(batch_size)

    logger.record_log = False

    gpu_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)

    with tf.device():
        tf.reset_default_graph()
        sess = tf.Session(config=gpu_config)
        model = LstmNet(cfg, load_train=False)

        sess.run(tf.global_variables_initializer())
        # Path to save forward results
        path_forward = cfg['path']['path_save'] + '/forward-' + path_model.split('/')[-1]

        time_used = model.forward(sess, path_forward, save_res)
    return path_forward, time_used


# Train or prune models with VIBNet
if __name__ == '__main__':
    exp(
        task_name='librispeech',
        plan_train=[{'n_epochs': 20, 'lr': 0.1}, {'n_epochs': 20, 'lr': 0.01}],
        pruning=True,
        pruning_set={'name': 'info_bottle',
                     # kl_factor in loss function
                     'kl_factor': 0.,
                     # kl_mult for each layer
                     'kl_mult_list': [1., 1., 1., 1.],
                     # pruning threshold
                     'threshold': 0.01,
                     'weight_save_clean': True,
                     'entropy_factor': 0.
                     },
        path_load=None
    )
