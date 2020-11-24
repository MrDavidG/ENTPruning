# encoding: utf-8
import sys

sys.path.append('..')

from models.base_model import BaseModel
from layers.fc_lstm_layer import FullConnectedLayer
from layers.lstm_cell import LSTMCell
from utils.configer import get_cfg, load_cfg
from utils.logger import *
from utils.beamsearch.beamsearch import beamsearch

import tensorflow as tf
import numpy as np

import json
import pickle
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class BiLstmNet(BaseModel):
    def __init__(self, config, load_train=True, load_test=True):
        super(BiLstmNet, self).__init__(config, load_train, load_test, data_type='conll')
        # [batch_size, n_step, dimension]
        self.X_ = tf.placeholder(dtype=tf.float32, shape=[None, None, self.cfg['data'].getint('dimension')])
        # [batch_size, n_steps]
        self.Y_ = tf.placeholder(dtype=tf.int32)

        self.seq_lengths = tf.placeholder(dtype=tf.int32)

        model_path = config['path']['path_load']

        if model_path and os.path.exists(model_path):
            log_t('Loading weight from %s' % model_path)
            self.load_model(model_path)
        else:
            log_l('Initialize weight matrix')
            self.weight_dict = self.construct_initial_weights()

        self.op_acc_forward = None
        self.op_acc_decode = None
        self.trans_params = None

        if self.cfg['basic'].getfloat('entropy_factor') is not None:
            self.entropy_factor = self.cfg['basic'].getfloat('entropy_factor')

        self.build()

    def build(self):
        self.inference()
        self.entropy()
        self.loss()
        self.decode()
        self.evaluate()

    def load_model(self, model_path):
        self.weight_dict = pickle.load(open(model_path, 'rb'), encoding='bytes')
        if 'trans_params' in self.weight_dict.keys():
            log_t('Loading trans_params from load model ...')
            self.trans_params = self.weight_dict['trans_params']

        # reset
        if self.cfg['structure'].getint('n_layers_lstm') == 1:
            hidden_size_fw = np.shape(self.weight_dict['bidirectional_rnn/fw/lstm_cell/weights'])[1] / 4
            hidden_size_bw = np.shape(self.weight_dict['bidirectional_rnn/bw/lstm_cell/weights'])[1] / 4

            self.cfg['structure']['layers'] = str(
                [self.cfg['data'].getint('dimension')] + [[hidden_size_fw, hidden_size_bw]] + [
                    self.cfg['data'].getint('n_classes')])
        else:
            pass

        log_t('Loading models as %s' % self.cfg['structure']['layers'])

    def construct_initial_weights(self):
        # only create fc tensors
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
            dim_in = dim_layers[index_layer - 1][0] + dim_layers[index_layer - 1][1]

            shape_weights = [dim_in, dim_layers[index_layer]]

            weight_dict['fc%d/weights' % (index_layer)] = weight_variable(shape_weights)
            weight_dict['fc%d/biases' % (index_layer)] = bias_variable(dim_layers[index_layer])

        return weight_dict

    def inference(self):
        self.layers.clear()
        with tf.variable_scope(self.task_name, reuse=tf.AUTO_REUSE):
            if self.cfg['basic']['pruning_method'] == 'info_bottle':
                self.kl_total = 0.
            y = self.X_

            y = tf.layers.dropout(y, rate=0.5, training=self.is_training)

            n_lstm = self.cfg['structure'].getint('n_layers_lstm')
            n_fc = self.cfg['structure'].getint('n_layers_fc')

            hidden_size = json.loads(self.cfg['structure']['layers'])[1:-1]

            # lstm
            flag = self.cfg['basic']['pruning_method'] == 'info_bottle'
            batch_size = self.cfg['basic'].getint('batch_size')

            if n_lstm == 1:
                lstm_cell_fw = LSTMCell(self.weight_dict, is_training=self.is_training, pruning=flag,
                                        batch_size=batch_size, layer_type='L_f', num_units=hidden_size[0][0])
                lstm_cell_bw = LSTMCell(self.weight_dict, is_training=self.is_training, pruning=flag,
                                        batch_size=batch_size, layer_type='L_b', num_units=hidden_size[0][1])

                self.layers.append(lstm_cell_fw)
                self.layers.append(lstm_cell_bw)
            else:
                cell_list_fw = [
                    LSTMCell(self.weight_dict, is_training=self.is_training, pruning=flag, batch_size=batch_size,
                             layer_type='L_f', num_units=hidden_size[i][0]) for i in range(n_lstm)]

                cell_list_bw = [
                    LSTMCell(self.weight_dict, is_training=self.is_training, pruning=flag, batch_size=batch_size,
                             layer_type='L_b', num_units=hidden_size[i][1]) for i in range(n_lstm)]

                lstm_cell_fw = tf.contrib.rnn.MultiRNNCell(cell_list_fw)
                lstm_cell_bw = tf.contrib.rnn.MultiRNNCell(cell_list_bw)

                self.layers += [cell_list_fw[i] for i in range(n_lstm)]
                self.layers += [cell_list_bw[i] for i in range(n_lstm)]

            _init_state_fw = lstm_cell_fw.zero_state(self.cfg['basic'].getint('batch_size'), dtype=tf.float32)
            _init_state_bw = lstm_cell_bw.zero_state(self.cfg['basic'].getint('batch_size'), dtype=tf.float32)

            # outputs: [batch_size, step, hidden_size]
            (y_fw, y_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw, cell_bw=lstm_cell_bw,
                                                              inputs=y,
                                                              initial_state_fw=_init_state_fw,
                                                              initial_state_bw=_init_state_bw,
                                                              dtype=tf.float32, sequence_length=self.seq_lengths,
                                                              time_major=False)

            y = tf.concat([y_fw, y_bw], axis=-1)

            y = tf.layers.dropout(y, rate=0.5, training=self.is_training)

            if self.cfg['basic']['pruning_method'] == 'info_bottle':
                if n_lstm == 1:
                    self.kl_total += lstm_cell_fw.get_kld() + lstm_cell_bw.get_kld()
                else:
                    for cell in cell_list_fw:
                        self.kl_total += cell.get_kld()
                    for cell in cell_list_bw:
                        self.kl_total += cell.get_kld()

            # output
            with tf.variable_scope('fc%d' % (n_lstm + n_fc)):
                fc_layer = FullConnectedLayer(self.weight_dict, regularizer_fc=self.regularizer_fc,
                                              is_musked=self.is_musked)
                self.layers.append(fc_layer)

                # [batch_size, n_step, hidden_size]
                y = fc_layer.forward(y)

                self.op_logits = y

    def loss(self):
        # self.Y_: [batch_size, n_step]
        # self.op_logits: [batch_size, n_step, hidden_size]
        loglikelihood, self.trans_params = tf.contrib.crf.crf_log_likelihood(self.op_logits, self.Y_, self.seq_lengths,
                                                                             self.trans_params)
        l2_loss = tf.losses.get_regularization_loss()

        if self.cfg['basic']['pruning_method'] == 'info_bottle':
            self.op_loss = tf.reduce_mean(-loglikelihood,
                                          name='loss') + l2_loss + self.kl_factor * self.kl_total + self.entropy_factor * self.op_entropy
        else:
            self.op_loss = tf.reduce_mean(-loglikelihood, name='loss') + l2_loss + self.entropy_factor * self.op_entropy

    def decode(self):
        # self.op_decode: [batch_size, n_step]
        self.op_decode_crf, _ = tf.contrib.crf.crf_decode(self.op_logits, self.trans_params, self.seq_lengths)

    def optimize(self, lr):
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            self.opt = tf.train.AdamOptimizer(learning_rate=lr)
            self.op_opt = self.opt.minimize(self.op_loss)

    def evaluate(self):
        num = tf.cast(tf.reduce_sum(self.seq_lengths), dtype=tf.float32)

        # [batch_size, n_step, n_tag]==>[batch_size, n_step]
        logits_f = tf.argmax(self.op_logits, axis=-1, output_type=tf.int32)
        mask = tf.cast(tf.sequence_mask(self.seq_lengths, tf.reduce_max(self.seq_lengths)), dtype=tf.float32)
        # [batch_size, n_steps]
        res_predict = tf.cast(tf.equal(logits_f, self.Y_), dtype=tf.float32) * mask
        self.op_acc_forward = tf.reduce_sum(res_predict) / num

        # [batch_size, n_step]
        res_decode = tf.cast(tf.equal(self.op_decode_crf, self.Y_), dtype=tf.float32) * mask
        self.op_acc_decode = tf.reduce_sum(res_decode) / num

    def entropy(self):
        with tf.name_scope('entropy'):
            # [batch_size, n_step, n_tag] ==> [batch_size*n_step, n_tag]
            logits = tf.reshape(self.op_logits, shape=[-1, self.cfg['data'].getint('n_classes')])
            # [batch_size*n_step, n_tag]
            reduce_mean = logits - tf.reshape(tf.reduce_max(logits, axis=-1), shape=(-1, 1))
            score = tf.exp(reduce_mean) / tf.reshape(tf.reduce_sum(tf.exp(reduce_mean), axis=1), shape=(-1, 1))

            self.op_entropy = - tf.reduce_mean(tf.reduce_sum(score * tf.log(score + 1e-30), axis=1))

    def train_one_epoch(self, sess, epoch):
        avg_loss = 0
        avg_kl_loss = 0
        avg_entropy = 0
        n_batches = 1

        time_last = time.time()

        data_loader = self.loader_train.get_data_in_batch()
        try:
            while True:
                x, y, seq_lens = data_loader.__next__()

                if self.cfg['basic']['pruning_method'] == 'info_bottle':
                    _, loss, loss_kl, entropy_batch = sess.run(
                        [self.op_opt, self.op_loss, self.kl_total, self.op_entropy],
                        feed_dict={self.is_training: True, self.X_: x, self.Y_: y, self.seq_lengths: seq_lens})

                    avg_kl_loss += (loss_kl * self.kl_factor - avg_kl_loss) / n_batches
                    avg_entropy += (entropy_batch - avg_entropy) / n_batches
                else:
                    _, loss = sess.run([self.op_opt, self.op_loss],
                                       feed_dict={self.is_training: True, self.X_: x, self.Y_: y,
                                                  self.seq_lengths: seq_lens})

                avg_loss += (loss - avg_loss) / n_batches

                n_batches += 1

                if n_batches % 5 == 0:

                    if self.cfg['basic']['pruning_method'] == 'info_bottle':
                        str_ = 'epoch={:d}, batch={:d}/{:d}, curr_loss={:f}, kl_loss={:f}, entropy={:.4f}, used_time:{:.4f}s'.format(
                            epoch + 1,
                            n_batches,
                            self.loader_train.get_N() // self.loader_train.batch_size + 1,
                            avg_loss,
                            avg_kl_loss,
                            # avg_acc,
                            avg_entropy,
                            time.time() - time_last)
                    else:
                        str_ = 'epoch={:d}, batch={:d}/{:d}, curr_loss={:f}, used_time:{:.4f}s'.format(
                            epoch + 1,
                            n_batches,
                            self.loader_train.get_N() // self.loader_train.batch_size + 1,
                            avg_loss,
                            # avg_acc,
                            time.time() - time_last)

                    print('\r' + str_, end=' ')

                    time_last = time.time()

        except IndexError:
            pass

        log(str_, need_print=False)

    def eval_once(self, sess, epoch):
        total_loss = 0
        total_entropy = 0
        total_kl_loss = 0
        total_acc_forward = 0
        total_acc_decode = 0
        n_batches = 0

        data_loader = self.loader_test.get_data_in_batch()
        try:
            while True:
                x, y, seq_lens = data_loader.__next__()
                if self.cfg['basic']['pruning_method'] == 'info_bottle':
                    loss_batch, loss_kl, acc_forward, acc_decode, entropy_batch = sess.run(
                        [self.op_loss, self.kl_total, self.op_acc_forward, self.op_acc_decode, self.op_entropy],
                        feed_dict={self.is_training: False, self.X_: x, self.Y_: y, self.seq_lengths: seq_lens})
                    total_kl_loss += loss_kl
                else:
                    loss_batch, acc_forward, acc_decode, entropy_batch = sess.run(
                        [self.op_loss, self.op_acc_forward, self.op_acc_decode, self.op_entropy],
                        feed_dict={self.is_training: False, self.X_: x, self.Y_: y, self.seq_lengths: seq_lens})

                total_loss += loss_batch
                total_entropy += entropy_batch
                total_acc_forward += acc_forward
                total_acc_decode += acc_decode
                n_batches += 1

        except IndexError:
            pass

        acc_forward = np.around(total_acc_forward / n_batches, decimals=4)
        acc_decode = np.around(total_acc_decode / n_batches, decimals=4)
        avg_entropy = np.around(total_entropy / n_batches, decimals=4)

        if self.cfg['basic']['pruning_method'] == 'info_bottle':
            log(
                '\nEpoch:{:d}, val_acc_for={:%}, val_acc_dec={:%}, val_loss={:f}, kl_loss={:f}, val_entropy={:f}'.format(
                    epoch + 1,
                    acc_forward,
                    acc_decode,
                    total_loss / n_batches,
                    total_kl_loss * self.kl_factor / n_batches,
                    avg_entropy))
        else:
            log('\nEpoch:{:d}, val_acc_for={:%}, val_acc_dec={:%}, val_loss={:f}, val_entropy={:f}'.format(epoch + 1,
                                                                                                           acc_forward,
                                                                                                           acc_decode,
                                                                                                           total_loss / n_batches,
                                                                                                           avg_entropy))
        return acc_forward, acc_decode, avg_entropy

    def run_evaluate(self, labels, labels_pred, sequence_lengths):
        """

        :param labels:  [batch_size, n_step]
        :param labels_pred:     [batch_size, n_step]
        :param sequence_lengths:    [batch_size]
        :return:
        """
        accs = []
        for lab, lab_pred, length in zip(labels, labels_pred, sequence_lengths):
            lab = lab[:length]
            lab_pred = lab_pred[:length]
            accs += [a == b for (a, b) in zip(lab, lab_pred)]
        return np.mean(accs)

    def train(self, sess, n_epochs, lr, save=False):
        self.optimize(lr)

        name = None

        sess.run(tf.variables_initializer(self.opt.variables()))
        for epoch in range(n_epochs):
            self.train_one_epoch(sess, epoch)
            accf, accd, ent = self.eval_once(sess, epoch)

            if self.cfg['basic']['pruning_method'] == 'info_bottle':
                cr = self.get_CR(sess)

            if self.is_evaluated(epoch + 1, mode='step') or save:
                if self.cfg['basic']['pruning_method'] != 'info_bottle':
                    name = '%s/tr%.2d-epo%.3d-accf%.4f-accd%.4f-ent%.4f' % (
                        self.cfg['path']['path_save'], self.cnt_train, epoch + 1, accf, accd, ent)
                else:
                    name = '%s/tr%.2d-epo%.3d-cr%.4f-accf%.4f-accd%.4f-ent%.4f' % (
                        self.cfg['path']['path_save'], self.cnt_train, epoch + 1, cr, accf, accd, ent)

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
        self.cfg.set(name_train, 'acc_forward', str(accf))
        self.cfg.set(name_train, 'acc_decode', str(accd))

        if self.cfg['basic']['pruning_method'] == 'info_bottle':
            self.cfg.set(name_train, 'cr', str(cr))

        self.cnt_train += 1

        return name

    def forward(self, sess, name, decode=True, beam_width=None, save_res=True):
        if os.path.exists(name) and save_res:
            log_t('Remove existed %s' % name)
            os.remove(name)
        if save_res:
            log_t('Save forward results into %s' % name)
        data_loader = self.loader_test.get_data_in_batch()

        n_batches = 1

        time_inference, time_decode = 0, 0

        acc_decode = 0

        total_entropy = 0

        res = dict()
        try:
            while True:
                x, y, seq_lens = data_loader.__next__()
                time_start = time.time()
                logits, trans_params = sess.run([self.op_logits, self.trans_params],
                                                feed_dict={self.is_training: False, self.X_: x, self.Y_: y,
                                                           self.seq_lengths: seq_lens})
                time_inference = time_inference + time.time() - time_start

                entropy = sess.run(self.op_entropy, feed_dict={self.is_training: False, self.X_: x, self.Y_: y,
                                                               self.seq_lengths: seq_lens})
                total_entropy += entropy

                if save_res:
                    res['l%d' % n_batches] = logits
                    res['t%d' % n_batches] = trans_params
                    res['s%d' % n_batches] = seq_lens

                # decode
                if decode:
                    # paths, y: [batch_size, n_step]
                    paths, time_decode_batch = beamsearch(logits, trans_params, seq_lens, beam_width)
                    time_decode += time_decode_batch
                    sum_lens = np.sum(seq_lens)
                    acc_batch = (np.sum(paths == y) - (np.prod(np.shape(y)) - sum_lens)) / sum_lens
                    acc_decode += (acc_batch - acc_decode) / n_batches

                n_batches += 1
        except IndexError:
            pass
        if save_res:
            pickle.dump(res, open(name, 'wb'))
        if decode:
            log_t(
                'Inference time: %.4fs, Entropy: %.4f, Decode time: %.4fs, Accuracy: %.4f' % (
                    time_inference, total_entropy / n_batches, time_decode, acc_decode))
        else:
            log_t(
                'Inference time: %.4fs, Entropy: %.4f' % (time_inference, total_entropy / n_batches))

        return time_inference, time_decode

    def save_weight(self, sess, save_path):
        file_handler = open(save_path, 'wb')
        weight_dict = self.fetch_weight(sess)
        weight_dict['trans_params'] = sess.run(self.trans_params)
        pickle.dump(weight_dict, file_handler)
        file_handler.close()

    def save_weight_clean(self, sess, save_path):
        def save_unidirectional_weight(dir_):
            # forward and backward
            mask_last = [True for _ in range(self.cfg['data'].getint('dimension'))]

            layers = list()
            for layer in self.layers:
                if layer.layer_type == dir_:
                    layers.append(layer)

            if dir_ == 'L_f':
                index_dir = 0
            elif dir_ == 'L_b':
                index_dir = 1

            for index_layer, layer in enumerate(layers):
                name_prefix = '/'.join(layer.scope_name.split('/')[1:])

                mask = sess.run(layer.get_mask(self.cfg['pruning'].getfloat('threshold'))).astype(bool)

                mask_kernel = np.concatenate([mask, mask, mask, mask]).astype(bool)
                mask_input = np.concatenate([mask_last, mask]).astype(bool)

                weight_dict[name_prefix + '/weights'] = weight_dict[name_prefix + '/weights'][mask_input, :][:,
                                                        mask_kernel]
                weight_dict[name_prefix + '/biases'] = weight_dict[name_prefix + '/biases'][mask_kernel]

                weight_dict[name_prefix + '/mu'] = weight_dict[name_prefix + '/mu'][mask]
                weight_dict[name_prefix + '/logD'] = weight_dict[name_prefix + '/logD'][mask]

                mask_last = mask

                structure[index_layer + 1][index_dir] = np.int(np.sum(mask))

            return mask_last

        weight_dict = self.fetch_weight(sess)

        n_lstm = self.cfg['structure'].getint('n_layers_lstm')
        n_fc = self.cfg['structure'].getint('n_layers_fc')

        structure = json.loads(self.cfg['structure']['layers'])

        mask_f = save_unidirectional_weight('L_f')
        mask_b = save_unidirectional_weight('L_b')
        mask_last = np.concatenate([mask_f, mask_b])

        self.cfg['structure']['layers'] = str(structure)

        for layer_index in range(n_lstm, n_lstm + n_fc):
            layer_name = 'fc%d' % (layer_index + 1)

            # output layer
            weight_dict[layer_name + '/weights'] = weight_dict[layer_name + '/weights'][mask_last, :]

        weight_dict['trans_params'] = sess.run(self.trans_params)

        file_handler = open(save_path, 'wb')
        pickle.dump(weight_dict, file_handler)
        file_handler.close()

    def get_CR(self, sess):
        def get_unidirectional_params(sess, masks):
            masks = sess.run(masks)

            # how many channels/dims are reserved in each layer
            reserve_state = [np.sum(mask == 1) for mask in masks]

            dim_lstm = self.cfg['structure'].getint('dim_lstm')

            total_params, remain_params = 0, 0

            in_channels, in_reserve = self.cfg['data'].getint('dimension'), self.cfg['data'].getint('dimension')
            for n in range(self.cfg['structure'].getint('n_layers_lstm')):
                # lstm
                total_params += (in_channels + dim_lstm) * 4 * dim_lstm
                remain_params += (in_reserve + reserve_state[n]) * 4 * reserve_state[n]

                # For next layer
                in_channels = dim_lstm
                in_reserve = reserve_state[n]

            return total_params, remain_params, in_channels, in_reserve, reserve_state

        masks_fw = list()
        masks_bw = list()
        for layer in self.layers:
            if layer.layer_type == 'L_f':
                masks_fw += [layer.get_mask(threshold=self.cfg['pruning'].getfloat('threshold'))]
            elif layer.layer_type == 'L_b':
                masks_bw += [layer.get_mask(threshold=self.cfg['pruning'].getfloat('threshold'))]

        total_params_fw, remain_params_fw, in_channels_fw, in_reserve_fw, reserve_state_fw = get_unidirectional_params(
            sess, masks_fw)
        total_params_bw, remain_params_bw, in_channels_bw, in_reserve_bw, reserve_state_bw = get_unidirectional_params(
            sess, masks_bw)

        n_classes = self.cfg['data'].getint('n_classes')

        total_params = total_params_fw + total_params_bw + (in_channels_fw + in_channels_bw) * n_classes
        remain_params = remain_params_fw + remain_params_bw + (in_reserve_fw + in_channels_bw) * n_classes
        pruned_params = total_params - remain_params

        cr = np.around(float(remain_params) / total_params, decimals=4)

        log(
            'Total | Pruned | Remain params: {} | {} | {}, CR: {}, Reserved fw | bw: {} | {}'.format(
                total_params, pruned_params, remain_params, cr, reserve_state_fw, reserve_state_bw))

        return cr


def exp_eval(task_name, path_model, batch_size):
    cfg = load_cfg('/'.join(path_model.split('/')[:-1]) + '/cfg.ini', localize=True, task_name=task_name)
    # Change the path of loading model
    cfg['path']['path_load'] = path_model
    cfg['basic']['batch_size'] = str(batch_size)

    logger.record_log = False

    gpu_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)

    tf.reset_default_graph()

    sess = tf.Session(config=gpu_config)
    model = BiLstmNet(cfg, load_train=False)

    sess.run(tf.global_variables_initializer())

    model.eval_once(sess, -1)


def exp(task_name, plan_train, pruning=False, pruning_set=None, path_load=None):
    """
    train and prune the network
    Args:
        task_name: dataset
        plan_train: training plan
        pruning: false if you are to train an normal model
        pruning_set: vib pruning settings
        path_load:

    Returns:

    """
    # Specific dataset and model
    cfg = get_cfg(dataset_name=task_name, file_cfg_model='net_bilstm.cfg')

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
    gpu_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    tf.reset_default_graph()
    sess = tf.Session(config=gpu_config)

    model = BiLstmNet(cfg)

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


def exp_forward(task_name, path_model, save_res, decode, batch_size, beam_width=0.1, device='/gpu:0', log=False):
    """
    used to test inference time
    Args:
        task_name: specific dataset
        path_model:
        save_res:
        decode:
        batch_size:
        beam_width:
        device:
        log:

    Returns:

    """
    cfg = load_cfg('/'.join(path_model.split('/')[:-1]) + '/cfg.ini', localize=True, task_name=task_name)
    # Change the path of loading model
    cfg['path']['path_load'] = path_model
    cfg['basic']['batch_size'] = str(batch_size)

    logger.record_log = False

    gpu_config = tf.ConfigProto(log_device_placement=log, allow_soft_placement=True)

    tf.reset_default_graph()

    with tf.device(device):
        sess = tf.Session(config=gpu_config)
        model = BiLstmNet(cfg, load_train=False)

        sess.run(tf.global_variables_initializer())
        # Path to save forward results
        path_forward = cfg['path']['path_save'] + '/forward-' + path_model.split('/')[-1]

        time_used = model.forward(sess, path_forward, decode=decode, beam_width=beam_width, save_res=save_res)

    return path_forward, time_used


# Train or prune models with VIBNet
if __name__ == '__main__':
    plan_train = [{'n_epochs': 20, 'lr': 0.001 * 0.9 ** e} for e in range(30)]
    plan_pruning = [{'n_epochs': 20, 'lr': 0.01},
                    {'n_epochs': 20, 'lr': 0.001},
                    {'n_epochs': 20, 'lr': 0.0001}]

    exp(
        task_name='conll45',
        plan_train=plan_train,  # plan_pruning
        pruning=False,
        pruning_set={
            'name': 'info_bottle',
            # kl_factor in loss function
            'kl_factor': 0.,
            # kl_mult for each layer
            'kl_mult_list': list(),
            # pruning threshold
            'threshold': 0.01,
            'weight_save_clean': True,
            'entropy_factor': 0.
        },
        path_load=None
    )
