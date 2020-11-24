# encoding: utf-8
import sys

sys.path.append('..')

from models.lstm_asr import LstmNet
from utils.configer import load_cfg
from utils.configer import get_cfg
from utils.logger import logger
from utils.logger import log_l
from datetime import datetime

import tensorflow as tf
import numpy as np

import pickle
import os


def get_mask_lstm(layer_weights, percent):
    """

    :param layer_weights: [input+hidden, 4*hidden]
    :param percent:
    :return:
    """
    # [4 * hidden]
    weights_sum_abs = np.sum(np.abs(layer_weights), axis=0)

    len_ = np.int(np.shape(weights_sum_abs)[0] / 4)

    l1_norm = weights_sum_abs[:len_] + weights_sum_abs[len_:2 * len_] + weights_sum_abs[2 * len_:3 * len_] + \
              weights_sum_abs[3 * len_:4 * len_]
    list_sort = np.sort(l1_norm)
    sum_threshold = list_sort[int(len(list_sort) * (1 - percent))]
    return sum_threshold > l1_norm


def pruning_naive(task_name, path_model, percent, plan_pruning, batch_size, entropy_factor, cnt_train):
    # load model
    path_cfg = '/'.join(path_model.split('/')[:-1]) + '/cfg.ini'
    if cnt_train:
        cfg = load_cfg(path_cfg, localize=False, task_name=task_name)
    else:
        cfg = get_cfg(dataset_name=task_name, file_cfg_model='net_lstm.cfg')

    weights_dict = pickle.load(open(path_model, 'rb'))

    n_lstm = cfg['structure'].getint('n_layers_lstm')

    mask_last = [True for _ in range(cfg['data'].getint('dimension'))]
    # lstm
    for layer_index in range(n_lstm):
        name_prefix = 'rnn/multi_rnn_cell/cell_%d/lstm_cell' % layer_index

        mask = get_mask_lstm(weights_dict[name_prefix + '/weights'], percent)

        mask_kernel = np.concatenate([mask, mask, mask, mask]).astype(bool)
        mask_input = np.concatenate([mask_last, mask]).astype(bool)

        weights_dict[name_prefix + '/weights'] = weights_dict[name_prefix + '/weights'][mask_input, :][:, mask_kernel]
        weights_dict[name_prefix + '/biases'] = weights_dict[name_prefix + '/biases'][mask_kernel]

        mask_last = mask
    # output layer
    layer_name = 'fc5'
    weights_dict[layer_name + '/weights'] = weights_dict[layer_name + '/weights'][mask_last, :]

    if cnt_train == 0:
        time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        cfg['basic']['task_name'] = task_name
        cfg['basic']['time_stamp'] = time_stamp
        cfg['basic']['batch_size'] = str(batch_size)
        cfg['basic']['pruning_method'] = 'naive_pruning'
        cfg.add_section('pruning')
        cfg.set('pruning', 'percent', str(percent))

        cfg['path']['path_save'] += '%s-%s' % (task_name, time_stamp)
        cfg['path']['path_cfg'] = cfg['path']['path_save'] + '/cfg.ini'
        cfg['path']['path_log'] = cfg['path']['path_save'] + '/log.log'
        cfg['path']['path_dataset'] = cfg['path']['path_dataset'] + cfg['basic']['task_name'] + '/'

    cfg.set('basic', 'entropy_factor', str(entropy_factor))
    logger(cfg['path']['path_log'])

    # save dir
    if not os.path.exists(cfg['path']['path_save']):
        os.mkdir(cfg['path']['path_save'])

    # save model
    pickle.dump(weights_dict, open(cfg['path']['path_save'] + '/test.pickle', 'wb'))
    cfg['path']['path_load'] = str(cfg['path']['path_save'] + '/test.pickle')

    gpu_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)

    tf.reset_default_graph()
    sess = tf.Session(config=gpu_config)

    model = LstmNet(cfg)
    model.cnt_train = cnt_train

    sess.run(tf.global_variables_initializer())

    log_l('Pre test')
    model.eval_once(sess, epoch=0)
    log_l('')

    model.save_cfg()
    for plan in plan_pruning:
        name = model.train(sess=sess, n_epochs=plan['n_epochs'], lr=plan['lr'], save=plan['save'])
        model.save_cfg()

    return name, model.cnt_train + 1


# prune models with MNP
if __name__ == '__main__':
    #
    path_model, cnt_train = '', 0
    for i in range(10):
        path_model, cnt_train = pruning_naive(
            task_name='librispeech',
            path_model=path_model,
            # pruning ratio in each step
            percent=0.1,
            plan_pruning=[
                {'n_epochs': 30, 'lr': 0.1, 'save': False}
            ],
            batch_size=32,
            entropy_factor=0.,
            cnt_train=cnt_train
        )
