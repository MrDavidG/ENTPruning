# encoding: utf-8
import sys

sys.path.append('..')

from models.bilstm_ner import BiLstmNet
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


def pruning_mnp(task_name, path_model, percent, plan_pruning, batch_size, entropy_factor, cnt_train):
    # load model
    path_cfg = '/'.join(path_model.split('/')[:-1]) + '/cfg.ini'
    if cnt_train != 0:
        cfg = load_cfg(path_cfg, localize=False, task_name=task_name)
    else:
        cfg = get_cfg(dataset_name=task_name, file_cfg_model='net_bilstm.cfg')

    weights_dict = pickle.load(open(path_model, 'rb'))

    mask_last = [True for _ in range(cfg['data'].getint('dimension'))]

    # lstm layer
    def mask_and_save(name_prefix, mask):
        mask_kernel = np.concatenate([mask, mask, mask, mask]).astype(bool)
        mask_input = np.concatenate([mask_last, mask]).astype(bool)

        weights_dict[name_prefix + '/weights'] = weights_dict[name_prefix + '/weights'][mask_input, :][:,
                                                 mask_kernel]
        weights_dict[name_prefix + '/biases'] = weights_dict[name_prefix + '/biases'][mask_kernel]

    # fw
    name_prefix = 'bidirectional_rnn/fw/lstm_cell'
    mask_fw = get_mask_lstm(weights_dict['bidirectional_rnn/fw/lstm_cell/weights'], percent)
    mask_and_save(name_prefix, mask_fw)

    # bw
    name_prefix = 'bidirectional_rnn/bw/lstm_cell'
    mask_bw = get_mask_lstm(weights_dict['bidirectional_rnn/bw/lstm_cell/weights'], percent)
    mask_and_save(name_prefix, mask_bw)

    # ouput layer
    weights_dict['fc2/weights'] = weights_dict['fc2/weights'][np.concatenate([mask_fw, mask_bw]), :]

    # cfg
    if cnt_train == 0:
        time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        cfg['basic']['task_name'] = task_name
        cfg['basic']['time_stamp'] = time_stamp
        cfg['basic']['pruning_methbod'] = 'naive_pruning'
        cfg.add_section('pruning')
        cfg.set('pruning', 'percent', str(percent))

        cfg['path']['path_save'] = '../exp_files/%s-%s' % (task_name, time_stamp)
        cfg['path']['path_cfg'] = cfg['path']['path_save'] + '/cfg.ini'
        cfg['path']['path_log'] = cfg['path']['path_save'] + '/log.log'
        cfg['path']['path_dataset'] = cfg['path']['path_dataset'] + cfg['basic']['task_name']
    cfg.set('basic', 'entropy_factor', str(entropy_factor))

    logger(cfg['path']['path_log'])

    # create dir
    if not os.path.exists(cfg['path']['path_save']):
        os.mkdir(cfg['path']['path_save'])

    # save
    pickle.dump(weights_dict, open(cfg['path']['path_save'] + '/test.pickle', 'wb'))
    cfg['path']['path_load'] = str(cfg['path']['path_save'] + '/test.pickle')

    gpu_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)

    tf.reset_default_graph()
    sess = tf.Session(config=gpu_config)

    model = BiLstmNet(cfg)
    model.cnt_train = cnt_train

    sess.run(tf.global_variables_initializer())

    log_l('Pre test')
    model.eval_once(sess, epoch=0)
    log_l('')

    model.save_cfg()
    for plan in plan_pruning:
        name = model.train(sess=sess, n_epochs=plan['n_epochs'], lr=plan['lr'], save=plan['save'])
        # save cfg
        model.save_cfg()

    return name, model.cnt_train + 1

# prune models with MNP
if __name__ == '__main__':
    # save path of the model
    path_model, cnt_train = '', 0
    for _ in range(10):
        path_model, cnt_train = pruning_mnp(
            task_name='conll45',
            path_model=path_model,
            # pruning ratio in each step
            percent=0.1,
            plan_pruning=[{'n_epochs': 40, 'save': 39, 'lr': 0.001 * 0.9 ** e} for e in range(40)],
            batch_size=50,
            entropy_factor=0.,
            cnt_train=cnt_train,
        )
