# encoding: utf-8
import tensorflow as tf
from data_loader.data_loader import CsvDataLoader
from data_loader.data_loader import SeqCsvDataLoader
from data_loader.data_loader import ConllDataLoader
import pickle
from utils.logger import log_t


class BaseModel:
    def __init__(self, config, load_train, load_test, data_type=None):
        self.regularizer_conv = None
        self.regularizer_fc = None

        self.op_loss = None
        self.op_accuracy = None
        self.op_logits = None
        self.op_opt = None
        self.opt = None
        self.op_predict = None

        self.X = None
        self.Y = None

        self.layers = list()

        self.task_name = config['basic']['task_name']
        self.dataset_path = config['path']['path_dataset'] + config['basic']['task_name'] + '/'

        self.pruning_method = config['basic']['pruning_method']

        self.kl_factor = None
        self.entropy_factor = None

        # VIBNet settings
        if self.pruning_method == 'info_bottle':
            self.kl_total = None
            self.kl_factor = config['pruning'].getfloat('kl_factor')
            self.entropy_factor = config['pruning'].getfloat('entropy_factor')

        self.cnt_train = 0

        self.is_musked = False

        self.set_global_tensor(config['train'].getfloat('regularizer_conv'), config['train'].getfloat('regularizer_fc'))
        self.is_training = tf.placeholder(dtype=tf.bool, name='training')

        # Load data
        if load_train:
            log_t('Loading training data...')

            if data_type == 'seq':
                self.loader_train = SeqCsvDataLoader(
                    # MFCC data is stored in 10 csv files
                    config['path']['path_dataset'] + 'train/Training_train_clean_100_chunk_%s.csv', n_chunks=10,
                    batch_size=config['basic'].getint('batch_size'))

            elif data_type == 'conll':
                self.loader_train = ConllDataLoader(config['path']['path_dataset'] + '/train/word_embedding.csv',
                                                    batch_size=config['basic'].getint('batch_size'),
                                                    dimension=config['data'].getint('dimension'))
            else:
                self.loader_train = CsvDataLoader(config['path']['path_dataset'] + 'train.csv',
                                                  config['basic'].getint('batch_size'))
                config['data']['n_samples_train'] = str(self.loader_train.get_N())
                self.total_batches_train = config['data'].getint('n_samples_train') // config['basic'].getint(
                    'batch_size') + 1
            log_t('Done')

        if load_test:
            log_t('Loading test data... ')
            if data_type == 'seq':
                self.loader_test = SeqCsvDataLoader(
                    config['path']['path_dataset'] + 'test/Testing_test_clean_chunk_%s.csv', n_chunks=1,
                    batch_size=config['basic'].getint('batch_size'))

            elif data_type == 'conll':
                self.loader_test = ConllDataLoader(config['path']['path_dataset'] + '/test/word_embedding.csv',
                                                   batch_size=config['basic'].getint('batch_size'),
                                                   dimension=config['data'].getint('dimension'))
            else:
                self.loader_test = CsvDataLoader(config['path']['path_dataset'] + 'test.csv',
                                                 config['basic'].getint('batch_size'))
                config['data']['n_samples_val'] = str(self.loader_test.get_N())
            log_t('Done')

        self.cfg = config

    def set_global_tensor(self, regu_conv, regu_fc):
        self.regularizer_conv = tf.contrib.layers.l2_regularizer(scale=regu_conv)
        self.regularizer_fc = tf.contrib.layers.l2_regularizer(scale=regu_fc)

    def get_layer(self, layer_name):
        for layer in self.layers:
            if layer.layer_name == layer_name:
                return layer
        return None

    def fetch_weight(self, sess):
        weight_dict = dict()
        weight_list = list()
        for layer in self.layers:
            weight_list.append(layer.get_params(sess))
        for params_dict in weight_list:
            for k, v in params_dict.items():
                weight_dict[k.split(':')[0]] = v
        return weight_dict

    def save_weight(self, sess, save_path):
        file_handler = open(save_path, 'wb')
        pickle.dump(self.fetch_weight(sess), file_handler)
        file_handler.close()

    def is_evaluated(self, epoch, mode='increase'):
        if mode == 'increase':
            return epoch <= 10 or epoch <= 50 and epoch % 5 == 0 or epoch > 50 and epoch % 50 == 0
        elif mode == 'step':
            return epoch % 10 == 0

    def build(self):
        self.inference()
        self.entropy()
        self.loss()
        self.evaluate()

    def save_cfg(self):
        with open(self.cfg['path']['path_cfg'], 'w') as file:
            self.cfg.write(file)
