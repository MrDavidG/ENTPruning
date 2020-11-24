# encoding: utf-8
from utils.logger import log_t

import pandas as pd
import numpy as np
import pickle
import random


class ConllDataLoader:
    def __init__(self, dataset_path, batch_size, dimension):
        # [n_word, 301]
        self.dataset = pd.read_csv(dataset_path, header=None, sep=',').values
        # [n_seq]
        self.seq_lens = np.int32(np.reshape(
            pd.read_csv('/'.join(dataset_path.split('/')[:-1]) + '/sentence_len.csv', header=None, sep=',').values,
            newshape=[-1]))

        self.batch_size = batch_size
        self.dimension = dimension

        self.n_sample = self.seq_lens.shape[0]
        # adopt the last batch
        self.n_batch = np.int(self.n_sample / self.batch_size)
        self.word_index_array = self.get_start_end_index(self.seq_lens)

    def get_start_end_index(self, seq_len):
        array = np.zeros(shape=[self.n_sample, 2], dtype=np.int32)
        n_word = 0
        for index_len, len_ in enumerate(seq_len):
            # not including array[index_len][1]
            array[index_len][0] = n_word
            array[index_len][1] = n_word + len_
            n_word += len_
        return array

    def get_N(self):
        return self.n_sample

    def get_data_in_batch(self):
        for index_batch in range(self.n_batch):
            # max seq len in this batch
            max_step = np.max(self.seq_lens[index_batch * self.batch_size: (index_batch + 1) * self.batch_size])

            data_batch = np.zeros(shape=[self.batch_size, max_step, self.dimension + 1])

            # get batched data
            for k in range(self.batch_size):
                index_seq = index_batch * self.batch_size + k
                # len
                len_seq = self.seq_lens[index_seq]
                # shape: [len_seq, 301]
                data_batch[k, :len_seq, :] = self.dataset[
                                             self.word_index_array[index_seq][0]:self.word_index_array[index_seq][1], :]

            labels = np.reshape(data_batch[:, :, -1], newshape=[self.batch_size, -1])
            # [batch_size, n_step, dimension]
            # [batch_size, n_step]
            yield data_batch[:, :, :-1], \
                  np.int32(data_batch[:, :, -1]), \
                  self.seq_lens[index_batch * self.batch_size: (index_batch + 1) * self.batch_size]

        raise IndexError


class CsvDataLoader():
    def __init__(self, dataset_path, batch_size=128):
        self.dataset = pd.read_csv(dataset_path, header=None, delim_whitespace=True).values
        self.batch_size = batch_size

    def get_N(self):
        return self.dataset.shape[0]

    def get_data_in_batch(self, shuffle=True):
        if shuffle:
            np.random.shuffle(self.dataset)

        num_batch = 0
        while True:
            if num_batch * self.batch_size > self.dataset.shape[0]:
                raise IndexError
            elif (num_batch + 1) * self.batch_size > self.dataset.shape[0]:
                data_in_batch = self.dataset[num_batch * self.batch_size:, :]
            else:
                data_in_batch = self.dataset[num_batch * self.batch_size:(num_batch + 1) * self.batch_size, :]

            yield data_in_batch[:, :-1], data_in_batch[:, -1]

            num_batch += 1


class SeqCsvDataLoader():
    def __init__(self, dataset_path, n_chunks, batch_size=16):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.n_chunks = n_chunks
        self.chunk_info = pickle.load(open('/'.join(dataset_path.split('/')[:-1]) + '/chunk_info', 'rb'),
                                      encoding='bytes')
        self.dataset = None
        self.n_samples = None
        self.ck = 0

    def get_N(self):
        return self.n_samples

    def get_data_in_batch(self, shuffle=False):
        if shuffle:
            np.random.shuffle(self.dataset)

        while True:
            for ck in range(0, self.n_chunks):

                self.ck = ck

                log_t('Loading data in chunk %d ...' % ck)
                self.dataset = pd.read_csv(self.dataset_path % ck, header=None, sep=',').values
                log_t('Done')

                data_name = self.chunk_info['chunk%s/data_name' % ck]
                data_end_index = self.chunk_info['chunk%s/data_end_index' % ck]
                arr_snt_len = self.chunk_info['chunk%s/arr_snt_len' % ck]

                self.n_samples = len(data_name)

                n_snt = len(data_name)  # # of sentences
                n_batches = int(n_snt / self.batch_size)

                snt_index = 0
                beg_snt = 0

                inp_dim = self.dataset.shape[1]

                for i in range(n_batches):

                    max_len = int(max(arr_snt_len[snt_index:snt_index + self.batch_size]))
                    inp = np.zeros((max_len, self.batch_size, inp_dim))

                    for k in range(self.batch_size):
                        snt_len = data_end_index[snt_index] - beg_snt
                        n_zeros = max_len - snt_len

                        n_zeros_left = random.randint(0, n_zeros)
                        inp[n_zeros_left:n_zeros_left + snt_len, k, :] = self.dataset[beg_snt:beg_snt + snt_len, :]

                        beg_snt = data_end_index[snt_index]
                        snt_index = snt_index + 1

                    # [n_steps, batch_size, dimension]
                    # [n_steps, batch_size]
                    yield inp[:, :, :-1], np.reshape(inp[:, :, -1], newshape=[-1])

            raise IndexError
