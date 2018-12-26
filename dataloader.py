# coding=utf-8
from __future__ import print_function, division
import utils
import numpy as np
import json
import collections
import pickle as pkl
import os
import re

class DataSet(object):
    def __init__(self, labeled=True):
        self.images = []
        if labeled:
            self.labels = []
        self.pointer = 0
        self.amount = 0
        self.num_batches = 0
        self.batches = []

class DataLoader(object):
    def __init__(self, args, infer=False):
        np.random.seed(args['debug'].getint('seed'))
        self.args = args
        self.infer = infer
        print("Reading data files")
        self.read_data()
        print("Finish reading")
        self.create_batches()
        print("Created batches")

    def read_data(self):
        self.train_data = DataSet()
        self.test_data = DataSet()

        print('Reading training file')
        if not os.path.exists(self.args['paths']['train_file_name']):
            raise Exception('Training data file {} not found.'
                            .format(self.args['paths']['train_file_name']))
        self._read_data_set('train')

        print('Reading testing file')
        if not os.path.exists(self.args['paths']['test_file_name']):
            raise Exception('Testing data file {} not found.'
                            .format(self.args['paths']['test_file_name']))
        self._read_data_set('test')

        print("Begin preprocessing")
        self.preprocess()
        print("Finished preprocessing")

        self.batch_size = self.args['train'].getint('batch_size')
        self.train_data.amount = len(self.train_data.images)
        self.test_data.amount = len(self.test_data.images)

    def create_batches(self):
        print('Creating training data batches')
        self._create_data_set_batches(self.train_data)
        print('Creating testing data batches')
        self._create_data_set_batches(self.test_data)

    def _read_data_set(self, mode):
        #Assuming the dataset is small enough to load into RAM
        #Use h5py/tfrecord if the dataset is larger
        if mode == 'train':
            data_set = self.train_data
            data_file_name = self.args['paths']['train_file_name']
        elif mode == 'test':
            data_set = self.test_data
            data_file_name = self.args['paths']['test_file_name']
        else:
            raise ValueError('Incorrect mode: %s given' % mode)

        data = np.genfromtxt(data_file_name, skip_header=1)
        data_set.images = data[:,1:]
        data_set.labels = data[:,0]


    def preprocess(self):
        print('Begin preprocessing training data')
        self._preprocess_data_set(self.train_data)
        print('Begin reprocessing testing data')
        self._preprocess_data_set(self.test_data)

    def _create_data_set_batches(self, data_set):
        data_set.num_batches = data_set.amount // self.batch_size
        assert data_set.num_batches != 0, "Not enough data. Make batch_size smaller."
        self.shuffle_data(data_set)
        num_of_items = data_set.num_batches*self.args['train'].getint('batch_size')

        def split(x):
            return np.split(x[:num_of_items], data_set.num_batches)
        data_set.batches = list(zip(split(data_set.images),
                                       split(data_set.labels),
                                       ))
        self.reset_batch_pointer()

    @staticmethod
    def shuffle_data(data_set):
        shuffle_key = np.arange(data_set.amount)
        np.random.shuffle(shuffle_key)
        data_set.images = data_set.images[shuffle_key]
        data_set.labels = data_set.labels[shuffle_key]

    def next_train_batch(self):
        return self._next_batch('train')

    def next_test_batch(self):
        return self._next_batch('test')

    def _next_batch(self, mode):
        if mode == 'train':
            data_set = self.train_data
        elif mode == 'test':
            data_set = self.test_data
        else:
            raise ValueError('input argument must be "train" or "test". {} given'.format(mode))
        batch = data_set.batches[data_set.pointer]
        data_set.pointer += 1
        return batch, seq_length

    def reset_batch_pointer(self):
        self.train_data.pointer = 0

    def reset_test_batch_pointer(self):
        self.test_data.pointer = 0
