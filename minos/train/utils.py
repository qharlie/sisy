'''
Created on Feb 12, 2017

@author: julien
'''
import os
import logging
from os import path, makedirs
from os.path import join
from posix import access, W_OK
from random import Random
import traceback

import numpy as np

from numpy import ndarray
import tensorflow as tf
random = Random()


class Environment(object):

    def __init__(self, devices=None, n_jobs=None,
                 data_dir=None, tf_logging_level=logging.ERROR):
        self.devices = devices
        self.n_jobs = n_jobs
        if devices and n_jobs and not isinstance(n_jobs, list):
            self.n_jobs = [n_jobs for _ in devices]
        self.data_dir = data_dir or self._init_minos_dir()
        self.tf_logging_level = tf_logging_level

    def _init_minos_dir(self):
        base_dir = path.expanduser('~')
        if not access(base_dir, W_OK):
            base_dir = path.join('/tmp')
        minos_dir = join(base_dir, 'minos')
        if not path.exists(minos_dir):
            makedirs(minos_dir)
        return minos_dir


class CpuEnvironment(Environment):

    def __init__(self, n_jobs=os.environ["N_JOBS"], data_dir=None,
                 tf_logging_level=logging.ERROR):
        super().__init__(
            ['/cpu:0'],
            n_jobs,
            data_dir=data_dir,
            tf_logging_level=tf_logging_level)


class GpuEnvironment(Environment):

    def __init__(self, devices=['/gpu:0'], n_jobs=os.environ["N_JOBS"], data_dir=None,
                 tf_logging_level=logging.ERROR):
        super().__init__(
            devices,
            n_jobs,
            data_dir=data_dir,
            tf_logging_level=tf_logging_level)

#
# class SimpleBatchIterator(object):
#
#     def __init__(self, X, y, batch_size,
#                  X_transform=None, y_transform=None,
#                  autoloop=False, autorestart=False, preload=False,shuffle=True):
#         self.X = X
#         self.y = y
#         self.shuffle = shuffle
#         self.batch_size = batch_size
#         self.X_transform = X_transform
#         self.y_transform = y_transform
#         self.autoloop = autoloop
#         self.autorestart = autorestart
#         self.index = 0
#         self.preload = preload
#         self.batch_size = batch_size
#         self.sample_count = len(X)
#         self.samples_per_epoch = self.sample_count / batch_size
#         self.X, self.y = create_batches(X, y, batch_size)
#
#     def _transform_data(self, X, y):
#         if self.X_transform:
#             X = self.X_transform.fit_transform(X)
#         if self.y_transform:
#             y = self.y_transform.fit_transform(y)
#         return numpy.asarray(X), numpy.asarray(y)
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         try:
#             if not self.shuffle:
#                 if self.index >= len(self.X):
#                     if self.autorestart or self.autoloop:
#                         self.index = 0
#                     if self.autorestart or not self.autoloop:
#                         return None
#                 X = self.X[self.index+self.index+self.batch_size]
#                 y = self.y[self.index+self.index+self.batch_size]
#                 #print("HERE in SIMPLE {} ({})\n\nReturning X, Y\n\n{}\n\n{}".format(self.index,self.batch_size, X.shape, y.shape))
#                 self.index += self.batch_size
#                 return X, y
#             else:
#                 if self.index >= len(self.X):
#                     shuffle_batch(self.X, self.y)
#                     if self.autorestart or self.autoloop:
#                         self.index = 0
#                     if self.autorestart or not self.autoloop:
#                         return None
#                 X, y = self._transform_data(
#                     self.X[self.index],
#                     self.y[self.index])
#                 shuffle_batch(X, y)
#                 self.index += 1
#                 #print("HERE in __next__ DEFAULT {} ({})\n\nReturning X, Y\n\n{}\n\n{}".format(self.index,self.batch_size, X.shape, y.shape))
#                 return X, y
#
#         except Exception as ex:
#             logging.error('Error while iterating %s' % str(ex))
#             try:
#                 logging.error(traceback.format_exc())
#             finally:
#                 pass
#             raise ex


NL_COUNTER = 0
def show_progress(c=1,break_point = 100):
    global NL_COUNTER
    import sys
    if NL_COUNTER % break_point == 0 :
        print(NL_COUNTER)
    else:
        sys.stdout.write('.')

    NL_COUNTER += c


def generator(features, labels, batch_size):
 # Create empty arrays to contain batch of features and labels#
 batch_features = np.zeros((batch_size, 64))
 batch_labels = np.zeros((batch_size,1))
 while True:
   for i in range(batch_size):
        index= random.randint(0,len(features)-1)
        batch_labels[i] = labels[index]
        batch_features[i] = features[index]
   yield batch_features, batch_labels

class SimpleBatchIterator(object):

    def __init__(self, X, y, batch_size,
                 X_transform=None, y_transform=None,
                 autoloop=False, autorestart=False, preload=False,shuffle=True):
        self.X = X
        self.y = y
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.X_transform = X_transform
        self.y_transform = y_transform
        self.autoloop = autoloop
        self.autorestart = autorestart
        self.index = 0
        self.preload = preload
        self.batch_size = batch_size
        self.sample_count = len(X)
        self.samples_per_epoch = int(self.sample_count / batch_size)

    def __next__(self):
        try:
            i = self.index
            end = self.index + self.batch_size
            if i+end >= len(self.X):
                if self.autorestart or self.autoloop:
                    self.index = 0
                    i = 0
                    end = i + self.batch_size

                if self.autorestart or not self.autoloop:
                    return None
            X = self.X[i:end]
            y = self.y[i:end]
            print("HERE in SIMPLE {} ({})\n\nReturning X, Y\n\n{}\n\n{}".format(self.index,self.batch_size, X.shape, y.shape))

            self.index = end
            show_progress()
            return X, y

        except Exception as ex:
            logging.error('Error while iterating %s' % str(ex))
            try:
                logging.error(traceback.format_exc())
            finally:
                pass
            raise ex


def create_batches(X, y, batch_size):
    X = [
        X[i:i + batch_size]
        for i in range(0, len(X), batch_size)]
    y = [
        y[i:i + batch_size]
        for i in range(0, len(y), batch_size)]
    return X, y


def shuffle_batches(X_batches, y_batches):
    for X, y in zip(X_batches, y_batches):
        shuffle_batch(X, y)


def shuffle_batch(X, y):
    for i in range(len(X)):
        swap_idx = random.randint(0, i)
        swap(X, i, swap_idx)
        swap(y, i, swap_idx)


def swap(values, idx1, idx2):
    if isinstance(values, ndarray):
        swap = np.copy(values[idx2])
        values[idx2] = np.copy(values[idx1])
        values[idx1] = swap
    else:
        swap = values[idx2]
        values[idx2] = values[idx1]
        values[idx1] = swap
