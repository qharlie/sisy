'''Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
'''
from __future__ import print_function

import numpy as np
import keras
from keras.datasets import reuters
from keras.preprocessing.text import Tokenizer

from  sisy import run_sisy_experiment, frange

max_words = 1000

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words,
                                                         test_split=0.2)

num_classes = np.max(y_train) + 1
tokenizer = Tokenizer(num_words=max_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# ######################
#
# Compare with https://github.com/fchollet/keras/blob/master/examples/reuters_mlp.py
#
# Here is the original keras layout
#
# model = Sequential()
# model.add(Dense(512, input_shape=(max_words,)))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))
#
# ######################

        # Our Input size is the number of words in our reuters data we want to examine
layout = [('Input', {'units': 10000}),
        # 'units' we specify a range of nodes we want to try
        # 'activation' we specify a list of the activation types we want to try
        ('Dense', {'units': range(400, 600), 'activation': ['relu','tanh']}),
        # 'rate' is a f(loat)range from 0.2 to 0.8 , forced into a list
        ('Dropout', {'rate': list(frange(0.2,0.8))}),
        # Our Output size is the number of categories we want to classify the article into
        ('Output', {'units': 42, 'activation': 'softmax'})]

run_sisy_experiment(layout, 'sisy_reuters_mlp', (x_train, y_train), (x_test, y_test),
                  optimizer='adam',
                  metric='acc',
                  epochs=10,
                  batch_size=32,
                  n_jobs=8,
                  # 'devices' : Lets run this on the gpus 0 and 1
                  devices=['/gpu:0','/gpu:1'],
                  # 'population_size' : The number of different blueprints to try per generation.
                  population_size=10,
                  # 'generations' : The number of times to evolve the generations
                  # ( evolving here means taking the best blueprints and
                  # combining them to create ${population_size} more new blueprints)
                  generations=10,
                  loss='categorical_crossentropy',
                  # 'shuffle' : Defaults to true
                  shuffle=False)
