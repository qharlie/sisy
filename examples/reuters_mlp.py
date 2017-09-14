'''Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
'''
from __future__ import print_function

import numpy as np
import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer

from api import run_sisy_experiment

max_words = 1000
# batch_size = 32
# epochs = 5

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words,
                                                         test_split=0.2)

num_classes = np.max(y_train) + 1
tokenizer = Tokenizer(num_words=max_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

layout = [('Input', {'units': max_words}),
          ('Dense', {'units': range(400, 600), 'activation': 'relu'}),
          ('Dropout', {'rate': 0.5}),
          ('Output', {'units': num_classes, 'activation': 'softmax'})]

run_sisy_experiment(layout, 'reuters_mlp', (x_train, y_train), (x_test, y_test),
                    epochs=5, batch_size=32, population_size=10, n_jobs=8,
                    loss='categorical_crossentropy', optimizer='adam', metric='acc', shuffle=False)
