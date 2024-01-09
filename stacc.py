import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from sisy import frange, run_sisy_experiment
from multiprocessing import freeze_support
import time
import random
import os
import numpy as np
import pandas as pd
import ccxt
import logging
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from deap import base, creator, tools, algorithms

# Utility function to create sequences for LSTM model
def create_sequences(data, n_steps, future_steps=None):
    X, y, Xv, Yv = [], [], [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data) - 1:
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        if i < len(data) / 10 * 9:
            X.append(seq_x)
            y.append(seq_y)
        else:
            Xv.append(seq_x)
            Yv.append(seq_y)
    return np.array(X), np.array(y), np.array(Xv), np.array(Yv)
import json 

# Fetch historical data from Binance
def fetch_historical_data(symbol, timeframe, lookback_days, binance_client):
    since = int(time.time() * 1000) - 24 * 60 * 60 * 1000 * lookback_days
    latest_date = since
    histories = []
    if os.path.exists('histories.txt'):
        with open('histories.txt', 'r') as f:
            histories = json.loads(f.read())
            latest_date = histories[-1][-1][0]
                
    while latest_date < int(time.time() * 1000) - 24 * 60 * 60 * 1000:
        history = binance_client.fetch_ohlcv(symbol, timeframe, since=latest_date, limit=1000)
        latest_date = history[-1][0]
        histories.append(history)
        date_human_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(latest_date / 1000))
        logging.info(f"Fetched data till {date_human_readable}")
    with open('histories.txt', 'w') as f:
        f.write(json.dumps(histories))
    return [item for sublist in histories for item in sublist]

# Normalize and reshape data for LSTM model
def prepare_data(data, n_steps, future_steps):
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(data[['Close']])
    X, y, Xv, Yv = create_sequences(normalized_data, n_steps, future_steps)
    X = np.reshape(X, (X.shape[0], X.shape[1]))
    Xv = np.reshape(Xv, (Xv.shape[0], Xv.shape[1]))
    return X, y, Xv, Yv, scaler

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from deap import base, creator, tools, algorithms
import random
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def build_model(layout):
    model = Sequential()

    for layer_name, layer_params in layout:
        if layer_name == 'Input' or layer_name == 'Dense':
            model.add(Dense(layer_params['units'], activation=layer_params['activation']))
        elif layer_name == 'Dropout':
            model.add(Dropout(layer_params['rate']))
        elif layer_name == 'Output':
            model.add(Dense(layer_params['units'], activation=layer_params['activation']))

    return model
# Genetic Algorithm evaluation function
def evaluate_individual(individual, input_shape, X, y):
    units, dropout, optimizer = individual
    model = build_lstm_model(input_shape, int(units), dropout, optimizer)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=0)
    loss = model.evaluate(X_val, y_val, verbose=0)
    return (loss,)
# Genetic Algorithm setup

# model.summary()
#
# model.compile(loss='categorical_crossentropy',
#               optimizer=RMSprop(),
#               metrics=['accuracy'])
#
# history = model.fit(x_train, y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=0,
#                     validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test, verbose=0)
# logging.info('Test loss:', score[0])
# logging.info('Test accuracy:', score[1])
if __name__ == '__main__':
    freeze_support()


    n_steps, future_steps = 100, 100
    binance_client = ccxt.binance({'enableRateLimit': False, 'rateLimit': 20, 
        'apiKey': "",
        "secret": ""})
    historical_data = fetch_historical_data('SOL/USDT', '1m', 365, binance_client)
    data_df = pd.DataFrame(historical_data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    trainX, trainY, valX, valY, scaler = prepare_data(data_df, 784, 784)

    n_jobs = int(os.environ.get('N_JOBS', 8))
    batch_size = int(os.environ.get('BATCH_SIZE', 128))
    num_classes = 10
    epochs = 10

    (x_train, y_train), (x_test, y_test) = (trainX, trainY), (valX, valY)

    #x_train = x_train.reshape(60000, 784)
    #x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # model = Sequential()
    # model.add(Dense(512, activation='relu', input_shape=(784,)))
    # model.add(Dropout(0.2))
    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(10, activation='softmax'))
    #
    def frange(start, stop, step=0.1):
        while start < stop:
            yield start
            start += step

    layout = [
        ('Input', {'units': 784, 'activation': 'relu'}),
        ('Dense', {
            'units': list(range(300, 700)),
            'activation': ['relu', 'tanh', 'sigmoid']
        }),
        ('Dropout', {
            'rate': list(frange(0.1, 0.9))
        }),
        ('Dense', {
            'units': list(range(300, 700)),
            'activation': ['relu', 'tanh', 'sigmoid']
        }),
        ('Dropout', {
            'rate': list(frange(0.1, 0.9))
        }),
        ('Output', {
            'units': 10,
            'activation': 'softmax'
        })
    ]

    run_sisy_experiment(layout,
                        "mnist_mlp",
                        (x_train, y_train),
                        (x_test, y_test),
                        batch_size=batch_size,
                        epochs=epochs,
                        shuffle=False,
                        n_jobs=n_jobs)