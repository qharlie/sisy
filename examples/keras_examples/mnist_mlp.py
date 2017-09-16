import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from sisy import frange, run_sisy_experiment

batch_size = 128
num_classes = 10
epochs = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
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

run_sisy_experiment(layout, "mnist_mlp", (x_train, y_train), (x_test, y_test),
                    batch_size=batch_size,
                    epochs=epochs,
                    shuffle=False,
                    n_jobs=8)

# model.summary()
#
# model.compile(loss='categorical_crossentropy',
#               optimizer=RMSprop(),
#               metrics=['accuracy'])
#
# history = model.fit(x_train, y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
