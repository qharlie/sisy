from keras.datasets import boston_housing
from sisy import run_sisy_experiment

layout = [('Input', {'units': 13, }),
          ('Dense', {'units': range(13, 26),  'kernel_initializer': 'normal', 'activation': ['relu','tanh','sigmoid']}),
          ('Dense', {'units': range(26, 100), 'kernel_initializer': 'normal', 'activation': ['relu','tanh','sigmoid']}),
          ('Output', {'units': 1, 'activation': 'linear', 'kernel_initializer': 'normal'})]

(X_train, y_train), (X_test, y_test) = boston_housing.load_data()
run_sisy_experiment(layout,
                    'regression',
                    (X_train, y_train),
                    (X_test, y_train),
                    fitness_type='FitnessMin',
                    loss='mean_squared_error',
                    metric='mean_squared_error',
                    optimizer='Adam',
                    n_jobs=8,
                    epochs=50,
                    batch_size=1,
                    mutation=10)
