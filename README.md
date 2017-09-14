# Sisyphus


Using Minos to search for neural networks architecture & hyper parameters with genetic algorithms.  Comes with a nice UI and easy syntax.



## Getting Started

You'll want to have tensorflow and keras already setup, this assumes that you do.

Install sisy:
```
pip install pysisy==0.1
```

## Examples

The  [examples/](https://github.com/qorrect/sisy/examples/) directory tries to mimic https://github.com/fchollet/keras/blob/master/examples/ with added parameter searches.

A minimal example showing searching a variable number of units

```python
layout = [('Input',   {'units': 1000}),
          ('Dense',   {'units': range(200, 600), 'activation': 'relu'}),
          ('Dropout', {'rate': 0.5}),
          ('Output',  {'units': 1, 'activation': 'softmax'})]

run_sisy_experiment(layout, 'reuters_mlp', (x_train, y_train), (x_test, y_test),
                    epochs=5, batch_size=32, population_size=10, n_jobs=8,
                    loss='categorical_crossentropy', optimizer='adam', metric='acc', shuffle=False)

```