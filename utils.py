import tensorflow as tf
import numpy as np
import pickle
import os.path as path
from clustering import Clustering
import keras.models


def get_mnist(size=None):
    np.random.seed(1234)  # set seed for deterministic ordering
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_all = np.concatenate((x_train, x_test), axis=0)
    Y = np.concatenate((y_train, y_test), axis=0)
    X = x_all.reshape(-1, x_all.shape[1] * x_all.shape[2])

    p = np.random.permutation(X.shape[0])
    X = X[p].astype(np.float32) * 0.02
    Y = Y[p]

    if size is not None:
        X = X[-size:]
        Y = Y[-size:]

    return X, Y


def load_model(name):
    filepath_1 = 'models/' + name + '.p'
    filepath_2 = 'models/' + name + '.h5'

    if path.exists(filepath_1):
        with open(filepath_1, 'rb') as fp:
            return pickle.load(fp)

    if path.exists(filepath_2):
        return keras.models.load_model(filepath_2, custom_objects={'Clustering': Clustering})

    raise EnvironmentError('Pretrained model ' + name + ' not found')


def save_model(name, model, mode=2):
    if mode == 1:
        filepath = 'models/' + name + '.p'
        with open(filepath, 'wb') as wf:
            pickle.dump(model, wf, protocol=pickle.HIGHEST_PROTOCOL)

    if mode == 2:
        filepath = 'models/' + name + '.h5'
        model.save(filepath)
