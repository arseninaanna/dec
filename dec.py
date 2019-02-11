from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import numpy as np
from keras.initializers import RandomNormal
from keras.optimizers import SGD
from keras.layers import Dropout


def train_autoencoder(x):
    std = 0.01
    drop = 0.2
    def_inp = 28 * 28
    dim = [500, 500, 2000, 10]
    lr = 0.1
    batch_size = 256
    pre_iter = 50000
    ae_iter = 100000

    encoders = []  # e0, e1, e2, e3
    decoders = []  # d3, d2, d1, d0
    outputs = [x]  # x, e0x = e0(x), e1x = e1(e0x), e2x = e2(e1x), e3x = e3(e2x)

    iters_per_epoch = int(len(x) / batch_size)
    pt_epochs = max(int(pre_iter / iters_per_epoch), 1)
    ae_epochs = max(int(ae_iter / iters_per_epoch), 1)

    for i in range(len(dim)):
        e_activ = 'linear' if i == (len(dim) - 1) else 'relu'
        d_activ = 'linear' if i == 0 else 'relu'
        input_size = def_inp if i == 0 else dim[i - 1]
        input_shape = (input_size,)

        enc_i = Dense(dim[i], activation=e_activ, bias_initializer='zeros', input_shape=input_shape,
                      kernel_initializer=RandomNormal(mean=0.0, stddev=std, seed=None))
        encoders.append(enc_i)

        dec_i = Dense(input_size, activation=d_activ, bias_initializer='zeros',
                      kernel_initializer=RandomNormal(mean=0.0, stddev=std, seed=None))
        decoders.append(dec_i)

        model = Sequential([
            Dropout(drop, input_shape=input_shape),
            encoders[i],
            Dropout(drop),
            decoders[i]
        ])

        model.compile(loss='mse', optimizer=SGD(lr=lr, decay=0, momentum=0.9))
        out = model.fit(outputs[i], outputs[i], batch_size=batch_size, epochs=pt_epochs, verbose=1)
        outputs.append(out)

    autoencoder = Sequential()
    for i in range(len(encoders)):
        autoencoder.add(encoders[i])
    for i in range(len(decoders) - 1, -1, -1):
        autoencoder.add(decoders[i])

    autoencoder.compile(loss='mse', optimizer=SGD(lr=lr, decay=0, momentum=0.9))
    autoencoder.fit(x, x, batch_size=batch_size, epochs=ae_epochs)

    return encoders


def get_mnist(size=10000):
    np.random.seed(1234)  # set seed for deterministic ordering
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_all = np.concatenate((x_train, x_test), axis=0)
    Y = np.concatenate((y_train, y_test), axis=0)
    X = x_all.reshape(-1, x_all.shape[1] * x_all.shape[2])

    p = np.random.permutation(X.shape[0])
    X = X[p].astype(np.float32) * 0.02
    Y = Y[p]
    return X[-size:], Y[-size:]


if __name__ == '__main__':
    X, Y = get_mnist()

    encoders = train_autoencoder(X)
    print()
