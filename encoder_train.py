from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import RandomNormal
from keras.optimizers import SGD
from keras.layers import Dropout
from keras.callbacks import LearningRateScheduler
import utils

TESTING = False


def init_step_decay(learning_rate, lr_epoch_update):
    return lambda epoch: learning_rate / (10 ** int(epoch / lr_epoch_update))


def train_autoencoder(x):
    std = 0.01
    drop = 0.2
    momentum = 0.9
    def_inp = 28 * 28
    dim = [500, 500, 2000, 10]
    lr = 0.1
    batch_size = 256
    pre_iter = 50000
    ae_iter = 100000
    iters_lr_update = 20000

    if TESTING:
        dim = [100, 100, 500, 10]
        pre_iter = 10000
        ae_iter = 20000

    encoders = []  # e0, e1, e2, e3
    decoders = []  # d3, d2, d1, d0
    outputs = [x]  # x, e0x = e0(x), e1x = e1(e0x), e2x = e2(e1x), e3x = e3(e2x)

    iters_per_epoch = int(len(x) / batch_size)
    pt_epochs = max(int(pre_iter / iters_per_epoch), 1)
    ae_epochs = max(int(ae_iter / iters_per_epoch), 1)

    lr_epoch_update = max(int(iters_lr_update / iters_per_epoch), 1)
    lr_schedule = LearningRateScheduler(init_step_decay(lr, lr_epoch_update))

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

        model.compile(loss='mse', optimizer=SGD(lr=lr, decay=0, momentum=momentum))
        model.fit(outputs[i], outputs[i], batch_size=batch_size, epochs=pt_epochs, callbacks=[lr_schedule], verbose=1)

        enc_model = Sequential([enc_i])
        enc_model.compile(loss='mse', optimizer=SGD(lr=lr, decay=0, momentum=momentum))
        out = enc_model.predict(outputs[i])

        outputs.append(out)

    autoencoder = Sequential(encoders + decoders[::-1])
    autoencoder.compile(loss='mse', optimizer=SGD(lr=lr, decay=0, momentum=momentum))
    autoencoder.fit(x, x, batch_size=batch_size, epochs=ae_epochs)

    encoder = Sequential(encoders)
    encoder.compile(loss='mse', optimizer=SGD(lr=lr, decay=0, momentum=momentum))

    return encoder


if __name__ == '__main__':
    X, Y = utils.get_mnist()
    if TESTING:
        X, Y = utils.get_mnist(35000)

    encoder = train_autoencoder(X)
    utils.save_model('encoder', encoder)

    print("Encoder trained and stored")
