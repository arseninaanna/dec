import clustering as cl
import utils
import numpy as np
from clustering import Clustering
from keras.models import Sequential
from keras.optimizers import SGD


def get_model(encoder, x):
    clusters = 10
    learning_rate = 0.01
    momentum = 0.9

    output = encoder.predict(x)
    centroids, prediction = cl.get_centroids(output, clusters)
    print("DEC: initial centroids found")

    clustering_layer = Clustering(clusters, weights=centroids, prediction=prediction, name='clustering')
    model = Sequential([
        encoder,
        clustering_layer
    ])
    # model.compile(loss='kullback_leibler_divergence', optimizer='adadelta')
    model.compile(loss=cl.calculate_kl, optimizer=SGD(lr=learning_rate, momentum=momentum))

    return model


def get_slice(x, batch_number, batches_count, batch_size):
    step = batch_number % batches_count
    start = step * batch_size
    end = min((step + 1) * batch_size, x.shape[0])
    idx = np.arange(x.shape[0])

    return idx[start:end]


def target_prediction(dec_model, x, idx=None):
    if idx is not None:
        x = x[idx]

    q = dec_model.predict_on_batch(x)
    p = cl.p_stat(q)

    return p


def train_stat(dec_model, x, prev_pred, y=None):
    q = dec_model.predict(x, verbose=0)

    y_pred = q.argmax(1)
    delta = cl.labels_delta(prev_pred, y_pred) * 100
    acc = None
    if y is not None:
        acc = cl.cluster_acc(y, y_pred)[0]

    return y_pred, delta, acc


def train_dec(dec_model, x, y=None):
    threshold = 0.1
    batch_size = 256
    check_interval = 32

    size = x.shape[0]
    old_pred = dec_model.get_layer('clustering').initial_prediction
    batches_count = int(np.ceil(size / batch_size))

    def current_slice():
        return get_slice(x, batch_number, batches_count, batch_size)

    p = np.zeros(size)
    batch_number = 0
    while True:
        if batch_number % check_interval == 0:
            y_pred, delta, acc = train_stat(dec_model, x, old_pred, y)
            old_pred = y_pred

            p = target_prediction(dec_model, x)

            log = 'Iter %d, delta %.2f%%' % (batch_number, delta)
            if acc is not None:
                log += ', accuracy %.5f' % (acc)
            print(log)

            if batch_number != 0 and delta <= threshold:
                return

        idx = current_slice()
        # p = target_prediction(dec_model, x, idx)
        loss = dec_model.train_on_batch(x[idx], p[idx])

        batch_number += 1


if __name__ == '__main__':
    X, Y = utils.get_mnist(5000)
    print("MNIST loaded")

    encoder = utils.load_model('encoder')
    print("Encoder loaded")

    try:
        dec_model = utils.load_model('dec')
        print('DEC loaded')
    except EnvironmentError:
        dec_model = get_model(encoder, X)

        print('Training DEC')
        train_dec(dec_model, X, Y)

        print('Saving DEC')
        utils.save_model('dec', dec_model)

    print()
